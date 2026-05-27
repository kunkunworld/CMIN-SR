from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_inverse.data import PROCESS_CODE_TO_NAME, SpectralRepresentationDatasetConfig, generate_spectral_representation_dataset
from mrw_inverse.models import CMINSRv3Model


CHECKPOINT_PATH = ROOT / "checkpoints" / "cmin" / "cmin_sr_v3_synthetic.pt"
REPORT_DIR = ROOT / "outputs" / "reports" / "cmin_sr_v3_eval"
TABLE_DIR = ROOT / "outputs" / "tables" / "cmin_sr_v3_eval"
FIG_DIR = ROOT / "outputs" / "figures" / "cmin_sr_v3_eval"


def _dataset(length: int, num_samples: int, seed: int):
    return generate_spectral_representation_dataset(
        SpectralRepresentationDatasetConfig(length=length, num_samples=num_samples, seed=seed, mrw_ratio=0.20, low_lambda2_mrw_ratio=0.10)
    )


def _eval_length(model: CMINSRv3Model, length: int, num_samples: int, seed: int, device: str):
    ds = _dataset(length, num_samples, seed)
    rows = []
    for i in range(num_samples):
        x = torch.tensor(ds["x"][i : i + 1], dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(x)
        proc = PROCESS_CODE_TO_NAME[int(ds["process_code"][i])]
        rows.append(
            {
                "T": length,
                "sample_id": int(ds["sample_id"][i]),
                "pair_id": int(ds["pair_id"][i]),
                "process_type": proc,
                "true_H": float(ds["H_true"][i, 0]),
                "pred_H_proj": float(out["H_proj"].squeeze().cpu()),
                "true_lambda2": float(ds["lambda2_true"][i, 0]),
                "pred_lambda2_proj": float(out["lambda2_proj"].squeeze().cpu()),
                "pred_p_scaling": float(out["p_scaling"].squeeze().cpu()),
                "pred_p_curved": float(out["p_curved"].squeeze().cpu()),
                "pred_p_mrw": float(out["p_mrw"].squeeze().cpu()),
                "pred_p_mono": float(out["p_mono"].squeeze().cpu()),
                "pred_boundary_mrw_score": float(out["boundary_mrw_score"].squeeze().cpu()),
                "pred_residual_norm": float(out["residual_norm"].squeeze().cpu()),
                "pred_mono_residual_norm": float(out["mono_residual_norm"].squeeze().cpu()),
                "pred_gain": float(out["mrw_vs_mono_gain"].squeeze().cpu()),
                "pred_curvature_score": float(out["curvature_score"].squeeze().cpu()),
                "pred_curvature_sig": float(out["curvature_significance"].squeeze().cpu()),
                "pred_linearity_score": float(out["linearity_score"].squeeze().cpu()),
                "pred_tail_instability": float(out["tail_instability"].squeeze().cpu()),
            }
        )
    df = pd.DataFrame(rows)
    pair_df = df[df["pair_id"] >= 0]
    p_gap = np.nan
    if not pair_df.empty:
        pivot = pair_df.pivot_table(index="pair_id", columns="process_type", values=["pred_p_mrw"], aggfunc="first")
        if ("pred_p_mrw", "MRW") in pivot.columns and ("pred_p_mrw", "Shuffled MRW") in pivot.columns:
            p_gap = float((pivot[("pred_p_mrw", "MRW")] - pivot[("pred_p_mrw", "Shuffled MRW")]).mean())
    return df, {
        "T": length,
        "train_length": int(length in {512, 1024}),
        "fgn_p_scaling": float(df.loc[df["process_type"] == "fGn", "pred_p_scaling"].mean()),
        "fgn_p_curved": float(df.loc[df["process_type"] == "fGn", "pred_p_curved"].mean()),
        "fgn_p_mrw": float(df.loc[df["process_type"] == "fGn", "pred_p_mrw"].mean()),
        "gaussian_p_mrw": float(df.loc[df["process_type"] == "iid Gaussian", "pred_p_mrw"].mean()),
        "mrw_p_mrw": float(df.loc[df["process_type"] == "MRW", "pred_p_mrw"].mean()),
        "mrw_p_curved": float(df.loc[df["process_type"] == "MRW", "pred_p_curved"].mean()),
        "low_mrw_p_mrw": float(df.loc[df["process_type"] == "Low-lambda2 MRW", "pred_p_mrw"].mean()),
        "student_t_p_mrw": float(df.loc[df["process_type"] == "iid Student-t", "pred_p_mrw"].mean()),
        "regime_p_mrw": float(df.loc[df["process_type"] == "Regime-switching Gaussian", "pred_p_mrw"].mean()),
        "mrw_vs_shuffled_p_mrw_gap": p_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CMIN-SR v3.")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--t-eval", nargs="*", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--seed", type=int, default=9090)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        out = {"status": "missing_checkpoint", "checkpoint": str(ckpt.relative_to(ROOT) if ckpt.is_absolute() else ckpt)}
        (REPORT_DIR / "cmin_sr_v3_eval_warning.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return

    state = torch.load(ckpt, map_location="cpu")
    model = CMINSRv3Model()
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state, strict=False)
    device = args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    metric_rows = []
    process_rows = []
    scatter_rows = []
    for i, length in enumerate(args.t_eval):
        df, metrics = _eval_length(model, length, args.num_samples, args.seed + 31 * i, device)
        df.to_csv(TABLE_DIR / f"predictions_T{length}.csv", index=False)
        metric_rows.append(metrics)
        scatter_rows.append(df.assign(T=length))
        proc = (
            df.groupby("process_type")
            .agg(
                mean_p_scaling=("pred_p_scaling", "mean"),
                mean_p_curved=("pred_p_curved", "mean"),
                mean_p_mrw=("pred_p_mrw", "mean"),
                mean_p_mono=("pred_p_mono", "mean"),
                mean_boundary_mrw_score=("pred_boundary_mrw_score", "mean"),
                mean_lambda2_proj=("pred_lambda2_proj", "mean"),
                mean_residual_norm=("pred_residual_norm", "mean"),
                mean_mono_residual_norm=("pred_mono_residual_norm", "mean"),
                mean_gain=("pred_gain", "mean"),
                mean_curvature_sig=("pred_curvature_sig", "mean"),
                mean_linearity_score=("pred_linearity_score", "mean"),
                mean_tail_instability=("pred_tail_instability", "mean"),
            )
            .reset_index()
        )
        proc["T"] = length
        process_rows.extend(proc.to_dict(orient="records"))

    metrics_df = pd.DataFrame(metric_rows)
    process_df = pd.DataFrame(process_rows)
    all_preds = pd.concat(scatter_rows, ignore_index=True)
    metrics_df.to_csv(TABLE_DIR / "metrics_by_T.csv", index=False)
    process_df.to_csv(TABLE_DIR / "process_by_T.csv", index=False)

    plot_df = all_preds[all_preds["T"] == 1024] if (all_preds["T"] == 1024).any() else all_preds
    process_colors = {
        "MRW": "#1f77b4",
        "Low-lambda2 MRW": "#17becf",
        "Shuffled MRW": "#7f7f7f",
        "fGn": "#2ca02c",
        "iid Gaussian": "#98df8a",
        "iid Student-t": "#d62728",
        "GARCH(1,1)": "#ff7f0e",
        "Regime-switching Gaussian": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(6.6, 5.0), constrained_layout=True)
    for proc, sub in plot_df.groupby("process_type"):
        ax.scatter(sub["pred_p_scaling"], sub["pred_p_mrw"], s=18, alpha=0.6, label=proc, color=process_colors.get(proc))
    ax.set_xlabel("p_scaling")
    ax.set_ylabel("p_MRW")
    ax.set_title("v3: p_scaling vs p_MRW")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(FIG_DIR / "p_scaling_vs_p_mrw.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 5.0), constrained_layout=True)
    for proc, sub in plot_df.groupby("process_type"):
        ax.scatter(sub["pred_p_curved"], sub["pred_p_mrw"], s=18, alpha=0.6, label=proc, color=process_colors.get(proc))
    ax.set_xlabel("p_curved")
    ax.set_ylabel("p_MRW")
    ax.set_title("v3: p_curved vs p_MRW")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(FIG_DIR / "p_curved_vs_p_mrw.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 5.0), constrained_layout=True)
    for proc, sub in plot_df.groupby("process_type"):
        ax.scatter(sub["pred_residual_norm"], sub["pred_mono_residual_norm"], s=18, alpha=0.6, label=proc, color=process_colors.get(proc))
    lim = max(float(plot_df["pred_residual_norm"].max()), float(plot_df["pred_mono_residual_norm"].max()))
    ax.plot([0, lim], [0, lim], linestyle="--", color="black", linewidth=1.0)
    ax.set_xlabel("MRW residual")
    ax.set_ylabel("Monofractal residual")
    ax.set_title("v3: MRW vs monofractal residual")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(FIG_DIR / "mrw_vs_mono_residual_scatter.png", dpi=220)
    plt.close(fig)

    mrw_boundary = all_preds[all_preds["process_type"].isin(["MRW", "Low-lambda2 MRW"])].copy()
    if not mrw_boundary.empty:
        fig, ax = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
        ax.scatter(mrw_boundary["true_lambda2"], mrw_boundary["pred_p_mrw"], s=18, alpha=0.6, label="p_MRW")
        ax.scatter(mrw_boundary["true_lambda2"], mrw_boundary["pred_p_curved"], s=18, alpha=0.6, label="p_curved")
        ax.scatter(mrw_boundary["true_lambda2"], mrw_boundary["pred_boundary_mrw_score"], s=18, alpha=0.6, label="boundary")
        ax.set_xlabel("true lambda2")
        ax.set_ylabel("score")
        ax.set_title("v3 boundary-MRW transition")
        ax.legend()
        fig.savefig(FIG_DIR / "boundary_mrw_transition.png", dpi=220)
        plt.close(fig)

    report = REPORT_DIR / "cmin_sr_v3_eval_summary.md"
    report.write_text(
        "\n".join(
            [
                "# CMIN-SR v3 Evaluation",
                "",
                f"- Checkpoint: `{ckpt.relative_to(ROOT) if ckpt.is_absolute() else ckpt}`",
                "",
                "## Metrics By T",
                "",
                metrics_df.to_csv(index=False),
                "",
                "## Process Means By T",
                "",
                process_df.to_csv(index=False),
            ]
        ),
        encoding="utf-8",
    )
    meta = {
        "metrics_by_T": str((TABLE_DIR / "metrics_by_T.csv").relative_to(ROOT)),
        "process_by_T": str((TABLE_DIR / "process_by_T.csv").relative_to(ROOT)),
        "report": str(report.relative_to(ROOT)),
    }
    (REPORT_DIR / "cmin_sr_v3_eval_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
