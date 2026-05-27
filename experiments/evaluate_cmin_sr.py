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
from mrw_inverse.models import CMINSRModel


CHECKPOINT_PATH = ROOT / "checkpoints" / "cmin" / "cmin_sr_synthetic.pt"
REPORT_DIR = ROOT / "outputs" / "reports" / "cmin_sr_eval"
TABLE_DIR = ROOT / "outputs" / "tables" / "cmin_sr_eval"
FIG_DIR = ROOT / "outputs" / "figures" / "cmin_sr_eval"


def _eval_length(model: CMINSRModel, length: int, num_samples: int, seed: int, device: str) -> tuple[pd.DataFrame, dict[str, float]]:
    ds = generate_spectral_representation_dataset(SpectralRepresentationDatasetConfig(length=length, num_samples=num_samples, seed=seed))
    rows = []
    for i in range(num_samples):
        x = torch.tensor(ds["x"][i : i + 1], dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(x)
        proc = PROCESS_CODE_TO_NAME[int(ds["process_code"][i])]
        zeta_diff = np.abs(out["zeta_emp"].squeeze(0).cpu().numpy() - ds["zeta_target"][i])
        zeta_mae = float(np.nanmean(zeta_diff)) if np.isfinite(zeta_diff).any() else float("nan")
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
                "pred_p_mrw": float(out["p_mrw"].squeeze().cpu()),
                "pred_residual_norm": float(out["residual_norm"].squeeze().cpu()),
                "pred_stability": float(out["spectrum_stability"].squeeze().cpu()),
                "pred_tail_instability": float(out["tail_instability"].squeeze().cpu()),
                "zeta_mae": zeta_mae,
            }
        )
    df = pd.DataFrame(rows)
    mrw = df["process_type"] == "MRW"
    pair_df = df[df["pair_id"] >= 0]
    p_gap = np.nan
    r_gap = np.nan
    l_gap = np.nan
    if not pair_df.empty:
        pivot = pair_df.pivot_table(index="pair_id", columns="process_type", values=["pred_p_mrw", "pred_residual_norm", "pred_lambda2_proj"], aggfunc="first")
        if ("pred_p_mrw", "MRW") in pivot.columns and ("pred_p_mrw", "Shuffled MRW") in pivot.columns:
            p_gap = float((pivot[("pred_p_mrw", "MRW")] - pivot[("pred_p_mrw", "Shuffled MRW")]).mean())
        if ("pred_residual_norm", "MRW") in pivot.columns and ("pred_residual_norm", "Shuffled MRW") in pivot.columns:
            r_gap = float((pivot[("pred_residual_norm", "Shuffled MRW")] - pivot[("pred_residual_norm", "MRW")]).mean())
        if ("pred_lambda2_proj", "MRW") in pivot.columns and ("pred_lambda2_proj", "Shuffled MRW") in pivot.columns:
            l_gap = float((pivot[("pred_lambda2_proj", "MRW")] - pivot[("pred_lambda2_proj", "Shuffled MRW")]).mean())
    metrics = {
        "T": length,
        "train_length": int(length in {512, 1024}),
        "mrw_mae_H_proj": float(np.mean(np.abs(df.loc[mrw, "pred_H_proj"] - df.loc[mrw, "true_H"]))) if mrw.any() else np.nan,
        "mrw_mae_lambda2_proj": float(np.mean(np.abs(df.loc[mrw, "pred_lambda2_proj"] - df.loc[mrw, "true_lambda2"]))) if mrw.any() else np.nan,
        "mrw_mean_p_scaling": float(df.loc[mrw, "pred_p_scaling"].mean()) if mrw.any() else np.nan,
        "mrw_mean_p_mrw": float(df.loc[mrw, "pred_p_mrw"].mean()) if mrw.any() else np.nan,
        "mrw_mean_residual_norm": float(df.loc[mrw, "pred_residual_norm"].mean()) if mrw.any() else np.nan,
        "mrw_vs_shuffled_p_mrw_gap": p_gap,
        "mrw_vs_shuffled_residual_gap": r_gap,
        "mrw_vs_shuffled_lambda2_gap": l_gap,
    }
    return df, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CMIN-SR across multiple lengths.")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--t-eval", nargs="*", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--num-samples", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=6060)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        out = {
            "status": "missing_checkpoint",
            "checkpoint": str(ckpt.relative_to(ROOT) if ckpt.is_absolute() else ckpt),
            "next_step": "Run conda run -n for_codex python experiments/train_cmin_sr.py first.",
        }
        (REPORT_DIR / "cmin_sr_eval_warning.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return

    state = torch.load(ckpt, map_location="cpu")
    model = CMINSRModel()
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state, strict=False)
    device = args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    metric_rows = []
    process_rows = []
    pred_paths = []
    for i, length in enumerate(args.t_eval):
        df, metrics = _eval_length(model, length, args.num_samples, args.seed + 37 * i, device)
        path = TABLE_DIR / f"predictions_T{length}.csv"
        df.to_csv(path, index=False)
        pred_paths.append(path)
        metric_rows.append(metrics)
        proc = (
            df.groupby("process_type")
            .agg(
                mean_H_proj=("pred_H_proj", "mean"),
                mean_lambda2_proj=("pred_lambda2_proj", "mean"),
                mean_p_scaling=("pred_p_scaling", "mean"),
                mean_p_mrw=("pred_p_mrw", "mean"),
                mean_residual_norm=("pred_residual_norm", "mean"),
                mean_stability=("pred_stability", "mean"),
                mean_tail_instability=("pred_tail_instability", "mean"),
                mean_zeta_mae=("zeta_mae", "mean"),
            )
            .reset_index()
        )
        proc["T"] = length
        process_rows.extend(proc.to_dict(orient="records"))

    metrics_df = pd.DataFrame(metric_rows)
    process_df = pd.DataFrame(process_rows)
    metrics_path = TABLE_DIR / "metrics_by_T.csv"
    process_path = TABLE_DIR / "process_by_T.csv"
    metrics_df.to_csv(metrics_path, index=False)
    process_df.to_csv(process_path, index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(metrics_df["T"], metrics_df["mrw_vs_shuffled_p_mrw_gap"], marker="o", label="p_MRW gap")
    ax.plot(metrics_df["T"], metrics_df["mrw_vs_shuffled_residual_gap"], marker="s", label="residual gap")
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xlabel("T")
    ax.set_ylabel("gap")
    ax.set_title("CMIN-SR surrogate gaps by T")
    ax.legend()
    gap_fig = FIG_DIR / "surrogate_gaps_by_T.png"
    fig.savefig(gap_fig, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for proc in ["MRW", "fGn", "iid Gaussian", "iid Student-t", "GARCH(1,1)", "Regime-switching Gaussian"]:
        sub = process_df[process_df["process_type"] == proc].sort_values("T")
        if not sub.empty:
            ax.plot(sub["T"], sub["mean_p_mrw"], marker="o", label=proc)
    ax.set_xlabel("T")
    ax.set_ylabel("mean p_MRW")
    ax.set_title("CMIN-SR p_MRW by process and T")
    ax.legend(fontsize=8)
    pmrw_fig = FIG_DIR / "p_mrw_by_process_T.png"
    fig.savefig(pmrw_fig, dpi=220)
    plt.close(fig)

    summary_lines = [
        "# CMIN-SR Evaluation",
        "",
        f"- Checkpoint: `{ckpt.relative_to(ROOT) if ckpt.is_absolute() else ckpt}`",
        "- Train lengths: `512, 1024`",
        "- OOD stress lengths: `256, 2048`",
        "",
        "## Metrics By T",
        "",
        metrics_df.to_csv(index=False),
        "",
        "## Process Means By T",
        "",
        process_df.to_csv(index=False),
    ]
    report_path = REPORT_DIR / "cmin_sr_eval_summary.md"
    report_path.write_text("\n".join(summary_lines), encoding="utf-8")
    meta = {
        "metrics_by_T": str(metrics_path.relative_to(ROOT)),
        "process_by_T": str(process_path.relative_to(ROOT)),
        "gap_figure": str(gap_fig.relative_to(ROOT)),
        "pmrw_figure": str(pmrw_fig.relative_to(ROOT)),
        "report": str(report_path.relative_to(ROOT)),
    }
    (REPORT_DIR / "cmin_sr_eval_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
