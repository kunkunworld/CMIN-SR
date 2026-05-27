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
from mrw_inverse.models import CMINSRv2Model


CHECKPOINT_PATH = ROOT / "checkpoints" / "cmin" / "cmin_sr_v2_synthetic.pt"
REPORT_DIR = ROOT / "outputs" / "reports" / "cmin_sr_v2_eval"
TABLE_DIR = ROOT / "outputs" / "tables" / "cmin_sr_v2_eval"
FIG_DIR = ROOT / "outputs" / "figures" / "cmin_sr_v2_eval"


def _dataset(length: int, num_samples: int, seed: int):
    return generate_spectral_representation_dataset(
        SpectralRepresentationDatasetConfig(length=length, num_samples=num_samples, seed=seed, mrw_ratio=0.20, low_lambda2_mrw_ratio=0.10)
    )


def _eval_length(model: CMINSRv2Model, length: int, num_samples: int, seed: int, device: str):
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
                "pred_p_mrw": float(out["p_mrw"].squeeze().cpu()),
                "pred_p_mono": float(out["p_mono"].squeeze().cpu()),
                "pred_residual_norm": float(out["residual_norm"].squeeze().cpu()),
                "pred_mono_residual_norm": float(out["mono_residual_norm"].squeeze().cpu()),
                "pred_gain": float(out["mrw_vs_mono_gain"].squeeze().cpu()),
                "pred_curvature_sig": float(out["curvature_significance"].squeeze().cpu()),
                "pred_tail_instability": float(out["tail_instability"].squeeze().cpu()),
            }
        )
    df = pd.DataFrame(rows)
    pair_df = df[df["pair_id"] >= 0]
    p_gap = np.nan
    if not pair_df.empty:
        pivot = pair_df.pivot_table(index="pair_id", columns="process_type", values=["pred_p_mrw", "pred_gain"], aggfunc="first")
        if ("pred_p_mrw", "MRW") in pivot.columns and ("pred_p_mrw", "Shuffled MRW") in pivot.columns:
            p_gap = float((pivot[("pred_p_mrw", "MRW")] - pivot[("pred_p_mrw", "Shuffled MRW")]).mean())
    return df, {
        "T": length,
        "train_length": int(length in {512, 1024}),
        "fgn_p_scaling": float(df.loc[df["process_type"] == "fGn", "pred_p_scaling"].mean()),
        "fgn_p_mrw": float(df.loc[df["process_type"] == "fGn", "pred_p_mrw"].mean()),
        "gaussian_p_mrw": float(df.loc[df["process_type"] == "iid Gaussian", "pred_p_mrw"].mean()),
        "mrw_p_mrw": float(df.loc[df["process_type"] == "MRW", "pred_p_mrw"].mean()),
        "low_mrw_p_mrw": float(df.loc[df["process_type"] == "Low-lambda2 MRW", "pred_p_mrw"].mean()),
        "student_t_p_mrw": float(df.loc[df["process_type"] == "iid Student-t", "pred_p_mrw"].mean()),
        "regime_p_mrw": float(df.loc[df["process_type"] == "Regime-switching Gaussian", "pred_p_mrw"].mean()),
        "mrw_vs_shuffled_p_mrw_gap": p_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CMIN-SR v2.")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--t-eval", nargs="*", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--seed", type=int, default=8080)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        out = {"status": "missing_checkpoint", "checkpoint": str(ckpt.relative_to(ROOT) if ckpt.is_absolute() else ckpt)}
        (REPORT_DIR / "cmin_sr_v2_eval_warning.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return

    state = torch.load(ckpt, map_location="cpu")
    model = CMINSRv2Model()
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state, strict=False)
    device = args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    metric_rows = []
    process_rows = []
    for i, length in enumerate(args.t_eval):
        df, metrics = _eval_length(model, length, args.num_samples, args.seed + 31 * i, device)
        df.to_csv(TABLE_DIR / f"predictions_T{length}.csv", index=False)
        metric_rows.append(metrics)
        proc = (
            df.groupby("process_type")
            .agg(
                mean_p_scaling=("pred_p_scaling", "mean"),
                mean_p_mrw=("pred_p_mrw", "mean"),
                mean_p_mono=("pred_p_mono", "mean"),
                mean_lambda2_proj=("pred_lambda2_proj", "mean"),
                mean_residual_norm=("pred_residual_norm", "mean"),
                mean_mono_residual_norm=("pred_mono_residual_norm", "mean"),
                mean_gain=("pred_gain", "mean"),
                mean_curvature_sig=("pred_curvature_sig", "mean"),
            )
            .reset_index()
        )
        proc["T"] = length
        process_rows.extend(proc.to_dict(orient="records"))

    metrics_df = pd.DataFrame(metric_rows)
    process_df = pd.DataFrame(process_rows)
    metrics_df.to_csv(TABLE_DIR / "metrics_by_T.csv", index=False)
    process_df.to_csv(TABLE_DIR / "process_by_T.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    ax.plot(metrics_df["T"], metrics_df["fgn_p_mrw"], marker="o", label="fGn p_MRW")
    ax.plot(metrics_df["T"], metrics_df["mrw_p_mrw"], marker="s", label="MRW p_MRW")
    ax.plot(metrics_df["T"], metrics_df["low_mrw_p_mrw"], marker="^", label="Low-lambda2 MRW p_MRW")
    ax.set_xlabel("T")
    ax.set_ylabel("p_MRW")
    ax.set_title("v2 monofractal boundary calibration by T")
    ax.legend()
    fig.savefig(FIG_DIR / "pmrw_boundary_by_T.png", dpi=220)
    plt.close(fig)

    report = REPORT_DIR / "cmin_sr_v2_eval_summary.md"
    report.write_text(
        "\n".join(
            [
                "# CMIN-SR v2 Evaluation",
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
    (REPORT_DIR / "cmin_sr_v2_eval_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

