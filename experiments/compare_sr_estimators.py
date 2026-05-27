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
from mrw_inverse.models import CMINSRModel, SpectralRepresentationModel
from mrw_inverse.proxy import estimate_window


REPORT_DIR = ROOT / "outputs" / "reports" / "cmin_sr_comparison"
TABLE_DIR = ROOT / "outputs" / "tables" / "cmin_sr_comparison"
FIG_DIR = ROOT / "outputs" / "figures" / "cmin_sr_comparison"


def _load_torch_model(path: Path, model_cls, name: str):
    if not path.exists():
        return None, {"model_name": name, "checkpoint": str(path.relative_to(ROOT)), "available": False}
    state = torch.load(path, map_location="cpu")
    model = model_cls()
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state, strict=False)
    model.eval()
    return model, {"model_name": name, "checkpoint": str(path.relative_to(ROOT)), "available": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare deterministic spectral baseline, proxy, robust CMIN, and CMIN-SR.")
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7070)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ds = generate_spectral_representation_dataset(SpectralRepresentationDatasetConfig(length=args.length, num_samples=args.num_samples, seed=args.seed))
    deterministic = SpectralRepresentationModel()
    cmin_sr, sr_meta = _load_torch_model(ROOT / "checkpoints" / "cmin" / "cmin_sr_synthetic.pt", CMINSRModel, "cmin_sr")
    rows = []

    for i in range(args.num_samples):
        x_np = ds["x"][i]
        x = torch.tensor(x_np[None, :], dtype=torch.float32)
        proc_name = PROCESS_CODE_TO_NAME[int(ds["process_code"][i])]
        with torch.no_grad():
            det = deterministic(x)
        rows.append(
            {
                "estimator": "deterministic_spectral_baseline",
                "process_type": proc_name,
                "sample_id": int(ds["sample_id"][i]),
                "pair_id": int(ds["pair_id"][i]),
                "pred_H_proj": float(det["H_proj"].squeeze().cpu()),
                "pred_lambda2_proj": float(det["lambda2_proj"].squeeze().cpu()),
                "pred_p_scaling": float(det["p_scaling"].squeeze().cpu()),
                "pred_p_mrw": float(det["p_mrw"].squeeze().cpu()),
                "pred_residual_norm": float(det["residual_norm"].squeeze().cpu()),
            }
        )
        proxy_est = estimate_window(x_np, mode="proxy")
        rows.append(
            {
                "estimator": "proxy",
                "process_type": proc_name,
                "sample_id": int(ds["sample_id"][i]),
                "pair_id": int(ds["pair_id"][i]),
                "pred_H_proj": proxy_est.pred_H,
                "pred_lambda2_proj": proxy_est.pred_lambda2,
                "pred_p_scaling": np.nan,
                "pred_p_mrw": proxy_est.p_MRW,
                "pred_residual_norm": proxy_est.residual_norm,
            }
        )
        robust_path = ROOT / "checkpoints" / "cmin" / "cmin_robust_synthetic.pt"
        if robust_path.exists():
            robust_est = estimate_window(x_np, checkpoint_path=robust_path, mode="model")
            rows.append(
                {
                    "estimator": "cmin_robust",
                    "process_type": proc_name,
                    "sample_id": int(ds["sample_id"][i]),
                    "pair_id": int(ds["pair_id"][i]),
                    "pred_H_proj": robust_est.pred_H,
                    "pred_lambda2_proj": robust_est.pred_lambda2,
                    "pred_p_scaling": np.nan,
                    "pred_p_mrw": robust_est.p_MRW,
                    "pred_residual_norm": robust_est.residual_norm,
                }
            )
        multi_path = ROOT / "checkpoints" / "cmin" / "cmin_robust_multilength.pt"
        if multi_path.exists():
            multi_est = estimate_window(x_np, checkpoint_path=multi_path, mode="model")
            rows.append(
                {
                    "estimator": "cmin_robust_multilength",
                    "process_type": proc_name,
                    "sample_id": int(ds["sample_id"][i]),
                    "pair_id": int(ds["pair_id"][i]),
                    "pred_H_proj": multi_est.pred_H,
                    "pred_lambda2_proj": multi_est.pred_lambda2,
                    "pred_p_scaling": np.nan,
                    "pred_p_mrw": multi_est.p_MRW,
                    "pred_residual_norm": multi_est.residual_norm,
                }
            )
        if cmin_sr is not None:
            with torch.no_grad():
                out = cmin_sr(x)
            rows.append(
                {
                    "estimator": "cmin_sr",
                    "process_type": proc_name,
                    "sample_id": int(ds["sample_id"][i]),
                    "pair_id": int(ds["pair_id"][i]),
                    "pred_H_proj": float(out["H_proj"].squeeze().cpu()),
                    "pred_lambda2_proj": float(out["lambda2_proj"].squeeze().cpu()),
                    "pred_p_scaling": float(out["p_scaling"].squeeze().cpu()),
                    "pred_p_mrw": float(out["p_mrw"].squeeze().cpu()),
                    "pred_residual_norm": float(out["residual_norm"].squeeze().cpu()),
                }
            )

    df = pd.DataFrame(rows)
    detail_path = TABLE_DIR / "estimator_samples.csv"
    df.to_csv(detail_path, index=False)

    summary = (
        df.groupby(["estimator", "process_type"])
        .agg(
            mean_H_proj=("pred_H_proj", "mean"),
            mean_lambda2_proj=("pred_lambda2_proj", "mean"),
            mean_p_scaling=("pred_p_scaling", "mean"),
            mean_p_mrw=("pred_p_mrw", "mean"),
            mean_residual_norm=("pred_residual_norm", "mean"),
        )
        .reset_index()
    )
    summary_path = TABLE_DIR / "estimator_process_summary.csv"
    summary.to_csv(summary_path, index=False)

    gap_rows = []
    for estimator, sub in df.groupby("estimator"):
        pair_df = sub[sub["pair_id"] >= 0]
        if pair_df.empty:
            continue
        pivot = pair_df.pivot_table(index="pair_id", columns="process_type", values=["pred_lambda2_proj", "pred_p_mrw", "pred_residual_norm"], aggfunc="first")
        row = {"estimator": estimator}
        if ("pred_lambda2_proj", "MRW") in pivot.columns and ("pred_lambda2_proj", "Shuffled MRW") in pivot.columns:
            row["mrw_vs_shuffled_lambda2_gap"] = float((pivot[("pred_lambda2_proj", "MRW")] - pivot[("pred_lambda2_proj", "Shuffled MRW")]).mean())
        if ("pred_p_mrw", "MRW") in pivot.columns and ("pred_p_mrw", "Shuffled MRW") in pivot.columns:
            row["mrw_vs_shuffled_p_mrw_gap"] = float((pivot[("pred_p_mrw", "MRW")] - pivot[("pred_p_mrw", "Shuffled MRW")]).mean())
        if ("pred_residual_norm", "MRW") in pivot.columns and ("pred_residual_norm", "Shuffled MRW") in pivot.columns:
            row["mrw_vs_shuffled_residual_gap"] = float((pivot[("pred_residual_norm", "Shuffled MRW")] - pivot[("pred_residual_norm", "MRW")]).mean())
        gap_rows.append(row)
    gaps = pd.DataFrame(gap_rows)
    gaps_path = TABLE_DIR / "estimator_surrogate_gaps.csv"
    gaps.to_csv(gaps_path, index=False)

    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    plot_df = summary[summary["process_type"].isin(["MRW", "iid Student-t", "GARCH(1,1)", "Regime-switching Gaussian"])]
    for estimator in plot_df["estimator"].unique():
        sub = plot_df[plot_df["estimator"] == estimator]
        ax.plot(sub["process_type"], sub["mean_p_mrw"], marker="o", label=estimator)
    ax.set_ylabel("mean p_MRW")
    ax.set_title("SR estimator comparison on ambiguous processes")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=25)
    fig_path = FIG_DIR / "pmrw_comparison.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    report_lines = [
        "# CMIN-SR Comparison",
        "",
        f"- Length: `{args.length}`",
        f"- Samples: `{args.num_samples}`",
        "",
        "## Estimator Availability",
        "",
        f"- `cmin_sr`: `{sr_meta['available']}`",
        f"- `cmin_robust`: `{(ROOT / 'checkpoints' / 'cmin' / 'cmin_robust_synthetic.pt').exists()}`",
        f"- `cmin_robust_multilength`: `{(ROOT / 'checkpoints' / 'cmin' / 'cmin_robust_multilength.pt').exists()}`",
        "",
        "## Process Summary",
        "",
        summary.to_csv(index=False),
        "",
        "## Surrogate Gaps",
        "",
        gaps.to_csv(index=False),
    ]
    report_path = REPORT_DIR / "cmin_sr_comparison_summary.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    meta = {
        "detail_csv": str(detail_path.relative_to(ROOT)),
        "summary_csv": str(summary_path.relative_to(ROOT)),
        "gaps_csv": str(gaps_path.relative_to(ROOT)),
        "figure": str(fig_path.relative_to(ROOT)),
        "report": str(report_path.relative_to(ROOT)),
    }
    (REPORT_DIR / "cmin_sr_comparison_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

