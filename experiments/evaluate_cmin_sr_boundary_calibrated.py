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

from mrw_inverse.data import (
    BoundaryCalibrationDatasetConfig,
    PROCESS_CODE_TO_NAME,
    SpectralRepresentationDatasetConfig,
    generate_boundary_calibration_dataset,
    generate_spectral_representation_dataset,
)
from mrw_inverse.models import CMINSRv3Model


CHECKPOINT_PATH = ROOT / "checkpoints" / "cmin" / "cmin_sr_calibrated_synthetic.pt"
REPORT_DIR = ROOT / "outputs" / "reports" / "cmin_sr_boundary_calibrated_eval"
TABLE_DIR = ROOT / "outputs" / "tables" / "cmin_sr_boundary_calibrated_eval"
FIG_DIR = ROOT / "outputs" / "figures" / "cmin_sr_boundary_calibrated_eval"


def _standard_dataset(length: int, num_samples: int, seed: int) -> dict[str, np.ndarray]:
    return generate_spectral_representation_dataset(
        SpectralRepresentationDatasetConfig(length=length, num_samples=num_samples, seed=seed, mrw_ratio=0.20, low_lambda2_mrw_ratio=0.10)
    )


def _predict_rows(model: CMINSRv3Model, x_arr: np.ndarray, process_code: np.ndarray, h_true: np.ndarray, lambda2_true: np.ndarray, device: str, extra: dict[str, np.ndarray] | None = None):
    rows = []
    with torch.no_grad():
        for i in range(x_arr.shape[0]):
            x = torch.tensor(x_arr[i : i + 1], dtype=torch.float32, device=device)
            out = model(x)
            row = {
                "process_type": PROCESS_CODE_TO_NAME[int(process_code[i])],
                "true_H": float(h_true[i, 0]),
                "true_lambda2": float(lambda2_true[i, 0]),
                "pred_H_proj": float(out["H_proj"].squeeze().cpu()),
                "pred_lambda2_proj": float(out["lambda2_proj"].squeeze().cpu()),
                "pred_p_scaling": float(out["p_scaling"].squeeze().cpu()),
                "pred_p_curved": float(out["p_curved"].squeeze().cpu()),
                "pred_p_mrw": float(out["p_mrw"].squeeze().cpu()),
                "pred_p_mono": float(out["p_mono"].squeeze().cpu()),
                "pred_boundary_mrw_score": float(out["boundary_mrw_score"].squeeze().cpu()),
                "pred_residual_norm": float(out["residual_norm"].squeeze().cpu()),
                "pred_mono_residual_norm": float(out["mono_residual_norm"].squeeze().cpu()),
                "pred_gain": float(out["mrw_vs_mono_gain"].squeeze().cpu()),
                "pred_tail_instability": float(out["tail_instability"].squeeze().cpu()),
            }
            if extra is not None:
                for key, arr in extra.items():
                    val = arr[i]
                    row[key] = val.item() if hasattr(val, "item") else val
            rows.append(row)
    return rows


def _monotonic_fraction(df: pd.DataFrame, metric: str, increasing: bool = True) -> float:
    vals = df.sort_values("true_lambda2")[metric].to_numpy(dtype=float)
    if len(vals) < 2:
        return float("nan")
    diffs = np.diff(vals)
    return float((diffs >= -1e-6).mean() if increasing else (diffs <= 1e-6).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate boundary-calibrated CMIN-SR.")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--t-eval", nargs="*", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--boundary-groups-per-h", type=int, default=16)
    parser.add_argument("--seed", type=int, default=9090)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        out = {"status": "missing_checkpoint", "checkpoint": str(ckpt)}
        (REPORT_DIR / "cmin_sr_boundary_calibrated_eval_warning.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return

    state = torch.load(ckpt, map_location="cpu")
    model = CMINSRv3Model()
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state, strict=False)
    device = args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    standard_rows = []
    metric_rows = []
    process_rows = []
    for i, length in enumerate(args.t_eval):
        ds = _standard_dataset(length, args.num_samples, args.seed + 31 * i)
        rows = _predict_rows(model, ds["x"], ds["process_code"], ds["H_true"], ds["lambda2_true"], device, {"sample_id": ds["sample_id"]})
        df = pd.DataFrame(rows)
        df["T"] = length
        df.to_csv(TABLE_DIR / f"standard_predictions_T{length}.csv", index=False)
        standard_rows.append(df)
        proc = (
            df.groupby("process_type")
            .agg(
                mean_p_scaling=("pred_p_scaling", "mean"),
                mean_p_curved=("pred_p_curved", "mean"),
                mean_p_mrw=("pred_p_mrw", "mean"),
                mean_p_mono=("pred_p_mono", "mean"),
                mean_boundary_mrw_score=("pred_boundary_mrw_score", "mean"),
                mean_tail_instability=("pred_tail_instability", "mean"),
                mean_residual_norm=("pred_residual_norm", "mean"),
                mean_mono_residual_norm=("pred_mono_residual_norm", "mean"),
            )
            .reset_index()
        )
        proc["T"] = length
        process_rows.extend(proc.to_dict(orient="records"))
        metric_rows.append(
            {
                "T": length,
                "fgn_p_scaling": float(df.loc[df["process_type"] == "fGn", "pred_p_scaling"].mean()),
                "fgn_p_curved": float(df.loc[df["process_type"] == "fGn", "pred_p_curved"].mean()),
                "fgn_p_mrw": float(df.loc[df["process_type"] == "fGn", "pred_p_mrw"].mean()),
                "mrw_p_curved": float(df.loc[df["process_type"] == "MRW", "pred_p_curved"].mean()),
                "mrw_p_mrw": float(df.loc[df["process_type"] == "MRW", "pred_p_mrw"].mean()),
                "gaussian_p_mrw": float(df.loc[df["process_type"] == "iid Gaussian", "pred_p_mrw"].mean()),
                "student_t_p_mrw": float(df.loc[df["process_type"] == "iid Student-t", "pred_p_mrw"].mean()),
                "regime_p_mrw": float(df.loc[df["process_type"] == "Regime-switching Gaussian", "pred_p_mrw"].mean()),
            }
        )

    all_standard = pd.concat(standard_rows, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    process_df = pd.DataFrame(process_rows)
    metrics_df.to_csv(TABLE_DIR / "metrics_by_T.csv", index=False)
    process_df.to_csv(TABLE_DIR / "process_by_T.csv", index=False)

    h_values = (0.2, 0.4, 0.6, 0.8)
    bds = generate_boundary_calibration_dataset(
        BoundaryCalibrationDatasetConfig(
            length=1024,
            num_groups=len(h_values) * args.boundary_groups_per_h,
            h_values=h_values,
            seed=args.seed + 777,
        )
    )
    boundary_rows = _predict_rows(
        model,
        bds["x"],
        bds["process_code"],
        bds["H_true"],
        bds["lambda2_true"],
        device,
        {"group_id": bds["group_id"], "H_group": bds["H_group"].reshape(-1), "rank_curvature_target": bds["rank_curvature_target"].reshape(-1)},
    )
    boundary_df = pd.DataFrame(boundary_rows)
    boundary_df.to_csv(TABLE_DIR / "boundary_sweep_predictions.csv", index=False)
    sweep = (
        boundary_df.groupby(["process_type", "true_lambda2"])
        .agg(
            mean_p_scaling=("pred_p_scaling", "mean"),
            mean_p_curved=("pred_p_curved", "mean"),
            mean_p_mrw=("pred_p_mrw", "mean"),
            mean_p_mono=("pred_p_mono", "mean"),
            mean_boundary_mrw_score=("pred_boundary_mrw_score", "mean"),
            mean_residual_norm=("pred_residual_norm", "mean"),
            mean_mono_residual_norm=("pred_mono_residual_norm", "mean"),
            mean_gain=("pred_gain", "mean"),
        )
        .reset_index()
    )
    sweep.to_csv(TABLE_DIR / "boundary_sweep_summary.csv", index=False)
    mrw_sweep = sweep[sweep["process_type"].isin(["MRW", "Low-lambda2 MRW"])].sort_values("true_lambda2")
    monotonic = {
        "p_curved_monotonic_fraction": _monotonic_fraction(mrw_sweep, "mean_p_curved", True),
        "p_mrw_monotonic_fraction": _monotonic_fraction(mrw_sweep, "mean_p_mrw", True),
        "p_mono_decreasing_fraction": _monotonic_fraction(mrw_sweep, "mean_p_mono", False),
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    for metric, label in [
        ("mean_p_curved", "p_curved"),
        ("mean_p_mrw", "p_MRW"),
        ("mean_p_mono", "p_mono"),
        ("mean_boundary_mrw_score", "boundary"),
    ]:
        ax.plot(mrw_sweep["true_lambda2"], mrw_sweep[metric], marker="o", label=label)
    ax.set_xlabel("lambda2_true")
    ax.set_ylabel("score")
    ax.set_title("Boundary calibration sweep")
    ax.legend()
    fig.savefig(FIG_DIR / "boundary_sweep_scores.png", dpi=220)
    plt.close(fig)

    plot_df = all_standard[all_standard["T"] == 1024]
    colors = {"MRW": "#1f77b4", "Low-lambda2 MRW": "#17becf", "fGn": "#2ca02c", "iid Gaussian": "#98df8a", "iid Student-t": "#d62728", "GARCH(1,1)": "#ff7f0e", "Regime-switching Gaussian": "#9467bd", "Shuffled MRW": "#7f7f7f"}
    for x_col, y_col, fname, title in [
        ("pred_p_scaling", "pred_p_curved", "p_scaling_vs_p_curved.png", "p_scaling vs p_curved"),
        ("pred_p_curved", "pred_p_mrw", "p_curved_vs_p_mrw.png", "p_curved vs p_MRW"),
        ("pred_residual_norm", "pred_mono_residual_norm", "mrw_vs_mono_residual.png", "MRW residual vs monofractal residual"),
    ]:
        fig, ax = plt.subplots(figsize=(6.6, 5.0), constrained_layout=True)
        for proc, sub in plot_df.groupby("process_type"):
            ax.scatter(sub[x_col], sub[y_col], s=18, alpha=0.6, label=proc, color=colors.get(proc))
        if "residual" in fname:
            lim = max(float(plot_df[x_col].max()), float(plot_df[y_col].max()))
            ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        fig.savefig(FIG_DIR / fname, dpi=220)
        plt.close(fig)

    report = REPORT_DIR / "cmin_sr_boundary_calibrated_eval_summary.md"
    report.write_text(
        "\n".join(
            [
                "# CMIN-SR Boundary-Calibrated Evaluation",
                "",
                f"- Checkpoint: `{ckpt.relative_to(ROOT) if ckpt.is_absolute() else ckpt}`",
                "",
                "## Standard Metrics By T",
                "",
                metrics_df.to_csv(index=False),
                "",
                "## Process Means By T",
                "",
                process_df.to_csv(index=False),
                "",
                "## Boundary Sweep Summary",
                "",
                sweep.to_csv(index=False),
                "",
                "## Monotonicity",
                "",
                json.dumps(monotonic, indent=2),
            ]
        ),
        encoding="utf-8",
    )
    meta = {
        "metrics_by_T": str((TABLE_DIR / "metrics_by_T.csv").relative_to(ROOT)),
        "process_by_T": str((TABLE_DIR / "process_by_T.csv").relative_to(ROOT)),
        "boundary_sweep": str((TABLE_DIR / "boundary_sweep_summary.csv").relative_to(ROOT)),
        "report": str(report.relative_to(ROOT)),
        **monotonic,
    }
    (REPORT_DIR / "cmin_sr_boundary_calibrated_eval_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
