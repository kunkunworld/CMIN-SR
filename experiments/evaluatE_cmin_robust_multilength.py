from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_inverse.data import AntiConfoundedConfig, PROCESS_CODE_TO_NAME, generate_anti_confounded_dataset
from mrw_inverse.models import CMINRegressor


CHECKPOINT_PATH = ROOT / "checkpoints" / "cmin" / "cmin_robust_multilength.pt"
REPORT_DIR = ROOT / "outputs" / "reports" / "cmin_robust_multilength_eval"
TABLE_DIR = ROOT / "outputs" / "tables" / "cmin_robust_multilength_eval"
FIG_DIR = ROOT / "outputs" / "figures" / "cmin_robust_multilength_eval"


def _eval_length(model: CMINRegressor, length: int, num_samples: int, seed: int, device: str) -> tuple[pd.DataFrame, dict[str, float]]:
    dataset = generate_anti_confounded_dataset(AntiConfoundedConfig(length=length, num_samples=num_samples, seed=seed))
    rows = []
    for i in range(num_samples):
        x = torch.tensor(dataset["x"][i : i + 1], dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(x)
        proc = PROCESS_CODE_TO_NAME[int(dataset["process_code"][i])]
        f_diff = np.abs(out.f_alpha_hat.squeeze(0).cpu().numpy() - dataset["f_alpha_true"][i])
        f_mae = float(np.nanmean(f_diff)) if np.isfinite(f_diff).any() else float("nan")
        rows.append(
            {
                "T": length,
                "sample_id": int(dataset["sample_id"][i]),
                "pair_id": int(dataset["pair_id"][i]),
                "process_type": proc,
                "is_mrw": float(dataset["is_mrw"][i, 0]),
                "true_H": float(dataset["H_true"][i, 0]),
                "pred_H": float(out.h_hat.squeeze().cpu()),
                "true_lambda2": float(dataset["lambda2_true"][i, 0]),
                "pred_lambda2": float(out.lambda2_hat.squeeze().cpu()),
                "pred_p_mrw": float(out.p_mrw.squeeze().cpu()),
                "zeta_mae": float(np.mean(np.abs(out.zeta_hat.squeeze(0).cpu().numpy() - dataset["zeta_true"][i]))),
                "f_alpha_mae": f_mae,
            }
        )
    df = pd.DataFrame(rows)
    mrw = df["process_type"] == "MRW"
    pair_df = df[df["pair_id"] >= 0]
    lambda_gap = np.nan
    p_gap = np.nan
    if not pair_df.empty:
        pivot = pair_df.pivot_table(index="pair_id", columns="process_type", values=["pred_lambda2", "pred_p_mrw"], aggfunc="first")
        if ("pred_lambda2", "MRW") in pivot.columns and ("pred_lambda2", "Shuffled MRW") in pivot.columns:
            lambda_gap = float((pivot[("pred_lambda2", "MRW")] - pivot[("pred_lambda2", "Shuffled MRW")]).mean())
        if ("pred_p_mrw", "MRW") in pivot.columns and ("pred_p_mrw", "Shuffled MRW") in pivot.columns:
            p_gap = float((pivot[("pred_p_mrw", "MRW")] - pivot[("pred_p_mrw", "Shuffled MRW")]).mean())
    metrics = {
        "T": length,
        "train_length": int(length in {512, 1024}),
        "mrw_mae_H": float(np.mean(np.abs(df.loc[mrw, "pred_H"] - df.loc[mrw, "true_H"]))),
        "mrw_mae_lambda2": float(np.mean(np.abs(df.loc[mrw, "pred_lambda2"] - df.loc[mrw, "true_lambda2"]))),
        "mrw_mae_zeta": float(df.loc[mrw, "zeta_mae"].mean()),
        "mrw_mae_f_alpha": float(df.loc[mrw, "f_alpha_mae"].mean()),
        "student_t_lambda2": float(df.loc[df["process_type"] == "iid Student-t", "pred_lambda2"].mean()),
        "gaussian_lambda2": float(df.loc[df["process_type"] == "iid Gaussian", "pred_lambda2"].mean()),
        "garch_lambda2": float(df.loc[df["process_type"] == "GARCH(1,1)", "pred_lambda2"].mean()),
        "regime_lambda2": float(df.loc[df["process_type"] == "Regime-switching Gaussian", "pred_lambda2"].mean()),
        "mrw_vs_shuffled_lambda2_gap": lambda_gap,
        "mrw_vs_shuffled_p_mrw_gap": p_gap,
        "validity_accuracy_05": float(np.mean(((df["pred_p_mrw"] >= 0.5).astype(float)) == df["is_mrw"])),
    }
    if roc_auc_score is not None:
        try:
            metrics["p_mrw_auc"] = float(roc_auc_score(df["is_mrw"], df["pred_p_mrw"]))
        except Exception:
            metrics["p_mrw_auc"] = float("nan")
    return df, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multi-length robust CMIN across T.")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--t-eval", nargs="*", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--num-samples", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=5050)
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
            "next_step": "Run conda run -n for_codex python experiments/train_cmin_robust_multilength.py first.",
        }
        (REPORT_DIR / "cmin_robust_multilength_eval_warning.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return

    state = torch.load(ckpt, map_location="cpu")
    model = CMINRegressor()
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    device = args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    metric_rows = []
    process_rows = []
    pred_paths = []
    for i, length in enumerate(args.t_eval):
        df, metrics = _eval_length(model, length, args.num_samples, args.seed + 50 * i, device)
        pred_path = TABLE_DIR / f"predictions_T{length}.csv"
        df.to_csv(pred_path, index=False)
        pred_paths.append(pred_path)
        metric_rows.append(metrics)
        proc = df.groupby("process_type").agg(mean_pred_lambda2=("pred_lambda2", "mean"), mean_pred_p_mrw=("pred_p_mrw", "mean")).reset_index()
        proc["T"] = length
        process_rows.extend(proc.to_dict(orient="records"))

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = TABLE_DIR / "metrics_by_T.csv"
    metrics_df.to_csv(metrics_path, index=False)
    process_df = pd.DataFrame(process_rows)
    process_path = TABLE_DIR / "process_by_T.csv"
    process_df.to_csv(process_path, index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(metrics_df["T"], metrics_df["mrw_vs_shuffled_lambda2_gap"], marker="o", label="lambda2 gap")
    ax.plot(metrics_df["T"], metrics_df["mrw_vs_shuffled_p_mrw_gap"], marker="s", label="p_MRW gap")
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xlabel("T")
    ax.set_ylabel("gap")
    ax.set_title("MRW vs shuffled separation by T")
    ax.legend()
    gap_fig = FIG_DIR / "gap_by_T.png"
    fig.savefig(gap_fig, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for metric, label in [("student_t_lambda2", "Student-t"), ("gaussian_lambda2", "Gaussian"), ("regime_lambda2", "RegimeSwitch")]:
        ax.plot(metrics_df["T"], metrics_df[metric], marker="o", label=label)
    ax.set_xlabel("T")
    ax.set_ylabel("mean pred lambda2")
    ax.set_title("False positives by T")
    ax.legend()
    fp_fig = FIG_DIR / "false_positive_by_T.png"
    fig.savefig(fp_fig, dpi=220)
    plt.close(fig)

    report_lines = [
        "# CMIN-Robust Multi-length Evaluation",
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
        "",
        "## Outputs",
        "",
        f"- Metrics: `{metrics_path.relative_to(ROOT)}`",
        f"- Process means: `{process_path.relative_to(ROOT)}`",
        f"- Gap figure: `{gap_fig.relative_to(ROOT)}`",
        f"- False-positive figure: `{fp_fig.relative_to(ROOT)}`",
    ]
    for path in pred_paths:
        report_lines.append(f"- Predictions: `{path.relative_to(ROOT)}`")
    report_path = REPORT_DIR / "cmin_robust_multilength_eval_summary.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    meta = {
        "metrics_by_T": str(metrics_path.relative_to(ROOT)),
        "process_by_T": str(process_path.relative_to(ROOT)),
        "gap_figure": str(gap_fig.relative_to(ROOT)),
        "false_positive_figure": str(fp_fig.relative_to(ROOT)),
        "report": str(report_path.relative_to(ROOT)),
    }
    (REPORT_DIR / "cmin_robust_multilength_eval_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
