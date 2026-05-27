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
except Exception:  # pragma: no cover - optional dependency
    roc_auc_score = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_inverse.data import AntiConfoundedConfig, PROCESS_CODE_TO_NAME, generate_anti_confounded_dataset
from mrw_inverse.models import CMINRegressor


CHECKPOINT_PATH = ROOT / "checkpoints" / "cmin" / "cmin_robust_synthetic.pt"
REPORT_DIR = ROOT / "outputs" / "reports" / "cmin_robust_eval"
TABLE_DIR = ROOT / "outputs" / "tables" / "cmin_robust_eval"
FIG_DIR = ROOT / "outputs" / "figures" / "cmin_robust_eval"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate robust anti-confounded CMIN.")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--num-samples", type=int, default=1500)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=4040)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        out = {
            "status": "missing_checkpoint",
            "checkpoint": str(ckpt_path.relative_to(ROOT) if ckpt_path.is_absolute() else ckpt_path),
            "next_step": "Run conda run -n for_codex python experiments/train_cmin_robust.py first.",
        }
        (REPORT_DIR / "cmin_robust_eval_warning.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return

    sklearn_available = roc_auc_score is not None

    state = torch.load(ckpt_path, map_location="cpu")
    model = CMINRegressor()
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    if args.device != "cpu" and torch.cuda.is_available():
        model = model.to(args.device)
    device = args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    model.eval()

    dataset = generate_anti_confounded_dataset(AntiConfoundedConfig(length=args.length, num_samples=args.num_samples, seed=args.seed))
    rows: list[dict[str, float | str]] = []
    for i in range(args.num_samples):
        x = torch.tensor(dataset["x"][i : i + 1], dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(x)
        proc_name = PROCESS_CODE_TO_NAME[int(dataset["process_code"][i])]
        f_diff = np.abs(out.f_alpha_hat.squeeze(0).cpu().numpy() - dataset["f_alpha_true"][i])
        f_mae = float(np.nanmean(f_diff)) if np.isfinite(f_diff).any() else float("nan")
        rows.append(
            {
                "sample_id": int(dataset["sample_id"][i]),
                "pair_id": int(dataset["pair_id"][i]),
                "process_type": proc_name,
                "is_mrw": float(dataset["is_mrw"][i, 0]),
                "true_H": float(dataset["H_true"][i, 0]),
                "pred_H": float(out.h_hat.squeeze().cpu()),
                "true_lambda2": float(dataset["lambda2_true"][i, 0]),
                "pred_lambda2": float(out.lambda2_hat.squeeze().cpu()),
                "target_p_mrw": float(dataset["target_p_mrw"][i, 0]),
                "pred_p_mrw": float(out.p_mrw.squeeze().cpu()),
                "zeta_mae": float(np.mean(np.abs(out.zeta_hat.squeeze(0).cpu().numpy() - dataset["zeta_true"][i]))),
                "f_alpha_mae": f_mae,
                "residual_norm": float(torch.mean(torch.abs(out.residual_zeta)).cpu()),
            }
        )
    df = pd.DataFrame(rows)
    pred_path = TABLE_DIR / "cmin_robust_eval_predictions.csv"
    df.to_csv(pred_path, index=False)

    mrw = df["process_type"] == "MRW"
    pair_df = df[df["pair_id"] >= 0]
    pair_gap_lambda = np.nan
    pair_gap_p = np.nan
    if not pair_df.empty:
        pivot = pair_df.pivot_table(index="pair_id", columns="process_type", values=["pred_lambda2", "pred_p_mrw"], aggfunc="first")
        if ("pred_lambda2", "MRW") in pivot.columns and ("pred_lambda2", "Shuffled MRW") in pivot.columns:
            pair_gap_lambda = float((pivot[("pred_lambda2", "MRW")] - pivot[("pred_lambda2", "Shuffled MRW")]).mean())
        if ("pred_p_mrw", "MRW") in pivot.columns and ("pred_p_mrw", "Shuffled MRW") in pivot.columns:
            pair_gap_p = float((pivot[("pred_p_mrw", "MRW")] - pivot[("pred_p_mrw", "Shuffled MRW")]).mean())

    metrics = {
        "mrw_mae_H": float(np.mean(np.abs(df.loc[mrw, "pred_H"] - df.loc[mrw, "true_H"]))),
        "mrw_mae_lambda2": float(np.mean(np.abs(df.loc[mrw, "pred_lambda2"] - df.loc[mrw, "true_lambda2"]))),
        "mrw_mae_zeta": float(df.loc[mrw, "zeta_mae"].mean()),
        "mrw_mae_f_alpha": float(df.loc[mrw, "f_alpha_mae"].mean()),
        "validity_accuracy_05": float(np.mean(((df["pred_p_mrw"] >= 0.5).astype(float)) == df["is_mrw"])),
        "student_t_lambda2": float(df.loc[df["process_type"] == "iid Student-t", "pred_lambda2"].mean()),
        "gaussian_lambda2": float(df.loc[df["process_type"] == "iid Gaussian", "pred_lambda2"].mean()),
        "garch_lambda2": float(df.loc[df["process_type"] == "GARCH(1,1)", "pred_lambda2"].mean()),
        "regime_lambda2": float(df.loc[df["process_type"] == "Regime-switching Gaussian", "pred_lambda2"].mean()),
        "mrw_vs_shuffled_lambda2_gap": pair_gap_lambda,
        "mrw_vs_shuffled_p_mrw_gap": pair_gap_p,
    }
    if sklearn_available:
        try:
            metrics["p_mrw_auc"] = float(roc_auc_score(df["is_mrw"], df["pred_p_mrw"]))
        except Exception:
            metrics["p_mrw_auc"] = float("nan")

    metrics_path = TABLE_DIR / "cmin_robust_eval_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    by_process = (
        df.groupby("process_type")
        .agg(
            mean_pred_H=("pred_H", "mean"),
            mean_pred_lambda2=("pred_lambda2", "mean"),
            mean_pred_p_mrw=("pred_p_mrw", "mean"),
            mean_residual_norm=("residual_norm", "mean"),
        )
        .reset_index()
    )
    by_process_path = TABLE_DIR / "cmin_robust_eval_by_process.csv"
    by_process.to_csv(by_process_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    df.boxplot(column="pred_lambda2", by="process_type", ax=axes[0], grid=False, rot=35)
    axes[0].set_title("lambda2 by process")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("pred_lambda2")
    df.boxplot(column="pred_p_mrw", by="process_type", ax=axes[1], grid=False, rot=35)
    axes[1].set_title("p_MRW by process")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("pred_p_mrw")
    fig.suptitle("")
    fig_path = FIG_DIR / "cmin_robust_eval_boxplots.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    report_lines = [
        "# CMIN-Robust Evaluation",
        "",
        f"- Checkpoint: `{ckpt_path.relative_to(ROOT) if ckpt_path.is_absolute() else ckpt_path}`",
        f"- Num samples: `{args.num_samples}`",
        "",
        "## Metrics",
        "",
        *[f"- `{k}`: `{v}`" for k, v in metrics.items()],
        "",
        "## Mean By Process",
        "",
        by_process.to_csv(index=False),
        "",
        "## Outputs",
        "",
        f"- Predictions: `{pred_path.relative_to(ROOT)}`",
        f"- Metrics: `{metrics_path.relative_to(ROOT)}`",
        f"- By process: `{by_process_path.relative_to(ROOT)}`",
        f"- Figure: `{fig_path.relative_to(ROOT)}`",
    ]
    report_path = REPORT_DIR / "cmin_robust_eval_summary.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    meta = {
        "predictions": str(pred_path.relative_to(ROOT)),
        "metrics_csv": str(metrics_path.relative_to(ROOT)),
        "by_process_csv": str(by_process_path.relative_to(ROOT)),
        "figure": str(fig_path.relative_to(ROOT)),
        "report": str(report_path.relative_to(ROOT)),
        "metrics": metrics,
    }
    (REPORT_DIR / "cmin_robust_eval_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
