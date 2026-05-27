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

from mrw_dl.baselines import legendre_spectrum_from_zeta, true_mrw_zeta
from mrw_dl.generation import MRWParams, generate_mrw_fgn
from mrw_inverse.models import CMINRegressor


CHECKPOINT_PATH = ROOT / "checkpoints" / "cmin" / "cmin_tiny_synthetic.pt"
REPORT_DIR = ROOT / "outputs" / "reports" / "cmin_eval"
TABLE_DIR = ROOT / "outputs" / "tables" / "cmin_eval"
FIG_DIR = ROOT / "outputs" / "figures" / "cmin_eval"
Q_GRID = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32)


def _sample_eval_dataset(num_samples: int, length: int, seed: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for sample_id in range(num_samples):
        h_i = float(rng.uniform(0.1, 0.9))
        lambda_i = float(rng.uniform(0.0, 0.2))
        params = MRWParams(
            length=length,
            H=h_i,
            lambda2=lambda_i,
            L=min(256, max(64, length // 2)),
            sigma=1.0,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        sample = generate_mrw_fgn(params)
        dx = sample["dx"].astype(np.float32)
        dx = dx / max(np.std(dx), 1e-6)
        zeta = true_mrw_zeta(Q_GRID, h=h_i, lambda2=lambda_i).astype(np.float32)
        alpha, f_alpha = legendre_spectrum_from_zeta(Q_GRID, zeta)
        rows.append(
            {
                "sample_id": sample_id,
                "x": dx,
                "true_H": h_i,
                "true_lambda2": lambda_i,
                "true_zeta": zeta,
                "true_f_alpha": f_alpha.astype(np.float32),
                "true_alpha": alpha.astype(np.float32),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the tiny synthetic CMIN checkpoint.")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        warning = {
            "status": "missing_checkpoint",
            "checkpoint": str(ckpt_path.relative_to(ROOT) if ckpt_path.is_absolute() else ckpt_path),
            "next_step": "Run conda run -n for_codex python experiments/train_cmin_synthetic.py first.",
        }
        (REPORT_DIR / "cmin_eval_warning.json").write_text(json.dumps(warning, indent=2), encoding="utf-8")
        print(json.dumps(warning, indent=2))
        return

    state = torch.load(ckpt_path, map_location="cpu")
    model = CMINRegressor()
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    if args.device != "cpu" and torch.cuda.is_available():
        model = model.to(args.device)
    model.eval()

    rows = _sample_eval_dataset(args.num_samples, args.length, args.seed)
    pred_rows: list[dict[str, float]] = []
    example_rows: list[dict[str, object]] = []
    for row in rows:
        x = torch.tensor(row["x"][None, :], dtype=torch.float32, device=args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            out = model(x)
        pred_zeta = out.zeta_hat.squeeze(0).cpu().numpy()
        pred_f = out.f_alpha_hat.squeeze(0).cpu().numpy()
        pred_rows.append(
            {
                "sample_id": int(row["sample_id"]),
                "true_H": float(row["true_H"]),
                "pred_H": float(out.h_hat.squeeze().cpu()),
                "true_lambda2": float(row["true_lambda2"]),
                "pred_lambda2": float(out.lambda2_hat.squeeze().cpu()),
                "zeta_mae": float(np.mean(np.abs(pred_zeta - row["true_zeta"]))),
                "f_alpha_mae": float(np.mean(np.abs(pred_f - row["true_f_alpha"]))),
                "lambda2_boundary_hit": float(float(out.lambda2_hat.squeeze().cpu()) <= 0.005 or float(out.lambda2_hat.squeeze().cpu()) >= 0.195),
            }
        )
        if len(example_rows) < 4:
            example_rows.append(
                {
                    "sample_id": int(row["sample_id"]),
                    "true_zeta": row["true_zeta"].tolist(),
                    "pred_zeta": pred_zeta.tolist(),
                }
            )

    pred_df = pd.DataFrame(pred_rows)
    pred_path = TABLE_DIR / "cmin_eval_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    metrics = {
        "mae_H": float(np.mean(np.abs(pred_df["pred_H"] - pred_df["true_H"]))),
        "mae_lambda2": float(np.mean(np.abs(pred_df["pred_lambda2"] - pred_df["true_lambda2"]))),
        "mae_zeta": float(pred_df["zeta_mae"].mean()),
        "mae_f_alpha": float(pred_df["f_alpha_mae"].mean()),
        "lambda2_boundary_rate": float(pred_df["lambda2_boundary_hit"].mean()),
    }
    metrics_path = TABLE_DIR / "cmin_eval_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    axes[0].scatter(pred_df["true_H"], pred_df["pred_H"], s=10, alpha=0.6)
    axes[0].plot([0.1, 0.9], [0.1, 0.9], "k--", linewidth=1.0)
    axes[0].set_xlabel("true H")
    axes[0].set_ylabel("pred H")
    axes[0].set_title("H true vs pred")
    axes[1].scatter(pred_df["true_lambda2"], pred_df["pred_lambda2"], s=10, alpha=0.6)
    axes[1].plot([0.0, 0.2], [0.0, 0.2], "k--", linewidth=1.0)
    axes[1].set_xlabel("true lambda2")
    axes[1].set_ylabel("pred lambda2")
    axes[1].set_title("lambda2 true vs pred")
    scatter_path = FIG_DIR / "cmin_eval_scatter.png"
    fig.savefig(scatter_path, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)
    for ex in example_rows:
        ax.plot(Q_GRID, ex["true_zeta"], linestyle="--", alpha=0.8)
        ax.plot(Q_GRID, ex["pred_zeta"], alpha=0.8)
    ax.set_xlabel("q")
    ax.set_ylabel("zeta(q)")
    ax.set_title("Example zeta curves (dashed=true, solid=pred)")
    zeta_path = FIG_DIR / "cmin_eval_zeta_examples.png"
    fig.savefig(zeta_path, dpi=220)
    plt.close(fig)

    report_lines = [
        "# CMIN Synthetic Evaluation",
        "",
        f"- Checkpoint: `{ckpt_path.relative_to(ROOT) if ckpt_path.is_absolute() else ckpt_path}`",
        f"- Number of samples: `{args.num_samples}`",
        f"- Sequence length: `{args.length}`",
        "",
        "## Metrics",
        "",
        *[f"- `{k}`: `{v:.6f}`" for k, v in metrics.items()],
        "",
        "## Outputs",
        "",
        f"- Predictions: `{pred_path.relative_to(ROOT)}`",
        f"- Metrics CSV: `{metrics_path.relative_to(ROOT)}`",
        f"- Scatter figure: `{scatter_path.relative_to(ROOT)}`",
        f"- Zeta examples: `{zeta_path.relative_to(ROOT)}`",
    ]
    report_path = REPORT_DIR / "cmin_eval_summary.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    meta = {
        "predictions": str(pred_path.relative_to(ROOT)),
        "metrics_csv": str(metrics_path.relative_to(ROOT)),
        "scatter_figure": str(scatter_path.relative_to(ROOT)),
        "zeta_figure": str(zeta_path.relative_to(ROOT)),
        "report": str(report_path.relative_to(ROOT)),
        "metrics": metrics,
    }
    (REPORT_DIR / "cmin_eval_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
