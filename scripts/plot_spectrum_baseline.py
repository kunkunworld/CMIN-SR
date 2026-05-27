from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def true_mrw_zeta(q: np.ndarray, h: float, lambda2: float) -> np.ndarray:
    return q * h - 0.5 * lambda2 * q * (q - 2.0)


def legendre_spectrum_from_zeta(q: np.ndarray, zeta_q: np.ndarray):
    alpha = np.gradient(zeta_q, q)
    f_alpha = q * alpha - zeta_q + 1.0
    return alpha, f_alpha


def main() -> None:
    output_dir_name = "dl_spectrum_cnn"
    if len(sys.argv) >= 2:
        output_dir_name = sys.argv[1]

    output_dir = ROOT / "outputs" / output_dir_name
    metrics_path = output_dir / "metrics.json"
    pred_path = output_dir / "test_predictions.npz"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    pred_data = np.load(pred_path)
    pred = pred_data["pred"]
    true = pred_data["true"]
    q_vals = pred_data["q_vals"]
    zeta_pred = pred_data["zeta_pred"] if "zeta_pred" in pred_data.files else None
    zeta_true = pred_data["zeta_true"] if "zeta_true" in pred_data.files else None

    loss_fig, loss_ax = plt.subplots(figsize=(6, 4))
    loss_ax.plot(metrics["history"]["train_loss"], label="Train total")
    loss_ax.plot(metrics["history"]["val_loss"], label="Val total")
    if "train_main_loss" in metrics["history"]:
        loss_ax.plot(metrics["history"]["train_main_loss"], label="Train main", linestyle="--")
    if "val_main_loss" in metrics["history"]:
        loss_ax.plot(metrics["history"]["val_main_loss"], label="Val main", linestyle="--")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("MSE Loss")
    loss_ax.set_title("Spectrum Baseline Training Curve")
    loss_ax.grid(True, alpha=0.3)
    loss_ax.legend(frameon=False)
    loss_path = output_dir / "training_curve.png"
    loss_fig.savefig(loss_path, dpi=200, bbox_inches="tight")
    plt.close(loss_fig)

    scatter_fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    names = ["lambda2", "H"]
    for i, ax in enumerate(axes):
        ax.scatter(true[:, i], pred[:, i], s=10, alpha=0.5)
        lo = min(float(true[:, i].min()), float(pred[:, i].min()))
        hi = max(float(true[:, i].max()), float(pred[:, i].max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
        ax.set_xlabel(f"True {names[i]}")
        ax.set_ylabel(f"Pred {names[i]}")
        ax.grid(True, alpha=0.3)
    scatter_path = output_dir / "parameter_scatter.png"
    scatter_fig.savefig(scatter_path, dpi=200, bbox_inches="tight")
    plt.close(scatter_fig)

    q_full = np.linspace(-3.0, 3.0, 25, dtype=np.float64)
    q_full = q_full[np.abs(q_full) > 1e-12]

    example_fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    axes = axes.ravel()
    for idx, ax in enumerate(axes):
        z_true = true_mrw_zeta(q_full, h=float(true[idx, 1]), lambda2=float(true[idx, 0]))
        z_pred = true_mrw_zeta(q_full, h=float(pred[idx, 1]), lambda2=float(pred[idx, 0]))
        alpha_true, f_true = legendre_spectrum_from_zeta(q_full, z_true)
        alpha_pred, f_pred = legendre_spectrum_from_zeta(q_full, z_pred)
        ax.plot(alpha_true, f_true, marker="o", label="True")
        ax.plot(alpha_pred, f_pred, marker="s", label="Pred")
        ax.set_title(
            f"Sample {idx} | true=({true[idx,0]:.3f}, {true[idx,1]:.3f}) "
            f"pred=({pred[idx,0]:.3f}, {pred[idx,1]:.3f})"
        )
        ax.set_xlabel("alpha")
        ax.set_ylabel("f(alpha)")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    example_fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    example_path = output_dir / "spectrum_examples.png"
    example_fig.savefig(example_path, dpi=200, bbox_inches="tight")
    plt.close(example_fig)

    result = {
        "training_curve": str(loss_path),
        "parameter_scatter": str(scatter_path),
        "spectrum_examples": str(example_path),
    }

    if zeta_pred is not None and zeta_true is not None:
        zeta_fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
        axes = axes.ravel()
        for idx, ax in enumerate(axes):
            ax.plot(q_vals, zeta_true[idx], marker="o", label="True zeta")
            ax.plot(q_vals, zeta_pred[idx], marker="s", label="Pred zeta")
            ax.set_title(f"Sample {idx} direct zeta(q)")
            ax.set_xlabel("q")
            ax.set_ylabel("zeta(q)")
            ax.grid(True, alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        zeta_fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        zeta_path = output_dir / "zeta_examples.png"
        zeta_fig.savefig(zeta_path, dpi=200, bbox_inches="tight")
        plt.close(zeta_fig)
        result["zeta_examples"] = str(zeta_path)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
