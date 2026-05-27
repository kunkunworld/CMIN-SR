from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    dataset_path = ROOT / "data" / "raw" / "mrw_dataset_robust_fgn.npz"
    if len(sys.argv) >= 2:
        dataset_path = Path(sys.argv[1])

    with np.load(dataset_path, allow_pickle=False) as data:
        params = data["params"]
        alpha = data["alpha_summary"]
        perturb = data["perturbations"]

    output_dir = ROOT / "outputs" / "dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes = axes.ravel()

    h = params[:, 3]
    lambda2 = params[:, 0]
    alpha_min = alpha[:, 0]
    alpha_peak = alpha[:, 1]
    alpha_max = alpha[:, 2]
    alpha_width = alpha[:, 3]

    sc = axes[0].scatter(h, lambda2, c=alpha_width, s=8, alpha=0.6, cmap="viridis")
    axes[0].set_xlabel("H")
    axes[0].set_ylabel("lambda2")
    axes[0].set_title("Parameter Coverage Colored by Alpha Width")
    fig.colorbar(sc, ax=axes[0], label="alpha width")

    sc2 = axes[1].scatter(alpha_peak, alpha_width, c=perturb[:, 0], s=8, alpha=0.6, cmap="magma")
    axes[1].set_xlabel("alpha peak")
    axes[1].set_ylabel("alpha width")
    axes[1].set_title("Spectrum Coverage Colored by Noise")
    fig.colorbar(sc2, ax=axes[1], label="noise std")

    axes[2].hist(alpha_min, bins=40, alpha=0.7, label="alpha min")
    axes[2].hist(alpha_peak, bins=40, alpha=0.7, label="alpha peak")
    axes[2].hist(alpha_max, bins=40, alpha=0.7, label="alpha max")
    axes[2].set_title("Alpha Value Distributions")
    axes[2].legend(frameon=False)

    axes[3].hist(perturb[:, 0], bins=30, alpha=0.7, label="noise std")
    axes[3].hist(perturb[:, 1] * 10.0, bins=30, alpha=0.7, label="outlier prob x10")
    axes[3].hist(perturb[:, 3], bins=30, alpha=0.7, label="drift")
    axes[3].set_title("Perturbation Distributions")
    axes[3].legend(frameon=False)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    output_path = output_dir / "robust_dataset_coverage.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "dataset_path": str(dataset_path),
        "plot_path": str(output_path),
        "num_samples": int(params.shape[0]),
        "lambda2_range": [float(lambda2.min()), float(lambda2.max())],
        "H_range": [float(h.min()), float(h.max())],
        "alpha_min_range": [float(alpha_min.min()), float(alpha_min.max())],
        "alpha_peak_range": [float(alpha_peak.min()), float(alpha_peak.max())],
        "alpha_max_range": [float(alpha_max.min()), float(alpha_max.max())],
        "alpha_width_range": [float(alpha_width.min()), float(alpha_width.max())],
    }
    (output_dir / "robust_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
