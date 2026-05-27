from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "outputs" / "baselines"
INPUT_PATH = BASE_DIR / "baseline_results_wide_improved_10.json"


def main() -> None:
    results = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(5, 2, figsize=(12, 18), constrained_layout=True)
    axes = axes.ravel()

    for ax, item in zip(axes, results):
        ax.plot(item["alpha_true"], item["f_true"], marker="o", label="True", linewidth=1.8)
        ax.plot(item["alpha_sf"], item["f_sf"], marker="s", label="Improved SF", linewidth=1.2)
        ax.plot(item["alpha_mfdfa"], item["f_mfdfa"], marker="^", label="Improved MFDFA", linewidth=1.2)
        ax.set_title(
            f"Sample {item['sample_index']} | "
            f"lambda2={item['params']['lambda2']:.3f}, H={item['params']['H']:.3f}"
        )
        ax.set_xlabel("alpha")
        ax.set_ylabel("f(alpha)")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Improved Traditional Baselines on Wide MRW Dataset", fontsize=16)
    output_path = BASE_DIR / "spectra_grid_wide_improved_10.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"saved_to": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
