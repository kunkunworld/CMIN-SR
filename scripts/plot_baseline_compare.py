from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "outputs" / "baselines"
OLD_PATH = BASE_DIR / "baseline_results_10.json"
WIDE_PATH = BASE_DIR / "baseline_results_wide_10.json"


def _plot_grid(results: list[dict], title: str, output_path: Path) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(12, 18), constrained_layout=True)
    axes = axes.ravel()

    for ax, item in zip(axes, results):
        alpha_true = np.array(item["alpha_true"], dtype=float)
        f_true = np.array(item["f_true"], dtype=float)
        alpha_sf = np.array(item["alpha_sf"], dtype=float)
        f_sf = np.array(item["f_sf"], dtype=float)
        alpha_mfdfa = np.array(item["alpha_mfdfa"], dtype=float)
        f_mfdfa = np.array(item["f_mfdfa"], dtype=float)

        ax.plot(alpha_true, f_true, marker="o", label="True", linewidth=1.8)
        ax.plot(alpha_sf, f_sf, marker="s", label="SF", linewidth=1.2)
        ax.plot(alpha_mfdfa, f_mfdfa, marker="^", label="MFDFA", linewidth=1.2)
        ax.set_title(
            f"Sample {item['sample_index']} | "
            f"lambda2={item['params']['lambda2']:.3f}, H={item['params']['H']:.3f}"
        )
        ax.set_xlabel("alpha")
        ax.set_ylabel("f(alpha)")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(title, fontsize=16)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _summarize_widths(results: list[dict]) -> dict[str, float]:
    true_widths = []
    sf_widths = []
    mfdfa_widths = []
    for item in results:
        alpha_true = np.array(item["alpha_true"], dtype=float)
        alpha_sf = np.array(item["alpha_sf"], dtype=float)
        alpha_mfdfa = np.array(item["alpha_mfdfa"], dtype=float)
        true_widths.append(float(alpha_true.max() - alpha_true.min()))
        sf_widths.append(float(alpha_sf.max() - alpha_sf.min()))
        mfdfa_widths.append(float(alpha_mfdfa.max() - alpha_mfdfa.min()))
    return {
        "true_mean_width": float(np.mean(true_widths)),
        "sf_mean_width": float(np.mean(sf_widths)),
        "mfdfa_mean_width": float(np.mean(mfdfa_widths)),
    }


def main() -> None:
    old_results = json.loads(OLD_PATH.read_text(encoding="utf-8"))
    wide_results = json.loads(WIDE_PATH.read_text(encoding="utf-8"))

    old_plot = BASE_DIR / "spectra_grid_old_10.png"
    wide_plot = BASE_DIR / "spectra_grid_wide_10.png"

    _plot_grid(old_results, "Original Dataset Spectra: 10 Samples", old_plot)
    _plot_grid(wide_results, "Wide Dataset Spectra: 10 Samples", wide_plot)

    summary = {
        "old": _summarize_widths(old_results),
        "wide": _summarize_widths(wide_results),
        "old_plot": str(old_plot),
        "wide_plot": str(wide_plot),
    }
    summary_path = BASE_DIR / "baseline_compare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
