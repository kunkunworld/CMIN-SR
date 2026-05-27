from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def plot_spectra_grid(results: list[dict], output_path: Path) -> None:
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
    fig.suptitle("MRW Baseline Spectra: 10 Samples", fontsize=16)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_zeta_grid(results: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(12, 18), constrained_layout=True)
    axes = axes.ravel()

    for ax, item in zip(axes, results):
        q_vals = np.array(item["q_vals"], dtype=float)
        zeta_true = np.array(item["zeta_true"], dtype=float)
        zeta_sf = np.array(item["zeta_sf"], dtype=float)
        zeta_mfdfa = np.array(item["zeta_mfdfa"], dtype=float)

        ax.plot(q_vals, zeta_true, marker="o", label="True", linewidth=1.8)
        ax.plot(q_vals, zeta_sf, marker="s", label="SF", linewidth=1.2)
        ax.plot(q_vals, zeta_mfdfa, marker="^", label="MFDFA", linewidth=1.2)
        ax.set_title(
            f"Sample {item['sample_index']} | "
            f"lambda2={item['params']['lambda2']:.3f}, H={item['params']['H']:.3f}"
        )
        ax.set_xlabel("q")
        ax.set_ylabel("zeta(q)")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("MRW Baseline Scaling Functions: 10 Samples", fontsize=16)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    stem = "baseline_results_10"
    if len(sys.argv) >= 2:
        stem = sys.argv[1]

    result_path = ROOT / "outputs" / "baselines" / f"{stem}.json"
    results = json.loads(result_path.read_text(encoding="utf-8"))
    output_dir = ROOT / "outputs" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    spectra_path = output_dir / f"{stem}_spectra.png"
    zeta_path = output_dir / f"{stem}_zeta.png"

    plot_spectra_grid(results, spectra_path)
    plot_zeta_grid(results, zeta_path)

    print(json.dumps({"spectra_plot": str(spectra_path), "zeta_plot": str(zeta_path)}, indent=2))


if __name__ == "__main__":
    main()
