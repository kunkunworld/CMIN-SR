from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_dl.generation import MRWParams, generate_mrw_fgn


def mrw_zeta(q: np.ndarray, h: float, lambda2: float) -> np.ndarray:
    return q * h - 0.5 * lambda2 * q * (q - 2.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate MRW simulation figure/data for the paper.")
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--H", type=float, default=0.60)
    parser.add_argument("--lambda2", type=float, default=0.12)
    parser.add_argument("--L", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    out_fig = ROOT / "paper_assets" / "figures"
    out_tab = ROOT / "paper_assets" / "tables"
    out_sum = ROOT / "paper_assets" / "summaries"
    tex_fig = ROOT / "paper_writing_workspace" / "figures"
    for path in (out_fig, out_tab, out_sum, tex_fig):
        path.mkdir(parents=True, exist_ok=True)

    params = MRWParams(length=args.length, H=args.H, lambda2=args.lambda2, L=args.L, seed=args.seed)
    sample = generate_mrw_fgn(params)
    t = sample["t"].astype(float)
    x = sample["x"].astype(float)
    dx = sample["dx"].astype(float)
    omega = sample["omega"].astype(float)

    q = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=float)
    z_linear = mrw_zeta(q, args.H, 0.0)
    z_boundary = mrw_zeta(q, args.H, 0.02)
    z_curved = mrw_zeta(q, args.H, args.lambda2)

    pd.DataFrame(
        {
            "t": t,
            "x": x,
            "dx": dx,
            "omega": omega,
            "H": args.H,
            "lambda2": args.lambda2,
            "L": args.L,
            "seed": args.seed,
        }
    ).to_csv(out_tab / "mrw_simulation_example_timeseries.csv", index=False)
    pd.DataFrame(
        {
            "q": q,
            "zeta_linear_lambda2_0": z_linear,
            "zeta_boundary_lambda2_0p02": z_boundary,
            "zeta_curved_lambda2": z_curved,
            "H": args.H,
            "lambda2_curved": args.lambda2,
        }
    ).to_csv(out_tab / "mrw_simulation_example_zeta.csv", index=False)

    show = min(args.length, 800)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), constrained_layout=False)
    axes[0, 0].plot(t[:show], omega[:show], color="#6b4c9a", lw=0.9)
    axes[0, 0].set_title("Log-volatility modulation")
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel(r"$\omega(t)$")

    axes[0, 1].plot(t[:show], dx[:show], color="#2f5d8c", lw=0.75)
    axes[0, 1].set_title("MRW increments")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel(r"$\Delta X(t)$")

    axes[1, 0].plot(t[:show], x[:show], color="#2f7f5f", lw=0.9)
    axes[1, 0].set_title("Integrated path")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel(r"$X(t)$")

    axes[1, 1].plot(q, z_linear, marker="o", label=r"linear, $\lambda^2=0$", color="#4c78a8")
    axes[1, 1].plot(q, z_boundary, marker="o", label=r"boundary, $\lambda^2=0.02$", color="#f58518")
    axes[1, 1].plot(q, z_curved, marker="o", label=rf"curved, $\lambda^2={args.lambda2:.2f}$", color="#54a24b")
    axes[1, 1].set_title("Analytic MRW scaling spectra")
    axes[1, 1].set_xlabel(r"$q$")
    axes[1, 1].set_ylabel(r"$\zeta(q)$")
    axes[1, 1].legend(fontsize=8)

    for ax in axes.ravel():
        ax.grid(alpha=0.25)

    fig.suptitle(rf"MRW simulation example ($H={args.H:.2f}$, $\lambda^2={args.lambda2:.2f}$)", y=0.985, fontsize=15)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.075, top=0.885, hspace=0.40, wspace=0.22)
    for base in (out_fig, tex_fig):
        fig.savefig(base / "fig9_mrw_simulation_example.png", dpi=240)
        fig.savefig(base / "fig9_mrw_simulation_example.pdf")
    plt.close(fig)

    prompt = [
        "# External Figure Prompt: MRW Simulation Example",
        "",
        "If a more polished illustrator-style figure is desired, redraw the same four-panel figure using the CSV files below.",
        "",
        "Data files:",
        "- `paper_assets/tables/mrw_simulation_example_timeseries.csv`",
        "- `paper_assets/tables/mrw_simulation_example_zeta.csv`",
        "",
        "Panel requirements:",
        "1. log-volatility modulation omega(t), first 800 observations;",
        "2. MRW increments Delta X(t), first 800 observations;",
        "3. integrated path X(t), first 800 observations;",
        "4. analytic zeta(q) curves for lambda2=0, 0.02, and 0.12.",
        "",
        "Style requirements: clean CSF/physics journal style, white background, readable axis labels, restrained colors, no decorative gradients, export PDF and PNG.",
    ]
    (out_sum / "mrw_simulation_figure_prompt.md").write_text("\n".join(prompt), encoding="utf-8")

    print(
        json.dumps(
            {
                "figure_pdf": "paper_assets/figures/fig9_mrw_simulation_example.pdf",
                "figure_png": "paper_assets/figures/fig9_mrw_simulation_example.png",
                "latex_figure_pdf": "paper_writing_workspace/figures/fig9_mrw_simulation_example.pdf",
                "timeseries_csv": "paper_assets/tables/mrw_simulation_example_timeseries.csv",
                "zeta_csv": "paper_assets/tables/mrw_simulation_example_zeta.csv",
                "external_prompt": "paper_assets/summaries/mrw_simulation_figure_prompt.md",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
