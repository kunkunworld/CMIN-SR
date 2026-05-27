from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "paper_writing_workspace" / "figures"
ASSET_DIR = ROOT / "paper_assets" / "figures"


COLORS = {
    "ink": "#243142",
    "muted": "#607086",
    "blue": "#3B74D7",
    "blue_light": "#DCE9FA",
    "green": "#4B9B6A",
    "green_light": "#DDF3E5",
    "orange": "#D59B37",
    "orange_light": "#F8E5BF",
    "red": "#C85A5A",
    "red_light": "#F7DADB",
    "gray": "#F5F7FA",
    "line": "#B7C0CD",
}


def _box(ax, xy, wh, title, subtitle="", fc="#FFFFFF", ec=None, lw=1.15, fontsize=9.6):
    ec = ec or COLORS["ink"]
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.028",
        fc=fc,
        ec=ec,
        lw=lw,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.60, title, ha="center", va="center", fontsize=fontsize, color=COLORS["ink"], weight="semibold")
        ax.text(x + w / 2, y + h * 0.31, subtitle, ha="center", va="center", fontsize=fontsize - 1.6, color=COLORS["muted"])
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fontsize, color=COLORS["ink"], weight="semibold")


def _arrow(ax, start, end, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1.25,
            color="#526174",
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=4,
            shrinkB=4,
        )
    )


def make_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.25), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.955,
        "CMIN-SR validity-aware spectral diagnostic pipeline",
        ha="center",
        va="top",
        fontsize=15,
        color=COLORS["ink"],
        weight="semibold",
    )

    stages = [
        (0.035, "Stage 1", "finite-sample spectrum"),
        (0.365, "Stage 2", "competing projections"),
        (0.695, "Stage 3", "validity diagnostics"),
    ]
    for x, stage, label in stages:
        _box(ax, (x, 0.11), (0.27, 0.76), "", fc=COLORS["gray"], ec="#D5DCE6", lw=0.9)
        ax.text(x + 0.02, 0.82, stage, color=COLORS["blue"], fontsize=11, weight="bold", ha="left", va="center")
        ax.text(x + 0.09, 0.82, label, color=COLORS["ink"], fontsize=11, weight="semibold", ha="left", va="center")

    _box(ax, (0.065, 0.62), (0.21, 0.105), "Raw finite increments", r"$x_t$, finite $T$", fc="#FFFFFF")
    _box(ax, (0.065, 0.43), (0.21, 0.105), "Empirical spectrum", r"$\widehat{\zeta}(q)$", fc=COLORS["blue_light"], ec="#4E6D94")
    _box(ax, (0.065, 0.245), (0.21, 0.095), "q-grid and scale range", fc="#FFFFFF", ec="#526174")

    _box(ax, (0.395, 0.64), (0.21, 0.105), "Monofractal", "linear projection", fc=COLORS["green_light"], ec="#4B8060")
    _box(ax, (0.395, 0.445), (0.21, 0.105), "MRW family", "parabolic projection", fc=COLORS["orange_light"], ec="#9A7330")
    _box(ax, (0.395, 0.25), (0.21, 0.095), "Residuals and", r"$\lambda^2_{\mathrm{proj}}$", fc="#FFFFFF")

    _box(ax, (0.725, 0.62), (0.21, 0.105), "Diagnostic scores", "scaling / curved / MRW", fc=COLORS["blue_light"], ec="#4E6D94")
    _box(ax, (0.725, 0.43), (0.21, 0.105), "Instability warning", "tail / regime / high-q", fc=COLORS["red_light"], ec="#9D5555")
    _box(ax, (0.725, 0.245), (0.21, 0.105), "Conservative interpretation", "evidence, not proof", fc="#FFFFFF")

    _arrow(ax, (0.17, 0.62), (0.17, 0.535))
    _arrow(ax, (0.17, 0.43), (0.17, 0.34))
    _arrow(ax, (0.275, 0.485), (0.395, 0.69), rad=-0.12)
    _arrow(ax, (0.275, 0.485), (0.395, 0.498), rad=0.02)
    _arrow(ax, (0.50, 0.64), (0.50, 0.55))
    _arrow(ax, (0.50, 0.445), (0.50, 0.345))
    _arrow(ax, (0.605, 0.298), (0.725, 0.675), rad=0.20)
    _arrow(ax, (0.605, 0.298), (0.725, 0.485), rad=0.08)
    _arrow(ax, (0.83, 0.62), (0.83, 0.535))
    _arrow(ax, (0.83, 0.43), (0.83, 0.35))

    ax.text(
        0.5,
        0.045,
        r"$\lambda^2_{\mathrm{proj}}$ is a projection coordinate; diagnostics organize evidence rather than prove mechanism.",
        ha="center",
        va="center",
        fontsize=9.2,
        color=COLORS["muted"],
    )

    for out in (FIG_DIR, ASSET_DIR):
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "fig1_journal_cmin_sr_pipeline.png", dpi=260)
        fig.savefig(out / "fig1_journal_cmin_sr_pipeline.pdf")
    plt.close(fig)


def make_identifiability() -> None:
    data_path = ROOT / "paper_writing_workspace" / "paper_assets" / "figure_data" / "fig5_lambda2_recovery.csv"
    if not data_path.exists():
        data_path = ROOT / "paper_assets" / "figure_data" / "fig5_lambda2_recovery.csv"
    df = pd.read_csv(data_path)
    rename = {
        "structure_ols": "OLS",
        "structure_bootstrap": "Bootstrap",
        "structure_smoothed": "Smoothed",
        "structure_trimmed": "Trimmed",
    }
    df = df[df["estimator"].isin(rename)].copy()
    df["label"] = df["estimator"].map(rename)
    palette = {"OLS": "#2F6BDE", "Bootstrap": "#E69F00", "Smoothed": "#1B9E77", "Trimmed": "#E64B4B"}

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.25))
    for label, group in df.groupby("label"):
        group = group.sort_values("T")
        axes[0].plot(group["T"], group["lambda2_corr"], marker="o", lw=1.8, ms=5, label=label, color=palette[label])
        axes[1].plot(group["T"], group["lambda2_mae"], marker="o", lw=1.8, ms=5, label=label, color=palette[label])

    axes[0].axhline(0.0, color="#7A8795", lw=1.0, ls="--")
    axes[0].set_title("A  Correlation with known synthetic $\\lambda^2$", loc="left", fontsize=11, weight="semibold")
    axes[0].set_ylabel(r"corr$(\lambda^2_{\mathrm{true}},\lambda^2_{\mathrm{proj}})$")
    axes[0].set_ylim(-0.22, 0.20)

    axes[1].set_title("B  Estimation error", loc="left", fontsize=11, weight="semibold")
    axes[1].set_ylabel(r"MAE of $\lambda^2_{\mathrm{proj}}$")
    axes[1].set_ylim(0.0788, 0.0834)

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df["T"].unique()))
        ax.set_xticklabels([str(int(t)) for t in sorted(df["T"].unique())])
        ax.set_xlabel("Window length $T$")
        ax.grid(axis="y", color="#D9DEE7", lw=0.8, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=COLORS["ink"])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.035))
    fig.suptitle("Finite-sample recovery of the MRW curvature coordinate", fontsize=14, weight="semibold", color=COLORS["ink"])
    fig.subplots_adjust(left=0.08, right=0.985, top=0.80, bottom=0.24, wspace=0.20)

    for out in (FIG_DIR, ASSET_DIR):
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "fig5_journal_finite_sample_identifiability.png", dpi=260, bbox_inches="tight")
        fig.savefig(out / "fig5_journal_finite_sample_identifiability.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    make_pipeline()
    make_identifiability()


if __name__ == "__main__":
    main()
