from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from _paper_utils import ROOT, add_common_args, ensure_paper_dirs, write_note


def _save(fig, base, name: str) -> None:
    fig.tight_layout()
    fig.savefig(base / "figures" / f"{name}.png", dpi=220)
    fig.savefig(base / "figures" / f"{name}.pdf")
    plt.close(fig)


def _read(path: str) -> pd.DataFrame | None:
    p = ROOT / path
    if not p.exists():
        return None
    return pd.read_csv(p)


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description="Generate paper-ready figures from existing CSV outputs."))
    args = parser.parse_args()
    base = ensure_paper_dirs(args.output_dir)
    warnings: list[str] = []

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.axis("off")
    labels = [
        "raw signal",
        "empirical\nzeta(q)",
        "mono / MRW\nprojection",
        "residuals +\ninstability",
        "calibrated\ndiagnostics",
    ]
    xs = [0.08, 0.28, 0.50, 0.72, 0.91]
    for x, label in zip(xs, labels):
        ax.text(
            x,
            0.52,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "fc": "#f4f4f4", "ec": "#3f3f3f", "lw": 1.0},
        )
    for left, right in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(right - 0.075, 0.52), xytext=(left + 0.075, 0.52), arrowprops={"arrowstyle": "->", "lw": 1.1})
    ax.set_title("CMIN-SR Diagnostic Pipeline")
    _save(fig, base, "fig1_cmin_sr_framework")

    sg = _read("outputs/tables/spectral_geometry_calibrator_eval/summary_by_spectrum_type.csv")
    if sg is not None and {"p_curved", "p_mrw", "spectrum_type"}.issubset(sg.columns):
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        ax.scatter(sg["p_curved"], sg["p_mrw"], s=70, color="#2f5d8c")
        for _, row in sg.iterrows():
            ax.annotate(str(row["spectrum_type"]), (row["p_curved"], row["p_mrw"]), fontsize=7)
        ax.set_xlabel("p_curved")
        ax.set_ylabel("p_MRW")
        ax.set_title("Spectral Geometry Map")
        ax.grid(alpha=0.25)
        _save(fig, base, "fig2_spectral_geometry_calibration")
    else:
        warnings.append("Missing spectral geometry summary for fig2.")

    ident = _read("outputs/tables/finite_sample_identifiability/lambda2_recovery_by_T.csv")
    if ident is not None and {"T", "estimator", "lambda2_corr"}.issubset(ident.columns):
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        for est, sub in ident.groupby("estimator"):
            ax.plot(sub["T"], sub["lambda2_corr"], marker="o", label=est)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xlabel("T")
        ax.set_ylabel("corr(lambda2_true, lambda2_proj)")
        ax.set_title("Finite-Sample Curvature Recovery")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        _save(fig, base, "fig5_finite_sample_identifiability")
    else:
        warnings.append("Missing identifiability table for fig5.")

    bridge = _read("outputs/tables/zeta_noise_bridge/separation_margin_vs_noise.csv")
    if bridge is not None and bridge.shape[1] >= 2:
        x = bridge.iloc[:, 0]
        y = bridge.iloc[:, 1]
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        ax.plot(x, y, marker="o", color="#7a3b2e")
        ax.set_xlabel("zeta noise level")
        ax.set_ylabel("MRW - linear p_MRW gap")
        ax.set_title("Zeta Noise Bridge")
        ax.grid(alpha=0.25)
        _save(fig, base, "fig6_zeta_noise_bridge")
    else:
        warnings.append("Missing zeta noise bridge table.")

    attr = _read("outputs/tables/cmin_sr_failure_attribution/failure_attribution_summary_table.csv")
    if attr is not None and {"process_type", "level", "p_mrw"}.issubset(attr.columns):
        keep = attr[attr["process_type"].isin(["MRW", "fGn", "iid Gaussian"])]
        pivot = keep.pivot_table(index="process_type", columns="level", values="p_mrw", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        pivot.plot(kind="bar", ax=ax, color=["#386cb0", "#7fc97f", "#fdb462"])
        ax.set_ylabel("p_MRW")
        ax.set_xlabel("")
        ax.set_title("Failure Attribution")
        ax.legend(title="")
        ax.grid(axis="y", alpha=0.25)
        _save(fig, base, "fig6_failure_attribution")
    else:
        warnings.append("Missing failure attribution table.")

    proc = _read("outputs/tables/curvature_preserving_zeta_eval/process_by_T_band.csv")
    if proc is not None and {"p_curved_cal", "p_mrw_cal", "process_type"}.issubset(proc.columns):
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        for process, sub in proc.groupby("process_type"):
            ax.scatter(sub["p_curved_cal"], sub["p_mrw_cal"], s=35, label=process, alpha=0.75)
        ax.set_xlabel("p_curved_cal")
        ax.set_ylabel("p_MRW_cal")
        ax.set_title("Process-Family Map")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.25)
        _save(fig, base, "fig3_process_family_map")
    else:
        warnings.append("Missing process family table for fig3.")

    proj = _read("outputs/tables/cmin_sr_v3_eval/process_by_T.csv")
    if proj is not None and {"mean_residual_norm", "mean_mono_residual_norm", "process_type"}.issubset(proj.columns):
        fig, ax = plt.subplots(figsize=(5.0, 4.4))
        for process, sub in proj.groupby("process_type"):
            ax.scatter(sub["mean_residual_norm"], sub["mean_mono_residual_norm"], s=34, label=process, alpha=0.75)
        lo = min(proj["mean_residual_norm"].min(), proj["mean_mono_residual_norm"].min())
        hi = max(proj["mean_residual_norm"].max(), proj["mean_mono_residual_norm"].max())
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.9, linestyle="--")
        ax.set_xlabel("MRW residual")
        ax.set_ylabel("monofractal residual")
        ax.set_title("Projection Residual Geometry")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.25)
        _save(fig, base, "fig4_mrw_vs_mono_projection")
    else:
        warnings.append("Missing projection residual table for fig4.")

    write_note(
        base / "figures" / "figure_manifest.md",
        "Figure Manifest",
        ["Generated paper-ready figures from existing CSV outputs.", "", "Warnings:", *(f"- {w}" for w in warnings)]
        if warnings
        else ["Generated paper-ready figures from existing CSV outputs.", "No missing configured figure sources."],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
