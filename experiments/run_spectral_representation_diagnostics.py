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

from mrw_inverse.data.process_generators import DEFAULT_Q_GRID, generate_process_sample
from mrw_inverse.models import SpectralRepresentationModel

BASE_REPORT_DIR = ROOT / "outputs" / "reports"
BASE_TABLE_DIR = ROOT / "outputs" / "tables"
BASE_FIG_DIR = ROOT / "outputs" / "figures"


def _interpret(summary: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    take = lambda name, col: float(summary.loc[summary["process_type"] == name, col].iloc[0]) if (summary["process_type"] == name).any() else float("nan")
    mrw_res = take("MRW", "mean_residual_norm")
    shuf_res = take("Shuffled MRW", "mean_residual_norm")
    mrw_p = take("MRW", "mean_p_mrw")
    student_ps = take("iid Student-t", "mean_p_scaling")
    garch_l2 = take("GARCH(1,1)", "mean_lambda2_proj")
    fgn_l2 = take("fGn", "mean_lambda2_proj")
    regime_stab = take("Regime-switching Gaussian", "mean_spectrum_stability")
    if np.isfinite(mrw_res) and np.isfinite(shuf_res):
        if mrw_res < shuf_res:
            lines.append(f"- MRW projection quality: MRW has lower projection residual than shuffled MRW ({mrw_res:.4f} vs {shuf_res:.4f}), which is consistent with genuine temporal multiscale dependence.")
        else:
            lines.append(f"- MRW projection quality: MRW residual is not lower than shuffled MRW ({mrw_res:.4f} vs {shuf_res:.4f}); projection validity should be treated cautiously.")
    if np.isfinite(mrw_p):
        lines.append(f"- MRW validity: mean `p_MRW` for MRW is {mrw_p:.4f}. This should be read as projection credibility, not a universal truth label.")
    if np.isfinite(student_ps):
        if student_ps < 0.6:
            lines.append(f"- Heavy-tail stability: Student-t `p_scaling` is only {student_ps:.4f}, suggesting unstable high-q empirical scaling rather than clean multifractal structure.")
        else:
            lines.append(f"- Heavy-tail stability: Student-t still shows moderate `p_scaling` ({student_ps:.4f}), so empirical scaling law and heavy-tail instability are not fully separable yet.")
    if np.isfinite(fgn_l2):
        lines.append(f"- Monofractal control: fGn projected `lambda2` is {fgn_l2:.4f}; this should be interpreted as MRW-family projection curvature, not as proof of true intermittency.")
    if np.isfinite(garch_l2):
        lines.append(f"- GARCH stress case: projected `lambda2` is {garch_l2:.4f}. Nonzero curvature can exist even when MRW residual remains non-negligible.")
    if np.isfinite(regime_stab):
        lines.append(f"- Regime-switching stability: mean spectrum stability is {regime_stab:.4f}; apparent spectra with lower stability are a key non-MRW mismatch mode.")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Run empirical spectrum and MRW projection diagnostics across stochastic processes.")
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--output-tag", default="spectral_representation_diagnostics")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    report_dir = BASE_REPORT_DIR / args.output_tag
    table_dir = BASE_TABLE_DIR / args.output_tag
    fig_dir = BASE_FIG_DIR / args.output_tag
    report_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    model = SpectralRepresentationModel()
    rng = np.random.default_rng(args.seed)
    process_names = ["MRW", "Shuffled MRW", "fGn", "iid Gaussian", "iid Student-t", "GARCH(1,1)", "Regime-switching Gaussian"]
    rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []

    for process_name in process_names:
        for sample_id in range(args.num_samples):
            if process_name in {"MRW", "Shuffled MRW"}:
                h = float(rng.uniform(0.3, 0.8))
                lambda2 = float(rng.uniform(0.04, 0.12))
                sample = generate_process_sample(process_name, args.length, rng, h=h, lambda2=lambda2)
            elif process_name == "fGn":
                h = float(rng.uniform(0.3, 0.8))
                lambda2 = np.nan
                sample = generate_process_sample(process_name, args.length, rng, h=h)
            else:
                h = np.nan
                lambda2 = np.nan
                sample = generate_process_sample(process_name, args.length, rng)
            x = torch.tensor(sample.x[None, :], dtype=torch.float32)
            with torch.no_grad():
                out = model(x)
            zeta_emp = out["zeta_emp"].squeeze(0).cpu().numpy()
            zeta_mrw = out["zeta_mrw"].squeeze(0).cpu().numpy()
            alpha_emp = out["alpha_emp"].squeeze(0).cpu().numpy()
            f_emp = out["f_emp"].squeeze(0).cpu().numpy()
            for qi, q in enumerate(DEFAULT_Q_GRID):
                curve_rows.append(
                    {
                        "process_type": process_name,
                        "sample_id": sample_id,
                        "q": float(q),
                        "zeta_emp": float(zeta_emp[qi]),
                        "zeta_mrw": float(zeta_mrw[qi]),
                        "alpha_emp": float(alpha_emp[qi]),
                        "f_emp": float(f_emp[qi]),
                    }
                )
            rows.append(
                {
                    "process_type": process_name,
                    "sample_id": sample_id,
                    "T": args.length,
                    "true_H": h,
                    "true_lambda2": lambda2,
                    "H_proj": float(out["H_proj"].squeeze().cpu()),
                    "lambda2_proj": float(out["lambda2_proj"].squeeze().cpu()),
                    "residual_norm": float(out["residual_norm"].squeeze().cpu()),
                    "p_scaling": float(out["p_scaling"].squeeze().cpu()),
                    "p_MRW": float(out["p_mrw"].squeeze().cpu()),
                    "spectrum_width": float(out["spectrum_width"].squeeze().cpu()),
                    "spectrum_curvature": float(out["spectrum_curvature"].squeeze().cpu()),
                    "scaling_fit_quality": float(out["scaling_fit_quality"].squeeze().cpu()),
                    "spectrum_stability": float(out["spectrum_stability"].squeeze().cpu()),
                    "tail_instability": float(out["tail_instability"].squeeze().cpu()),
                    "projection_gain": float(out["projection_gain"].squeeze().cpu()),
                    "mode": out["mode"],
                    "notes": sample.notes,
                }
            )

    detail = pd.DataFrame(rows)
    curves = pd.DataFrame(curve_rows)
    detail_path = table_dir / "spectral_representation_samples.csv"
    curves_path = table_dir / "spectral_representation_curves.csv"
    detail.to_csv(detail_path, index=False)
    curves.to_csv(curves_path, index=False)

    summary = (
        detail.groupby("process_type")
        .agg(
            n_samples=("sample_id", "count"),
            mean_H_proj=("H_proj", "mean"),
            mean_lambda2_proj=("lambda2_proj", "mean"),
            mean_residual_norm=("residual_norm", "mean"),
            mean_p_scaling=("p_scaling", "mean"),
            mean_p_mrw=("p_MRW", "mean"),
            mean_spectrum_width=("spectrum_width", "mean"),
            mean_spectrum_curvature=("spectrum_curvature", "mean"),
            mean_scaling_fit_quality=("scaling_fit_quality", "mean"),
            mean_spectrum_stability=("spectrum_stability", "mean"),
            mean_tail_instability=("tail_instability", "mean"),
            mean_projection_gain=("projection_gain", "mean"),
        )
        .reset_index()
    )
    summary_path = table_dir / "spectral_representation_summary.csv"
    summary.to_csv(summary_path, index=False)

    curve_summary = (
        curves.groupby(["process_type", "q"])
        .agg(
            zeta_emp_mean=("zeta_emp", "mean"),
            zeta_emp_std=("zeta_emp", "std"),
            zeta_mrw_mean=("zeta_mrw", "mean"),
        )
        .reset_index()
    )
    curve_summary_path = table_dir / "spectral_representation_curve_summary.csv"
    curve_summary.to_csv(curve_summary_path, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for process_name in process_names:
        subset = curve_summary[curve_summary["process_type"] == process_name]
        axes[0, 0].plot(subset["q"], subset["zeta_emp_mean"], label=process_name)
    axes[0, 0].set_title("Empirical zeta(q) mean")
    axes[0, 0].set_xlabel("q")
    axes[0, 0].set_ylabel("zeta_emp")
    axes[0, 0].legend(fontsize=8)
    detail.boxplot(column="lambda2_proj", by="process_type", ax=axes[0, 1], rot=35, grid=False)
    axes[0, 1].set_title("MRW projected lambda2")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("lambda2_proj")
    detail.boxplot(column="p_scaling", by="process_type", ax=axes[1, 0], rot=35, grid=False)
    axes[1, 0].set_title("p_scaling by process")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("p_scaling")
    detail.boxplot(column="p_MRW", by="process_type", ax=axes[1, 1], rot=35, grid=False)
    axes[1, 1].set_title("p_MRW by process")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("p_MRW")
    fig.suptitle("Spectral representation diagnostics")
    fig_path = fig_dir / "spectral_representation_diagnostics.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    report_lines = [
        "# Spectral Representation Diagnostics",
        "",
        f"- Sequence length: `{args.length}`",
        f"- Samples per process: `{args.num_samples}`",
        f"- Detail CSV: `{detail_path.relative_to(ROOT)}`",
        f"- Curve CSV: `{curves_path.relative_to(ROOT)}`",
        f"- Summary CSV: `{summary_path.relative_to(ROOT)}`",
        f"- Curve summary CSV: `{curve_summary_path.relative_to(ROOT)}`",
        f"- Figure: `{fig_path.relative_to(ROOT)}`",
        "",
        "## Aggregate Summary",
        "",
        summary.to_csv(index=False),
        "",
        "## Cautious Interpretation",
        "",
    ]
    report_lines.extend(_interpret(summary))
    report_lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `zeta_emp` / `f_emp` are empirical spectral summaries, not proof of MRW.",
            "- `H_proj` / `lambda2_proj` are MRW-family projection parameters.",
            "- `p_scaling` asks whether a stable empirical scaling law is present.",
            "- `p_MRW` asks whether the empirical spectrum is well explained by the MRW parametric family.",
            "- `residual_norm` measures MRW projection mismatch.",
        ]
    )
    report_path = report_dir / "spectral_representation_diagnostics_summary.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    meta = {
        "detail_csv": str(detail_path.relative_to(ROOT)),
        "summary_csv": str(summary_path.relative_to(ROOT)),
        "curve_csv": str(curves_path.relative_to(ROOT)),
        "curve_summary_csv": str(curve_summary_path.relative_to(ROOT)),
        "figure": str(fig_path.relative_to(ROOT)),
        "report": str(report_path.relative_to(ROOT)),
    }
    (report_dir / "spectral_representation_diagnostics_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
