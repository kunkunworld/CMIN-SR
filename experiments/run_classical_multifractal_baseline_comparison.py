from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_finite_sample_curvature_identifiability import run_grid, summarize


CLASSICAL_ESTIMATORS = (
    "structure_aggregated_ols",
    "mfdfa",
    "mfdfa_quadratic",
    "wavelet_leader_haar",
    "wtmm_haar",
)


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x)


def _write_report(summary: pd.DataFrame, out: Path, quick: bool) -> None:
    display_cols = [
        "T",
        "estimator",
        "lambda2_mae",
        "lambda2_corr",
        "high_lambda_detection_rate",
        "boundary_accuracy",
    ]
    compact = summary[display_cols].copy()
    for col in compact.columns:
        if col not in {"T", "estimator"}:
            compact[col] = compact[col].map(lambda x: "--" if not np.isfinite(x) else f"{x:.3f}")
    header = "| " + " | ".join(compact.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(compact.columns)) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in compact.to_numpy()]
    table_md = "\n".join([header, sep, *body])
    lines = [
        "# Classical Multifractal Baseline Comparison",
        "",
        "This experiment adds paper-ready classical multifractal baselines to the finite-sample curvature recovery study.",
        "",
        "Implemented estimators:",
        "",
        "- `structure_aggregated_ols`: overlapping aggregate-increment structure functions.",
        "- `mfdfa`: first-order MFDFA under the project zeta convention.",
        "- `mfdfa_quadratic`: second-order MFDFA.",
        "- `wavelet_leader_haar`: compact Haar wavelet-leader approximation.",
        "- `wtmm_haar`: compact Haar WTMM-style modulus-maxima approximation.",
        "",
        "The two wavelet estimators are dependency-free Haar baselines. They are useful as classical controls, but they should not be described as full production WTMM/wavelet-leader packages.",
        "",
        f"Run mode: {'quick' if quick else 'standard'}",
        "",
        "## Summary by T and estimator",
        "",
        table_md,
        "",
        "## Interpretation",
        "",
        "The baseline is intended to test whether stronger classical estimators remove the short-window lambda2 recovery bottleneck. If correlations remain weak or estimator-dependent, the paper should keep the conservative claim that finite-sample empirical spectrum estimation is limiting under the tested settings.",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--T-values", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    out_root = Path(args.output_dir) if args.output_dir else ROOT / "outputs"
    report_dir = out_root / "reports" / "classical_multifractal_baselines"
    table_dir = out_root / "tables" / "classical_multifractal_baselines"
    fig_dir = out_root / "figures" / "classical_multifractal_baselines"
    for path in (report_dir, table_dir, fig_dir):
        path.mkdir(parents=True, exist_ok=True)

    Hs = [0.4, 0.6] if args.quick else [0.2, 0.4, 0.6, 0.8]
    lams = [0.0, 0.03, 0.10, 0.20] if args.quick else [0.0, 0.005, 0.01, 0.03, 0.06, 0.10, 0.15, 0.20]
    Ts = _parse_ints(args.T_values) if args.T_values else ([512, 1024, 2048] if args.quick else [512, 1024, 2048, 4096])
    n = args.num_samples if args.num_samples is not None else (10 if args.quick else 20)
    q_grid = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    scales = (2, 4, 8, 16, 32, 64)

    sample = run_grid(Hs, lams, Ts, n, q_grid, scales, CLASSICAL_ESTIMATORS, args.seed)
    summary = summarize(sample)
    sample.to_csv(table_dir / "classical_baseline_sample_level.csv", index=False)
    summary.to_csv(table_dir / "classical_baseline_lambda2_recovery_by_T.csv", index=False)
    summary.groupby("estimator").mean(numeric_only=True).reset_index().to_csv(
        table_dir / "classical_baseline_by_estimator.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for estimator, g in summary.groupby("estimator"):
        ax.plot(g["T"], g["lambda2_corr"], marker="o", label=estimator)
    ax.axhline(0.0, color="0.5", linewidth=0.9, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sample length T")
    ax.set_ylabel("corr(lambda2_true, lambda2_proj)")
    ax.legend(fontsize=7)
    fig.savefig(fig_dir / "classical_lambda2_corr_vs_T.png", dpi=220)
    fig.savefig(fig_dir / "classical_lambda2_corr_vs_T.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for estimator, g in summary.groupby("estimator"):
        ax.plot(g["T"], g["high_lambda_detection_rate"], marker="o", label=estimator)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sample length T")
    ax.set_ylabel("High-lambda detection rate")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=7)
    fig.savefig(fig_dir / "classical_high_lambda_detection_vs_T.png", dpi=220)
    fig.savefig(fig_dir / "classical_high_lambda_detection_vs_T.pdf")
    plt.close(fig)

    report = report_dir / "classical_multifractal_baseline_summary.md"
    _write_report(summary, report, quick=args.quick)
    meta = {
        "sample_level": str((table_dir / "classical_baseline_sample_level.csv").resolve()),
        "summary_by_T": str((table_dir / "classical_baseline_lambda2_recovery_by_T.csv").resolve()),
        "summary_by_estimator": str((table_dir / "classical_baseline_by_estimator.csv").resolve()),
        "report": str(report.resolve()),
    }
    (report_dir / "classical_multifractal_baseline_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
