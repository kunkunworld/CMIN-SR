from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports" / "estimator_comparison_phase5"
TABLE_DIR = ROOT / "outputs" / "tables" / "estimator_comparison_phase5"
FIG_DIR = ROOT / "outputs" / "figures" / "estimator_comparison_phase5"


def _load(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _proc_value(df: pd.DataFrame, proc: str, col: str) -> float:
    row = df.loc[df["process_type"] == proc, col]
    return float(row.iloc[0]) if not row.empty else np.nan


def _interp(neg_df: pd.DataFrame, raw_market_available: bool) -> list[str]:
    lines: list[str] = []
    pivot = neg_df.set_index("metric")
    if {"proxy", "cmin_tiny", "cmin_robust", "cmin_robust_multilength"}.issubset(pivot.columns):
        student = pivot.loc["student_t_lambda2"]
        gauss = pivot.loc["gaussian_lambda2"]
        regime = pivot.loc["regime_switch_lambda2"]
        gap = pivot.loc["mrw_minus_shuffled_lambda2"]
        pmrw_gauss = pivot.loc["gaussian_p_mrw"] if "gaussian_p_mrw" in pivot.index else None
        if student["cmin_robust"] < student["cmin_tiny"]:
            lines.append(f"- Student-t false positive drops from tiny CMIN {student['cmin_tiny']:.4f} to robust CMIN {student['cmin_robust']:.4f}.")
        else:
            lines.append(f"- Student-t false positive is not reduced enough by robust CMIN ({student['cmin_tiny']:.4f} -> {student['cmin_robust']:.4f}).")
        if student["cmin_robust_multilength"] <= student["cmin_robust"] + 0.01:
            lines.append(f"- Multi-length robust training keeps Student-t false positives low ({student['cmin_robust_multilength']:.4f}).")
        else:
            lines.append(f"- Multi-length robust training shows some Student-t rebound ({student['cmin_robust_multilength']:.4f}).")
        if gauss["cmin_robust"] < gauss["cmin_tiny"]:
            lines.append(f"- Gaussian null lambda2 is reduced by robust CMIN ({gauss['cmin_tiny']:.4f} -> {gauss['cmin_robust']:.4f}).")
        else:
            lines.append(f"- Gaussian null lambda2 is not reduced by robust CMIN ({gauss['cmin_tiny']:.4f} -> {gauss['cmin_robust']:.4f}).")
        if regime["cmin_robust"] < regime["cmin_tiny"]:
            lines.append(f"- Regime-switching false positive is more cautious under robust CMIN ({regime['cmin_tiny']:.4f} -> {regime['cmin_robust']:.4f}).")
        else:
            lines.append(f"- Regime-switching false positive remains difficult ({regime['cmin_tiny']:.4f} -> {regime['cmin_robust']:.4f}).")
        if gap["cmin_robust"] > 0:
            lines.append(f"- Robust CMIN restores a positive MRW-vs-shuffled lambda2 gap ({gap['cmin_robust']:.4f}).")
        else:
            lines.append(f"- Robust CMIN still fails to restore a positive MRW-vs-shuffled lambda2 gap ({gap['cmin_robust']:.4f}).")
        if gap["cmin_robust_multilength"] > gap["cmin_robust"]:
            lines.append(f"- Multi-length robust training improves the external MRW-vs-shuffled gap relative to single-length robust ({gap['cmin_robust_multilength']:.4f} vs {gap['cmin_robust']:.4f}).")
        else:
            lines.append(f"- Multi-length robust training does not improve the external MRW-vs-shuffled gap enough ({gap['cmin_robust_multilength']:.4f}).")
        if pmrw_gauss is not None:
            lines.append(f"- Gaussian p_MRW moves from proxy {pmrw_gauss['proxy']:.4f} to tiny {pmrw_gauss['cmin_tiny']:.4f} to robust {pmrw_gauss['cmin_robust']:.4f} to multi-length {pmrw_gauss['cmin_robust_multilength']:.4f}.")
    if not raw_market_available:
        lines.append("- Raw market CSVs are still missing, so SPY/QQQ/BTC/ETH surrogate comparison is not yet available.")
    return lines or ["- Estimator comparison inputs are incomplete."]


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    neg_proxy = _load(ROOT / "outputs" / "tables" / "negative_controls_proxy" / "negative_controls_summary.csv")
    neg_tiny = _load(ROOT / "outputs" / "tables" / "negative_controls_cmin" / "negative_controls_summary.csv")
    neg_robust = _load(ROOT / "outputs" / "tables" / "negative_controls_cmin_robust" / "negative_controls_summary.csv")
    neg_multilength = _load(ROOT / "outputs" / "tables" / "negative_controls_multilength" / "negative_controls_summary.csv")
    sur_proxy = _load(ROOT / "outputs" / "tables" / "real_surrogate_validation_proxy" / "real_surrogate_gap_summary.csv")
    sur_tiny = _load(ROOT / "outputs" / "tables" / "real_surrogate_validation_cmin" / "real_surrogate_gap_summary.csv")
    sur_robust = _load(ROOT / "outputs" / "tables" / "real_surrogate_validation_cmin_robust" / "real_surrogate_gap_summary.csv")
    sur_multilength = _load(ROOT / "outputs" / "tables" / "real_surrogate_validation_cmin_robust_multilength" / "real_surrogate_gap_summary.csv")
    raw_market = _load(ROOT / "outputs" / "tables" / "raw_market_surrogate_validation" / "raw_market_surrogate_gap_summary.csv")

    if any(df is None for df in [neg_proxy, neg_tiny, neg_robust, neg_multilength, sur_proxy, sur_tiny, sur_robust, sur_multilength]):
        out = {
            "status": "missing_inputs",
            "needed": [
                "negative_controls_proxy",
                "negative_controls_cmin",
                "negative_controls_cmin_robust",
                "negative_controls_multilength",
                "real_surrogate_validation_proxy",
                "real_surrogate_validation_cmin",
                "real_surrogate_validation_cmin_robust",
                "real_surrogate_validation_cmin_robust_multilength",
            ],
        }
        (REPORT_DIR / "estimator_comparison_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return

    neg_compare = pd.DataFrame(
        [
            {"metric": "mrw_lambda2", "proxy": _proc_value(neg_proxy, "MRW", "mean_pred_lambda2"), "cmin_tiny": _proc_value(neg_tiny, "MRW", "mean_pred_lambda2"), "cmin_robust": _proc_value(neg_robust, "MRW", "mean_pred_lambda2"), "cmin_robust_multilength": _proc_value(neg_multilength, "MRW", "mean_pred_lambda2")},
            {"metric": "shuffled_lambda2", "proxy": _proc_value(neg_proxy, "Shuffled MRW", "mean_pred_lambda2"), "cmin_tiny": _proc_value(neg_tiny, "Shuffled MRW", "mean_pred_lambda2"), "cmin_robust": _proc_value(neg_robust, "Shuffled MRW", "mean_pred_lambda2"), "cmin_robust_multilength": _proc_value(neg_multilength, "Shuffled MRW", "mean_pred_lambda2")},
            {"metric": "mrw_minus_shuffled_lambda2", "proxy": _proc_value(neg_proxy, "MRW", "mean_pred_lambda2") - _proc_value(neg_proxy, "Shuffled MRW", "mean_pred_lambda2"), "cmin_tiny": _proc_value(neg_tiny, "MRW", "mean_pred_lambda2") - _proc_value(neg_tiny, "Shuffled MRW", "mean_pred_lambda2"), "cmin_robust": _proc_value(neg_robust, "MRW", "mean_pred_lambda2") - _proc_value(neg_robust, "Shuffled MRW", "mean_pred_lambda2"), "cmin_robust_multilength": _proc_value(neg_multilength, "MRW", "mean_pred_lambda2") - _proc_value(neg_multilength, "Shuffled MRW", "mean_pred_lambda2")},
            {"metric": "gaussian_lambda2", "proxy": _proc_value(neg_proxy, "iid Gaussian", "mean_pred_lambda2"), "cmin_tiny": _proc_value(neg_tiny, "iid Gaussian", "mean_pred_lambda2"), "cmin_robust": _proc_value(neg_robust, "iid Gaussian", "mean_pred_lambda2"), "cmin_robust_multilength": _proc_value(neg_multilength, "iid Gaussian", "mean_pred_lambda2")},
            {"metric": "student_t_lambda2", "proxy": _proc_value(neg_proxy, "iid Student-t", "mean_pred_lambda2"), "cmin_tiny": _proc_value(neg_tiny, "iid Student-t", "mean_pred_lambda2"), "cmin_robust": _proc_value(neg_robust, "iid Student-t", "mean_pred_lambda2"), "cmin_robust_multilength": _proc_value(neg_multilength, "iid Student-t", "mean_pred_lambda2")},
            {"metric": "garch_lambda2", "proxy": _proc_value(neg_proxy, "GARCH(1,1)", "mean_pred_lambda2"), "cmin_tiny": _proc_value(neg_tiny, "GARCH(1,1)", "mean_pred_lambda2"), "cmin_robust": _proc_value(neg_robust, "GARCH(1,1)", "mean_pred_lambda2"), "cmin_robust_multilength": _proc_value(neg_multilength, "GARCH(1,1)", "mean_pred_lambda2")},
            {"metric": "regime_switch_lambda2", "proxy": _proc_value(neg_proxy, "Regime-switching Gaussian", "mean_pred_lambda2"), "cmin_tiny": _proc_value(neg_tiny, "Regime-switching Gaussian", "mean_pred_lambda2"), "cmin_robust": _proc_value(neg_robust, "Regime-switching Gaussian", "mean_pred_lambda2"), "cmin_robust_multilength": _proc_value(neg_multilength, "Regime-switching Gaussian", "mean_pred_lambda2")},
            {"metric": "mrw_p_mrw", "proxy": _proc_value(neg_proxy, "MRW", "mean_p_MRW"), "cmin_tiny": _proc_value(neg_tiny, "MRW", "mean_p_MRW"), "cmin_robust": _proc_value(neg_robust, "MRW", "mean_p_MRW"), "cmin_robust_multilength": _proc_value(neg_multilength, "MRW", "mean_p_MRW")},
            {"metric": "gaussian_p_mrw", "proxy": _proc_value(neg_proxy, "iid Gaussian", "mean_p_MRW"), "cmin_tiny": _proc_value(neg_tiny, "iid Gaussian", "mean_p_MRW"), "cmin_robust": _proc_value(neg_robust, "iid Gaussian", "mean_p_MRW"), "cmin_robust_multilength": _proc_value(neg_multilength, "iid Gaussian", "mean_p_MRW")},
            {"metric": "student_t_p_mrw", "proxy": _proc_value(neg_proxy, "iid Student-t", "mean_p_MRW"), "cmin_tiny": _proc_value(neg_tiny, "iid Student-t", "mean_p_MRW"), "cmin_robust": _proc_value(neg_robust, "iid Student-t", "mean_p_MRW"), "cmin_robust_multilength": _proc_value(neg_multilength, "iid Student-t", "mean_p_MRW")},
        ]
    )
    neg_path = TABLE_DIR / "negative_control_comparison.csv"
    neg_compare.to_csv(neg_path, index=False)

    sur_compare = sur_proxy[["source_name", "mean_lambda2_gap_shuffle", "mean_p_MRW_gap_shuffle", "mean_f_width_gap_shuffle"]].merge(
        sur_tiny[["source_name", "mean_lambda2_gap_shuffle", "mean_p_MRW_gap_shuffle", "mean_f_width_gap_shuffle"]],
        on="source_name",
        suffixes=("_proxy", "_cmin_tiny"),
    ).merge(
        sur_robust[["source_name", "mean_lambda2_gap_shuffle", "mean_p_MRW_gap_shuffle", "mean_f_width_gap_shuffle"]],
        on="source_name",
    ).rename(
        columns={
            "mean_lambda2_gap_shuffle": "mean_lambda2_gap_shuffle_cmin_robust",
            "mean_p_MRW_gap_shuffle": "mean_p_MRW_gap_shuffle_cmin_robust",
            "mean_f_width_gap_shuffle": "mean_f_width_gap_shuffle_cmin_robust",
        }
    ).merge(
        sur_multilength[["source_name", "mean_lambda2_gap_shuffle", "mean_p_MRW_gap_shuffle", "mean_f_width_gap_shuffle"]],
        on="source_name",
    ).rename(
        columns={
            "mean_lambda2_gap_shuffle": "mean_lambda2_gap_shuffle_cmin_robust_multilength",
            "mean_p_MRW_gap_shuffle": "mean_p_MRW_gap_shuffle_cmin_robust_multilength",
            "mean_f_width_gap_shuffle": "mean_f_width_gap_shuffle_cmin_robust_multilength",
        }
    )
    sur_path = TABLE_DIR / "real_surrogate_comparison.csv"
    sur_compare.to_csv(sur_path, index=False)

    raw_market_path = TABLE_DIR / "raw_market_surrogate_comparison.csv"
    if raw_market is not None:
        raw_market.to_csv(raw_market_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    metrics_to_plot = ["mrw_minus_shuffled_lambda2", "gaussian_lambda2", "student_t_lambda2", "regime_switch_lambda2"]
    plot_df = neg_compare[neg_compare["metric"].isin(metrics_to_plot)]
    xs = np.arange(len(plot_df))
    axes[0].bar(xs - 0.30, plot_df["proxy"], width=0.20, label="proxy", color="tab:orange")
    axes[0].bar(xs - 0.10, plot_df["cmin_tiny"], width=0.20, label="cmin_tiny", color="tab:blue")
    axes[0].bar(xs + 0.10, plot_df["cmin_robust"], width=0.20, label="cmin_robust", color="tab:green")
    axes[0].bar(xs + 0.30, plot_df["cmin_robust_multilength"], width=0.20, label="cmin_robust_multilength", color="tab:red")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(plot_df["metric"], rotation=30, ha="right")
    axes[0].set_title("Negative-control comparison")
    axes[0].legend()

    xs = np.arange(len(sur_compare))
    axes[1].bar(xs - 0.30, sur_compare["mean_lambda2_gap_shuffle_proxy"], width=0.20, label="proxy", color="tab:orange")
    axes[1].bar(xs - 0.10, sur_compare["mean_lambda2_gap_shuffle_cmin_tiny"], width=0.20, label="cmin_tiny", color="tab:blue")
    axes[1].bar(xs + 0.10, sur_compare["mean_lambda2_gap_shuffle_cmin_robust"], width=0.20, label="cmin_robust", color="tab:green")
    axes[1].bar(xs + 0.30, sur_compare["mean_lambda2_gap_shuffle_cmin_robust_multilength"], width=0.20, label="cmin_robust_multilength", color="tab:red")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(sur_compare["source_name"], rotation=35, ha="right")
    axes[1].set_title("Real surrogate lambda2 gaps")
    axes[1].legend()
    fig_path = FIG_DIR / "estimator_comparison.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    report_lines = [
        "# Estimator Comparison Summary",
        "",
        f"- Negative-control comparison: `{neg_path.relative_to(ROOT)}`",
        f"- Real-surrogate comparison: `{sur_path.relative_to(ROOT)}`",
        f"- Raw-market surrogate comparison available: `{raw_market is not None}`",
        f"- Figure: `{fig_path.relative_to(ROOT)}`",
        "",
        "## Cautious Interpretation",
        "",
        *_interp(neg_compare, raw_market is not None),
        "",
        "## Policy",
        "",
        "- Proxy is transparent but confounded.",
        "- Tiny CMIN was trained on positive MRW only and showed always-MRW bias.",
        "- Robust CMIN should be judged primarily on false-positive reduction and restored MRW-vs-shuffled gaps, not only on synthetic MAE.",
        "- Multi-length robust CMIN targets length OOD, but external negative controls may still expose distribution-shift failures.",
        "- Volatility forecasting remains secondary evidence.",
    ]
    report_path = REPORT_DIR / "estimator_comparison_summary.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    meta = {
        "negative_controls_csv": str(neg_path.relative_to(ROOT)),
        "real_surrogate_csv": str(sur_path.relative_to(ROOT)),
        "raw_market_surrogate_csv": str(raw_market_path.relative_to(ROOT)) if raw_market is not None else "",
        "figure": str(fig_path.relative_to(ROOT)),
        "report": str(report_path.relative_to(ROOT)),
    }
    (REPORT_DIR / "estimator_comparison_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
