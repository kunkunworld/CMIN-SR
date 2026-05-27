from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports" / "proxy_vs_cmin"
TABLE_DIR = ROOT / "outputs" / "tables" / "proxy_vs_cmin"
FIG_DIR = ROOT / "outputs" / "figures" / "proxy_vs_cmin"


def _load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _interpret(neg_compare: pd.DataFrame, gap_compare: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if not neg_compare.empty:
        idx = neg_compare.set_index("metric")
        mrw_drop_proxy = float(idx.loc["mrw_minus_shuffled_lambda2", "proxy"]) if "mrw_minus_shuffled_lambda2" in idx.index else np.nan
        mrw_drop_cmin = float(idx.loc["mrw_minus_shuffled_lambda2", "cmin"]) if "mrw_minus_shuffled_lambda2" in idx.index else np.nan
        student_proxy = float(idx.loc["student_t_lambda2", "proxy"]) if "student_t_lambda2" in idx.index else np.nan
        student_cmin = float(idx.loc["student_t_lambda2", "cmin"]) if "student_t_lambda2" in idx.index else np.nan
        gauss_proxy = float(idx.loc["gaussian_lambda2", "proxy"]) if "gaussian_lambda2" in idx.index else np.nan
        gauss_cmin = float(idx.loc["gaussian_lambda2", "cmin"]) if "gaussian_lambda2" in idx.index else np.nan
        regime_proxy = float(idx.loc["regime_switch_lambda2", "proxy"]) if "regime_switch_lambda2" in idx.index else np.nan
        regime_cmin = float(idx.loc["regime_switch_lambda2", "cmin"]) if "regime_switch_lambda2" in idx.index else np.nan

        if np.isfinite(mrw_drop_proxy) and np.isfinite(mrw_drop_cmin):
            if mrw_drop_cmin > 0 and mrw_drop_cmin >= 0.8 * mrw_drop_proxy:
                lines.append(f"- MRW vs shuffled MRW: CMIN keeps a positive lambda2 drop ({mrw_drop_cmin:.4f}) while preserving most of the proxy separation ({mrw_drop_proxy:.4f}).")
            elif mrw_drop_cmin > 0:
                lines.append(f"- MRW vs shuffled MRW: CMIN still shows a positive lambda2 drop ({mrw_drop_cmin:.4f}), but the separation is weaker than proxy ({mrw_drop_proxy:.4f}).")
            else:
                lines.append("- MRW vs shuffled MRW: CMIN does not preserve a positive lambda2 drop in this tiny run, so the learned intermittency signal is not yet reliable.")

        if np.isfinite(student_proxy) and np.isfinite(student_cmin):
            if student_cmin < student_proxy:
                lines.append(f"- Student-t false positive: CMIN lowers the heavy-tail lambda2 response from {student_proxy:.4f} to {student_cmin:.4f}, which is the main desired direction.")
            else:
                lines.append(f"- Student-t false positive: CMIN does not reduce Student-t lambda2 ({student_proxy:.4f} -> {student_cmin:.4f}); heavy-tail confounding remains.")

        if np.isfinite(gauss_proxy) and np.isfinite(gauss_cmin):
            if gauss_cmin < gauss_proxy:
                lines.append(f"- Gaussian null: CMIN pushes the null-process lambda2 closer to zero ({gauss_proxy:.4f} -> {gauss_cmin:.4f}).")
            else:
                lines.append(f"- Gaussian null: CMIN does not improve the Gaussian baseline ({gauss_proxy:.4f} -> {gauss_cmin:.4f}).")

        if np.isfinite(regime_proxy) and np.isfinite(regime_cmin):
            if regime_cmin < regime_proxy:
                lines.append(f"- Regime-switching stress test: CMIN is more cautious than proxy on nonstationary Gaussian windows ({regime_proxy:.4f} -> {regime_cmin:.4f}).")
            else:
                lines.append(f"- Regime-switching stress test: CMIN is not more cautious than proxy ({regime_proxy:.4f} -> {regime_cmin:.4f}).")

    if not gap_compare.empty:
        improved = int((gap_compare["cmin"] > gap_compare["proxy"]).sum())
        total = int(len(gap_compare))
        lines.append(f"- Real surrogate validation: CMIN increases the original-vs-shuffled lambda2 gap on {improved} out of {total} factor series.")

    if not lines:
        lines.append("- Comparison could not be generated because one or more input tables are missing.")
    return lines


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    neg_proxy = _load_csv(ROOT / "outputs" / "tables" / "negative_controls_proxy" / "negative_controls_summary.csv")
    neg_cmin = _load_csv(ROOT / "outputs" / "tables" / "negative_controls_cmin" / "negative_controls_summary.csv")
    gap_proxy = _load_csv(ROOT / "outputs" / "tables" / "real_surrogate_validation_proxy" / "real_surrogate_gap_summary.csv")
    gap_cmin = _load_csv(ROOT / "outputs" / "tables" / "real_surrogate_validation_cmin" / "real_surrogate_gap_summary.csv")

    if neg_proxy is None or neg_cmin is None or gap_proxy is None or gap_cmin is None:
        out = {
            "status": "missing_inputs",
            "needed": {
                "negative_controls_proxy": "outputs/tables/negative_controls_proxy/negative_controls_summary.csv",
                "negative_controls_cmin": "outputs/tables/negative_controls_cmin/negative_controls_summary.csv",
                "surrogate_proxy": "outputs/tables/real_surrogate_validation_proxy/real_surrogate_gap_summary.csv",
                "surrogate_cmin": "outputs/tables/real_surrogate_validation_cmin/real_surrogate_gap_summary.csv",
            },
        }
        (REPORT_DIR / "proxy_vs_cmin_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return

    def get(df: pd.DataFrame, proc: str, col: str) -> float:
        row = df.loc[df["process_type"] == proc, col]
        return float(row.iloc[0]) if not row.empty else np.nan

    neg_compare = pd.DataFrame(
        [
            {"metric": "mrw_minus_shuffled_lambda2", "proxy": get(neg_proxy, "MRW", "mean_pred_lambda2") - get(neg_proxy, "Shuffled MRW", "mean_pred_lambda2"), "cmin": get(neg_cmin, "MRW", "mean_pred_lambda2") - get(neg_cmin, "Shuffled MRW", "mean_pred_lambda2")},
            {"metric": "student_t_lambda2", "proxy": get(neg_proxy, "iid Student-t", "mean_pred_lambda2"), "cmin": get(neg_cmin, "iid Student-t", "mean_pred_lambda2")},
            {"metric": "gaussian_lambda2", "proxy": get(neg_proxy, "iid Gaussian", "mean_pred_lambda2"), "cmin": get(neg_cmin, "iid Gaussian", "mean_pred_lambda2")},
            {"metric": "garch_lambda2", "proxy": get(neg_proxy, "GARCH(1,1)", "mean_pred_lambda2"), "cmin": get(neg_cmin, "GARCH(1,1)", "mean_pred_lambda2")},
            {"metric": "garch_p_mrw", "proxy": get(neg_proxy, "GARCH(1,1)", "mean_p_MRW"), "cmin": get(neg_cmin, "GARCH(1,1)", "mean_p_MRW")},
            {"metric": "regime_switch_lambda2", "proxy": get(neg_proxy, "Regime-switching Gaussian", "mean_pred_lambda2"), "cmin": get(neg_cmin, "Regime-switching Gaussian", "mean_pred_lambda2")},
            {"metric": "gaussian_boundary_rate", "proxy": get(neg_proxy, "iid Gaussian", "lambda2_boundary_rate"), "cmin": get(neg_cmin, "iid Gaussian", "lambda2_boundary_rate")},
        ]
    )
    neg_compare_path = TABLE_DIR / "proxy_vs_cmin_negative_controls.csv"
    neg_compare.to_csv(neg_compare_path, index=False)

    gap_compare = gap_proxy[["source_name", "mean_lambda2_gap_shuffle", "mean_p_MRW_gap_shuffle", "mean_f_width_gap_shuffle"]].merge(
        gap_cmin[["source_name", "mean_lambda2_gap_shuffle", "mean_p_MRW_gap_shuffle", "mean_f_width_gap_shuffle"]],
        on="source_name",
        suffixes=("_proxy", "_cmin"),
    )
    gap_compare_path = TABLE_DIR / "proxy_vs_cmin_surrogate_gaps.csv"
    gap_compare.to_csv(gap_compare_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    plot_df = neg_compare.melt(id_vars="metric", value_vars=["proxy", "cmin"], var_name="estimator", value_name="value")
    for estimator, color in [("proxy", "tab:orange"), ("cmin", "tab:blue")]:
        subset = plot_df[plot_df["estimator"] == estimator]
        axes[0].bar(
            np.arange(len(subset)) + (-0.2 if estimator == "proxy" else 0.2),
            subset["value"],
            width=0.35,
            label=estimator,
            color=color,
        )
    axes[0].set_xticks(np.arange(len(neg_compare)))
    axes[0].set_xticklabels(neg_compare["metric"], rotation=35, ha="right")
    axes[0].set_title("Negative-control summary comparison")
    axes[0].legend()

    gap_plot = gap_compare[["source_name", "mean_lambda2_gap_shuffle_proxy", "mean_lambda2_gap_shuffle_cmin"]]
    axes[1].bar(np.arange(len(gap_plot)) - 0.2, gap_plot["mean_lambda2_gap_shuffle_proxy"], width=0.35, label="proxy", color="tab:orange")
    axes[1].bar(np.arange(len(gap_plot)) + 0.2, gap_plot["mean_lambda2_gap_shuffle_cmin"], width=0.35, label="cmin", color="tab:blue")
    axes[1].set_xticks(np.arange(len(gap_plot)))
    axes[1].set_xticklabels(gap_plot["source_name"], rotation=35, ha="right")
    axes[1].set_title("Original - shuffled lambda2 gaps")
    axes[1].legend()
    fig_path = FIG_DIR / "proxy_vs_cmin_summary.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    report_lines = [
        "# Proxy vs CMIN Summary",
        "",
        f"- Negative-control comparison: `{neg_compare_path.relative_to(ROOT)}`",
        f"- Surrogate-gap comparison: `{gap_compare_path.relative_to(ROOT)}`",
        f"- Figure: `{fig_path.relative_to(ROOT)}`",
        "",
        "## Cautious Interpretation",
        "",
        *_interpret(neg_compare, gap_compare.rename(columns={"mean_lambda2_gap_shuffle_proxy": "proxy", "mean_lambda2_gap_shuffle_cmin": "cmin"})),
        "",
        "## Policy",
        "",
        "- Improvements are meaningful even if they are partial.",
        "- If Student-t or regime-switching false positives remain high, tiny synthetic training is likely still insufficient.",
        "- This comparison should not be read as final proof; it is a stress test for whether learned physics constraints improve selectivity over the proxy baseline.",
    ]
    report_path = REPORT_DIR / "proxy_vs_cmin_summary.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    meta = {
        "negative_controls_csv": str(neg_compare_path.relative_to(ROOT)),
        "surrogate_gap_csv": str(gap_compare_path.relative_to(ROOT)),
        "figure": str(fig_path.relative_to(ROOT)),
        "report": str(report_path.relative_to(ROOT)),
    }
    (REPORT_DIR / "proxy_vs_cmin_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
