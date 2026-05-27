from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports" / "cmin_sr_final_version_comparison"
TABLE_DIR = ROOT / "outputs" / "tables" / "cmin_sr_final_version_comparison"
FIG_DIR = ROOT / "outputs" / "figures" / "cmin_sr_final_version_comparison"


def _read_optional(path: Path):
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    v1 = _read_optional(ROOT / "outputs" / "tables" / "cmin_sr_eval" / "process_by_T.csv")
    v2 = _read_optional(ROOT / "outputs" / "tables" / "cmin_sr_v2_eval" / "process_by_T.csv")
    v3 = _read_optional(ROOT / "outputs" / "tables" / "cmin_sr_v3_eval" / "process_by_T.csv")
    calibrated = _read_optional(ROOT / "outputs" / "tables" / "cmin_sr_boundary_calibrated_eval" / "process_by_T.csv")

    rows = []
    for name, df in [("v1", v1), ("v2", v2), ("v3", v3), ("calibrated", calibrated)]:
        if df is None:
            continue
        for _, row in df.iterrows():
            rows.append(
                {
                    "version": name,
                    "T": int(row["T"]),
                    "process_type": row["process_type"],
                    "mean_p_scaling": float(row.get("mean_p_scaling", np.nan)),
                    "mean_p_curved": float(row.get("mean_p_curved", np.nan)),
                    "mean_p_mrw": float(row.get("mean_p_mrw", np.nan)),
                    "mean_p_mono": float(row.get("mean_p_mono", np.nan)),
                    "mean_boundary_mrw_score": float(row.get("mean_boundary_mrw_score", np.nan)),
                    "mean_residual_norm": float(row.get("mean_residual_norm", np.nan)),
                    "mean_mono_residual_norm": float(row.get("mean_mono_residual_norm", np.nan)),
                    "mean_gain": float(row.get("mean_gain", np.nan)),
                }
            )
    comp = pd.DataFrame(rows)
    comp_path = TABLE_DIR / "version_process_comparison.csv"
    comp.to_csv(comp_path, index=False)

    plot_df = comp[comp["T"] == 1024] if (comp["T"] == 1024).any() else comp
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    focus = plot_df[plot_df["process_type"].isin(["MRW", "Low-lambda2 MRW", "fGn", "iid Gaussian", "iid Student-t", "GARCH(1,1)", "Regime-switching Gaussian"])]
    for version in focus["version"].unique():
        sub = focus[focus["version"] == version]
        ax.plot(sub["process_type"], sub["mean_p_mrw"], marker="o", label=version)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("mean p_MRW")
    ax.set_title("CMIN-SR version comparison")
    ax.legend()
    fig.savefig(FIG_DIR / "pmrw_versions.png", dpi=220)
    plt.close(fig)

    bar = plot_df[plot_df["process_type"].isin(["MRW", "fGn"])]
    if not bar.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
        pivot = bar.pivot_table(index="version", columns="process_type", values="mean_p_mrw", aggfunc="first")
        pivot = pivot.reindex([v for v in ["v1", "v2", "v3", "calibrated"] if v in pivot.index])
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("mean p_MRW")
        ax.set_title("fGn vs MRW p_MRW by version")
        ax.tick_params(axis="x", rotation=0)
        fig.savefig(FIG_DIR / "fgn_mrw_pmrw_versions.png", dpi=220)
        plt.close(fig)

    report_lines = [
        "# CMIN-SR Final Version Comparison",
        "",
        "## Process Summary",
        "",
        comp.to_csv(index=False),
    ]
    report_path = REPORT_DIR / "cmin_sr_final_version_comparison_summary.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    meta = {
        "comparison_csv": str(comp_path.relative_to(ROOT)),
        "report": str(report_path.relative_to(ROOT)),
    }
    (REPORT_DIR / "cmin_sr_final_version_comparison_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
