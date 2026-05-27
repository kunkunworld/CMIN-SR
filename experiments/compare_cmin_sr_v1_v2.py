from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports" / "cmin_sr_v1_v2_comparison"
TABLE_DIR = ROOT / "outputs" / "tables" / "cmin_sr_v1_v2_comparison"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    v1 = pd.read_csv(ROOT / "outputs" / "tables" / "cmin_sr_eval" / "process_by_T.csv")
    v2 = pd.read_csv(ROOT / "outputs" / "tables" / "cmin_sr_v2_eval" / "process_by_T.csv")
    rows = []
    for T in sorted(set(v1["T"]).intersection(set(v2["T"]))):
        for proc in sorted(set(v1["process_type"]).intersection(set(v2["process_type"]))):
            a = v1[(v1["T"] == T) & (v1["process_type"] == proc)]
            b = v2[(v2["T"] == T) & (v2["process_type"] == proc)]
            if a.empty or b.empty:
                continue
            rows.append(
                {
                    "T": T,
                    "process_type": proc,
                    "v1_p_scaling": float(a["mean_p_scaling"].iloc[0]),
                    "v2_p_scaling": float(b["mean_p_scaling"].iloc[0]),
                    "v1_p_mrw": float(a["mean_p_mrw"].iloc[0]),
                    "v2_p_mrw": float(b["mean_p_mrw"].iloc[0]),
                    "delta_p_mrw": float(b["mean_p_mrw"].iloc[0] - a["mean_p_mrw"].iloc[0]),
                    "v1_residual": float(a["mean_residual_norm"].iloc[0]),
                    "v2_residual": float(b["mean_residual_norm"].iloc[0]),
                    "v2_mono_residual": float(b["mean_mono_residual_norm"].iloc[0]) if "mean_mono_residual_norm" in b.columns else float("nan"),
                    "v2_gain": float(b["mean_gain"].iloc[0]) if "mean_gain" in b.columns else float("nan"),
                }
            )
    df = pd.DataFrame(rows)
    summary_path = TABLE_DIR / "v1_v2_process_comparison.csv"
    df.to_csv(summary_path, index=False)

    report_lines = [
        "# CMIN-SR v1 vs v2 Comparison",
        "",
        "## Process Comparison",
        "",
        df.to_csv(index=False),
    ]
    report_path = REPORT_DIR / "cmin_sr_v1_v2_comparison_summary.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    meta = {"comparison_csv": str(summary_path.relative_to(ROOT)), "report": str(report_path.relative_to(ROOT))}
    (REPORT_DIR / "cmin_sr_v1_v2_comparison_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

