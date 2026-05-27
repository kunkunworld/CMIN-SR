from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _paper_utils import ROOT, add_common_args, ensure_paper_dirs, write_note


def _fmt(value) -> str:
    if pd.isna(value):
        return "--"
    if isinstance(value, float):
        return f"{value:.3f}"
    text = str(value)
    try:
        return f"{float(text):.3f}"
    except ValueError:
        return text.replace("_", "\\_")


def _latex_table(df: pd.DataFrame, columns: list[str], caption: str, label: str) -> str:
    available = [c for c in columns if c in df.columns]
    if not available:
        available = list(df.columns[: min(6, len(df.columns))])
    small = df[available].copy()
    header = " & ".join(c.replace("_", "\\_") for c in available) + r" \\"
    rows = [" & ".join(_fmt(v) for v in row) + r" \\" for row in small.itertuples(index=False, name=None)]
    body = "\n".join(rows)
    align = "l" * len(available)
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{align}}}\n\\toprule\n"
        f"{header}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


SPECS = [
    (
        "table1_process_family_diagnostics",
        "outputs/tables/curvature_preserving_zeta_eval/process_by_T_band.csv",
        ["T", "process_type", "lambda_band", "p_curved_cal", "p_mrw_cal", "p_mono_cal", "p_boundary_cal"],
        "Process-family calibrated diagnostics.",
    ),
    (
        "table2_mrw_mono_projection",
        "outputs/tables/cmin_sr_v3_eval/process_by_T.csv",
        ["T", "process_type", "p_scaling", "p_curved", "p_mrw", "p_mono", "residual_norm", "mono_residual_norm"],
        "MRW and monofractal projection diagnostics.",
    ),
    (
        "table3_ablation",
        "outputs/tables/cmin_sr_final_version_comparison/version_process_comparison.csv",
        ["version", "process_type", "T", "p_scaling", "p_curved", "p_mrw", "p_mono"],
        "CMIN-SR ablation and version comparison.",
    ),
    (
        "table4_identifiability",
        "outputs/tables/finite_sample_identifiability/lambda2_recovery_by_T.csv",
        ["T", "estimator", "lambda2_mae", "lambda2_corr", "high_lambda_detection_rate", "boundary_accuracy"],
        "Finite-sample lambda2 recovery.",
    ),
    (
        "table5_real_world_sanity",
        "outputs/tables/raw_market_surrogate_validation/raw_market_surrogate_summary.csv",
        ["asset", "window", "p_mrw", "p_scaling", "warning_rate"],
        "Optional real-world sanity check.",
    ),
]


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description="Generate booktabs-style LaTeX tables."))
    args = parser.parse_args()
    base = ensure_paper_dirs(args.output_dir)
    max_rows = 12 if args.quick else 30
    for name, src, columns, caption in SPECS:
        src_path = ROOT / src
        md_path = base / "tables" / f"{name}.md"
        tex_path = base / "tables" / f"{name}.tex"
        if not src_path.exists():
            write_note(md_path, name, [f"Missing source CSV: {src_path}", "Generate the corresponding experiment first."])
            continue
        df = pd.read_csv(src_path).head(max_rows)
        tex_path.write_text(
            _latex_table(df, columns, caption, f"tab:{name}"),
            encoding="utf-8",
        )
        write_note(md_path, name, [f"Source: {src}", f"Rows included: {len(df)}", caption])
    write_note(base / "latex" / "latex_table_manifest.md", "LaTeX Table Manifest", [f"Generated tables in {base / 'tables'}"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
