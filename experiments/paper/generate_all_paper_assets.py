from __future__ import annotations

import argparse

from _paper_utils import add_common_args, ensure_paper_dirs, run_python, write_note


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description="Generate all paper asset manifests, tables, and figures."))
    args = parser.parse_args()
    ensure_paper_dirs(args.output_dir)
    common = ["--output-dir", args.output_dir, "--seed", str(args.seed)]
    if args.quick:
        common.insert(0, "--quick")
    code = run_python("experiments/paper/collect_paper_assets.py", common)
    if code == 0:
        code = run_python("experiments/paper/generate_latex_tables.py", common)
    if code == 0:
        code = run_python("experiments/paper/generate_paper_figures.py", common)
    write_note(
        f"{args.output_dir}/summaries/paper_asset_generation.md",
        "Paper Asset Generation",
        [f"Quick mode: {args.quick}", "Ran collection, LaTeX table generation, and figure generation."],
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
