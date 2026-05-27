from __future__ import annotations

import argparse

from _paper_utils import add_common_args, ensure_paper_dirs, run_python, write_note


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description="Paper Exp2: process-family diagnostics."))
    args = parser.parse_args()
    ensure_paper_dirs(args.output_dir)
    n = "80" if args.quick else "600"
    code = run_python(
        "experiments/evaluate_curvature_preserving_zeta_alignment.py",
        ["--num-samples", n, "--seed", str(args.seed)],
    )
    write_note(
        f"{args.output_dir}/summaries/exp2_process_family_diagnostics.md",
        "Exp2 Process-Family Diagnostics",
        [
            "Source: experiments/evaluate_curvature_preserving_zeta_alignment.py",
            "Purpose: summarize calibrated diagnostics by stochastic process family.",
            f"Quick mode: {args.quick}",
            "Outputs: outputs/tables/curvature_preserving_zeta_eval.",
        ],
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
