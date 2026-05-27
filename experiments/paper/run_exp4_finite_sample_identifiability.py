from __future__ import annotations

import argparse

from _paper_utils import add_common_args, ensure_paper_dirs, run_python, write_note


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description="Paper Exp4: finite-sample curvature identifiability."))
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args()
    ensure_paper_dirs(args.output_dir)
    n = args.num_samples if args.num_samples is not None else (8 if args.quick else 20)
    common = ["--num-samples", str(n), "--seed", str(args.seed)]
    if args.quick:
        common = ["--quick", *common]
    code = run_python("experiments/run_finite_sample_curvature_identifiability.py", common)
    if code == 0:
        code = run_python("experiments/run_scale_length_sensitivity.py", common)
    if code == 0:
        code = run_python("experiments/run_qgrid_sensitivity.py", common)
    write_note(
        f"{args.output_dir}/summaries/exp4_finite_sample_identifiability.md",
        "Exp4 Finite-Sample Identifiability",
        [
            "Sources: finite-sample, scale-length, and q-grid sensitivity scripts.",
            "Purpose: test whether deterministic estimators recover MRW curvature from finite raw samples.",
            f"Quick mode: {args.quick}",
            "Outputs: outputs/tables/finite_sample_identifiability.",
        ],
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
