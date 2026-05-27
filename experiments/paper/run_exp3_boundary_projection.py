from __future__ import annotations

import argparse

from _paper_utils import add_common_args, ensure_paper_dirs, run_python, write_note


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description="Paper Exp3: MRW-vs-mono boundary/projection comparison."))
    args = parser.parse_args()
    ensure_paper_dirs(args.output_dir)
    n = "80" if args.quick else "600"
    code = run_python(
        "experiments/evaluate_cmin_sr_v3.py",
        ["--num-samples", n, "--seed", str(args.seed)],
    )
    write_note(
        f"{args.output_dir}/summaries/exp3_boundary_projection.md",
        "Exp3 Boundary Projection",
        [
            "Source: experiments/evaluate_cmin_sr_v3.py",
            "Purpose: report projection residuals and MRW-vs-monofractal boundary behavior.",
            f"Quick mode: {args.quick}",
            "Outputs: outputs/tables/cmin_sr_v3_eval.",
        ],
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
