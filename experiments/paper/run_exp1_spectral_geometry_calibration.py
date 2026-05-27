from __future__ import annotations

import argparse

from _paper_utils import add_common_args, ensure_paper_dirs, run_python, write_note


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description="Paper Exp1: spectral geometry calibration."))
    args = parser.parse_args()
    ensure_paper_dirs(args.output_dir)
    n = "600" if args.quick else "6000"
    code = run_python("experiments/evaluate_spectral_geometry_calibrator.py", ["--num-samples", n, "--seed", str(args.seed)])
    write_note(
        f"{args.output_dir}/summaries/exp1_spectral_geometry_calibration.md",
        "Exp1 Spectral Geometry Calibration",
        [
            "Source: experiments/evaluate_spectral_geometry_calibrator.py",
            "Purpose: evaluate the analytic zeta-space calibrator.",
            f"Quick mode: {args.quick}",
            "Outputs: outputs/tables/spectral_geometry_calibrator_eval and outputs/figures/spectral_geometry_calibrator_eval.",
        ],
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
