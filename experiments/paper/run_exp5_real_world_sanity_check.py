from __future__ import annotations

import argparse
from pathlib import Path

from _paper_utils import ROOT, add_common_args, ensure_paper_dirs, run_python, write_note


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description="Paper Exp5: optional real-world sanity check."))
    parser.add_argument("--input", default="data/processed/raw_market_panel.csv")
    args = parser.parse_args()
    ensure_paper_dirs(args.output_dir)
    input_path = ROOT / args.input
    if not input_path.exists():
        write_note(
            f"{args.output_dir}/summaries/exp5_real_world_sanity_check.md",
            "Exp5 Real-World Sanity Check",
            [
                f"Skipped: input file not found: {input_path}",
                "This is optional and should not block the synthetic paper pipeline.",
                "Suggested command after preprocessing market data:",
                "python experiments/run_raw_market_surrogate_validation.py --mode auto",
            ],
        )
        return 0
    extra = ["--input", str(input_path), "--seed", str(args.seed)]
    if args.quick:
        extra.extend(["--assets", "SPY", "--step", "80"])
    return run_python("experiments/run_raw_market_surrogate_validation.py", extra)


if __name__ == "__main__":
    raise SystemExit(main())
