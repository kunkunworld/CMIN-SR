from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "run_real_surrogate_validation.py"),
        "--window",
        "128",
        "--step",
        "120",
        "--max-series",
        "2",
        "--mode",
        "proxy",
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    gap = ROOT / "outputs" / "tables" / "real_surrogate_validation_proxy" / "real_surrogate_gap_table.csv"
    report = ROOT / "outputs" / "reports" / "real_surrogate_validation_proxy" / "real_surrogate_validation_summary.md"
    df = pd.read_csv(gap)
    out = {
        "gap_csv_exists": gap.exists(),
        "report_exists": report.exists(),
        "num_rows": int(len(df)),
        "has_nan_lambda2_gap": bool(df["pred_lambda2_gap_shuffle"].isna().all()),
    }
    out_path = ROOT / "outputs" / "reports" / "real_surrogate_validation_proxy" / "smoke_test_surrogate_validation.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
