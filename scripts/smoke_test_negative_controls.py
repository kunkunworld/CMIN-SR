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
        str(ROOT / "experiments" / "run_negative_controls.py"),
        "--length",
        "512",
        "--num-samples",
        "3",
        "--mode",
        "proxy",
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    detail = ROOT / "outputs" / "tables" / "negative_controls_proxy" / "negative_controls_samples.csv"
    report = ROOT / "outputs" / "reports" / "negative_controls_proxy" / "negative_controls_summary.md"
    df = pd.read_csv(detail)
    out = {
        "detail_csv_exists": detail.exists(),
        "report_exists": report.exists(),
        "num_rows": int(len(df)),
        "has_nan_pred_H": bool(df["pred_H"].isna().any()),
        "has_nan_pred_lambda2": bool(df["pred_lambda2"].isna().any()),
        "modes": sorted(df["mode"].dropna().unique().tolist()),
    }
    out_path = ROOT / "outputs" / "reports" / "negative_controls_proxy" / "smoke_test_negative_controls.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
