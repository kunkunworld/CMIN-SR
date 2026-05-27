from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "evaluate_cmin_sr.py"),
        "--num-samples",
        "128",
        "--t-eval",
        "256",
        "512",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    summary = ROOT / "outputs" / "reports" / "cmin_sr_eval" / "cmin_sr_eval_summary.md"
    metrics = ROOT / "outputs" / "tables" / "cmin_sr_eval" / "metrics_by_T.csv"
    result = {
        "status": "ok",
        "summary_exists": summary.exists(),
        "metrics_exists": metrics.exists(),
        "stdout_tail": proc.stdout.strip()[-400:],
    }
    out_path = ROOT / "outputs" / "reports" / "cmin_sr_eval_smoke_test.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

