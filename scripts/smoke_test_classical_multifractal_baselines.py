from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "run_classical_multifractal_baseline_comparison.py"),
        "--quick",
        "--num-samples",
        "2",
        "--T-values",
        "512",
        "--output-dir",
        str(ROOT / "outputs" / "classical_baseline_smoke"),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=120)
    ok = proc.returncode == 0
    table = ROOT / "outputs" / "classical_baseline_smoke" / "tables" / "classical_multifractal_baselines" / "classical_baseline_lambda2_recovery_by_T.csv"
    report = ROOT / "outputs" / "classical_baseline_smoke" / "reports" / "classical_multifractal_baselines" / "classical_multifractal_baseline_summary.md"
    result = {
        "ok": ok and table.exists() and report.exists(),
        "returncode": proc.returncode,
        "table_exists": table.exists(),
        "report_exists": report.exists(),
        "stdout_tail": proc.stdout[-1000:],
        "stderr_tail": proc.stderr[-1000:],
    }
    out = ROOT / "outputs" / "reports" / "classical_baseline_smoke_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
