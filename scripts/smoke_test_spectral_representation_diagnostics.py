from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "run_spectral_representation_diagnostics.py"),
        "--length",
        "512",
        "--num-samples",
        "3",
        "--output-tag",
        "spectral_representation_diagnostics_smoke",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    summary = ROOT / "outputs" / "reports" / "spectral_representation_diagnostics_smoke" / "spectral_representation_diagnostics_summary.md"
    csv_path = ROOT / "outputs" / "tables" / "spectral_representation_diagnostics_smoke" / "spectral_representation_summary.csv"
    result = {
        "status": "ok",
        "summary_exists": summary.exists(),
        "csv_exists": csv_path.exists(),
        "stdout": proc.stdout.strip()[-400:],
    }
    out_path = ROOT / "outputs" / "reports" / "spectral_representation_diagnostics_smoke_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

