from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    proc = subprocess.run(
        ["conda", "run", "-n", "for_codex", "python", "experiments/run_raw_market_surrogate_validation.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    summary_json = ROOT / "outputs" / "reports" / "raw_market_surrogate_validation" / "raw_market_surrogate_validation_summary.json"
    out = {
        "summary_json_exists": summary_json.exists(),
        "stdout_tail": proc.stdout[-500:],
    }
    out_path = ROOT / "outputs" / "reports" / "raw_market_surrogate_validation" / "smoke_test_raw_market_surrogate_validation.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
