from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    proc = subprocess.run(
        ["conda", "run", "-n", "for_codex", "python", "scripts/preprocess_market_price_csvs.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    summary_json = ROOT / "outputs" / "reports" / "market_price_preprocessing_summary.json"
    report_md = ROOT / "outputs" / "reports" / "market_price_preprocessing_summary.md"
    out = {
        "summary_json_exists": summary_json.exists(),
        "report_exists": report_md.exists(),
        "stdout_tail": proc.stdout[-500:],
    }
    out_path = ROOT / "outputs" / "reports" / "market_price_preprocessing_smoke_test.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
