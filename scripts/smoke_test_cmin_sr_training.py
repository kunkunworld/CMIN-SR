from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "train_cmin_sr.py"),
        "--num-train",
        "256",
        "--num-val",
        "128",
        "--epochs",
        "1",
        "--batch-size",
        "32",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    checkpoint = ROOT / "checkpoints" / "cmin" / "cmin_sr_synthetic.pt"
    summary = ROOT / "outputs" / "reports" / "cmin_sr_training" / "cmin_sr_training_summary.md"
    result = {
        "status": "ok",
        "checkpoint_exists": checkpoint.exists(),
        "summary_exists": summary.exists(),
        "stdout_tail": proc.stdout.strip()[-400:],
    }
    out_path = ROOT / "outputs" / "reports" / "cmin_sr_training_smoke_test.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

