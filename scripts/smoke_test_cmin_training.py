from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cmd = [
        "conda",
        "run",
        "-n",
        "for_codex",
        "python",
        "experiments/train_cmin_synthetic.py",
        "--num-train",
        "96",
        "--num-val",
        "32",
        "--batch-size",
        "16",
        "--epochs",
        "1",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    checkpoint = ROOT / "checkpoints" / "cmin" / "cmin_tiny_synthetic.pt"
    summary = ROOT / "outputs" / "reports" / "cmin_training" / "cmin_tiny_training_summary.md"
    history = ROOT / "outputs" / "tables" / "cmin_training" / "train_history.csv"
    out = {
        "checkpoint_exists": checkpoint.exists(),
        "summary_exists": summary.exists(),
        "history_exists": history.exists(),
        "stdout_tail": proc.stdout[-500:],
    }
    report_path = ROOT / "outputs" / "reports" / "cmin_training" / "smoke_test_cmin_training.json"
    report_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
