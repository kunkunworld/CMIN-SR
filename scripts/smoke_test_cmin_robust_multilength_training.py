from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    proc = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "for_codex",
            "python",
            "experiments/train_cmin_robust_multilength.py",
            "--num-train",
            "256",
            "--num-val",
            "128",
            "--epochs",
            "1",
            "--batch-size",
            "16",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    checkpoint = ROOT / "checkpoints" / "cmin" / "cmin_robust_multilength.pt"
    summary = ROOT / "outputs" / "reports" / "cmin_robust_multilength_training" / "cmin_robust_multilength_training_summary.md"
    out = {
        "checkpoint_exists": checkpoint.exists(),
        "summary_exists": summary.exists(),
        "stdout_tail": proc.stdout[-500:],
    }
    report_path = ROOT / "outputs" / "reports" / "cmin_robust_multilength_training" / "smoke_test_cmin_robust_multilength_training.json"
    report_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
