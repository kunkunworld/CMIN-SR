from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    checkpoint = ROOT / "checkpoints" / "cmin" / "cmin_tiny_synthetic.pt"
    if not checkpoint.exists():
        subprocess.run(
            [
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
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

    proc = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "for_codex",
            "python",
            "experiments/evaluate_cmin_synthetic.py",
            "--num-samples",
            "48",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    summary = ROOT / "outputs" / "reports" / "cmin_eval" / "cmin_eval_summary.md"
    metrics = ROOT / "outputs" / "tables" / "cmin_eval" / "cmin_eval_metrics.csv"
    out = {
        "checkpoint_exists": checkpoint.exists(),
        "summary_exists": summary.exists(),
        "metrics_exists": metrics.exists(),
        "stdout_tail": proc.stdout[-500:],
    }
    report_path = ROOT / "outputs" / "reports" / "cmin_eval" / "smoke_test_cmin_checkpoint.json"
    report_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
