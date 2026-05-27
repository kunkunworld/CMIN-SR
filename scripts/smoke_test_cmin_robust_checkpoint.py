from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    checkpoint = ROOT / "checkpoints" / "cmin" / "cmin_robust_synthetic.pt"
    if not checkpoint.exists():
        subprocess.run(
            [
                "conda",
                "run",
                "-n",
                "for_codex",
                "python",
                "experiments/train_cmin_robust.py",
                "--num-train",
                "192",
                "--num-val",
                "96",
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
            "experiments/evaluate_cmin_robust.py",
            "--num-samples",
            "120",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    summary = ROOT / "outputs" / "reports" / "cmin_robust_eval" / "cmin_robust_eval_summary.md"
    metrics = ROOT / "outputs" / "tables" / "cmin_robust_eval" / "cmin_robust_eval_metrics.csv"
    out = {
        "checkpoint_exists": checkpoint.exists(),
        "summary_exists": summary.exists(),
        "metrics_exists": metrics.exists(),
        "stdout_tail": proc.stdout[-500:],
    }
    report_path = ROOT / "outputs" / "reports" / "cmin_robust_eval" / "smoke_test_cmin_robust_checkpoint.json"
    report_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
