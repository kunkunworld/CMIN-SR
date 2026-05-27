from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    backup_dir = ROOT / "outputs" / "tmp" / "cmin_sr_v3_training_smoke_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    tracked = [
        ROOT / "checkpoints" / "cmin" / "cmin_sr_v3_synthetic.pt",
        ROOT / "outputs" / "reports" / "cmin_sr_v3_training" / "cmin_sr_v3_training_summary.md",
        ROOT / "outputs" / "reports" / "cmin_sr_v3_training" / "cmin_sr_v3_training_summary.json",
        ROOT / "outputs" / "tables" / "cmin_sr_v3_training" / "train_history.csv",
        ROOT / "outputs" / "tables" / "cmin_sr_v3_training" / "val_predictions.csv",
        ROOT / "outputs" / "tables" / "cmin_sr_v3_training" / "val_by_process.csv",
        ROOT / "outputs" / "tables" / "cmin_sr_v3_training" / "metrics_by_T.csv",
    ]
    backups = []
    for path in tracked:
        backup_path = backup_dir / path.name
        if path.exists():
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_path)
            backups.append((path, backup_path))
        else:
            backups.append((path, None))

    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "train_cmin_sr_v3.py"),
        "--num-train",
        "256",
        "--num-val",
        "128",
        "--epochs",
        "1",
        "--batch-size",
        "32",
    ]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
        checkpoint = ROOT / "checkpoints" / "cmin" / "cmin_sr_v3_synthetic.pt"
        summary = ROOT / "outputs" / "reports" / "cmin_sr_v3_training" / "cmin_sr_v3_training_summary.md"
        result = {
            "status": "ok",
            "checkpoint_exists": checkpoint.exists(),
            "summary_exists": summary.exists(),
            "stdout_tail": proc.stdout.strip()[-400:],
        }
        out_path = ROOT / "outputs" / "reports" / "cmin_sr_v3_training_smoke_test.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        for path, backup_path in backups:
            if backup_path is not None and backup_path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_path, path)


if __name__ == "__main__":
    main()
