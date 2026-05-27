from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    backup_dir = ROOT / "outputs" / "tmp" / "cmin_sr_boundary_calibrated_eval_smoke_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    tracked = [
        ROOT / "outputs" / "reports" / "cmin_sr_boundary_calibrated_eval" / "cmin_sr_boundary_calibrated_eval_summary.md",
        ROOT / "outputs" / "reports" / "cmin_sr_boundary_calibrated_eval" / "cmin_sr_boundary_calibrated_eval_summary.json",
        ROOT / "outputs" / "tables" / "cmin_sr_boundary_calibrated_eval" / "metrics_by_T.csv",
        ROOT / "outputs" / "tables" / "cmin_sr_boundary_calibrated_eval" / "process_by_T.csv",
        ROOT / "outputs" / "tables" / "cmin_sr_boundary_calibrated_eval" / "boundary_sweep_summary.csv",
    ]
    backups = []
    for path in tracked:
        backup = backup_dir / path.name
        if path.exists():
            shutil.copy2(path, backup)
            backups.append((path, backup))
        else:
            backups.append((path, None))
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "evaluate_cmin_sr_boundary_calibrated.py"),
        "--num-samples",
        "128",
        "--boundary-groups-per-h",
        "2",
        "--t-eval",
        "256",
        "512",
    ]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
        summary = ROOT / "outputs" / "reports" / "cmin_sr_boundary_calibrated_eval" / "cmin_sr_boundary_calibrated_eval_summary.md"
        result = {"status": "ok", "summary_exists": summary.exists(), "stdout_tail": proc.stdout.strip()[-400:]}
        out = ROOT / "outputs" / "reports" / "cmin_sr_boundary_calibrated_eval_smoke_test.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        for path, backup in backups:
            if backup is not None and backup.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, path)


if __name__ == "__main__":
    main()
