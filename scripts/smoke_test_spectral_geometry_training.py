from __future__ import annotations
import json, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main():
    ckpt = ROOT/"checkpoints"/"cmin"/"spectral_geometry_calibrator.pt"
    backup = ROOT/"outputs"/"tmp"/"spectral_geometry_calibrator_smoke_backup.pt"
    backup.parent.mkdir(parents=True, exist_ok=True)
    had = ckpt.exists()
    if had: shutil.copy2(ckpt, backup)
    cmd = [sys.executable, str(ROOT/"experiments"/"train_spectral_geometry_calibrator.py"), "--num-train","512","--num-val","128","--epochs","1","--batch-size","64"]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
        result = {"status":"ok","checkpoint_exists":ckpt.exists(),"stdout_tail":proc.stdout.strip()[-300:]}
        p = ROOT/"outputs"/"reports"/"spectral_geometry_training_smoke_test.json"; p.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))
    finally:
        if had and backup.exists(): shutil.copy2(backup, ckpt)
if __name__ == "__main__": main()
