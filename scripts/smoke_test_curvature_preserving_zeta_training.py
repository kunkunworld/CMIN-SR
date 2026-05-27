from __future__ import annotations
import json, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    ckpt=ROOT/"checkpoints"/"cmin"/"cmin_sr_zeta_curvature_preserved.pt"; backup=ROOT/"outputs"/"tmp"/"curv_preserve_smoke_backup.pt"; backup.parent.mkdir(parents=True,exist_ok=True); had=ckpt.exists()
    if had: shutil.copy2(ckpt,backup)
    cmd=[sys.executable,str(ROOT/"experiments"/"train_curvature_preserving_zeta_alignment.py"),"--num-train","128","--num-val","64","--epochs","1","--batch-size","16"]
    try:
        proc=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,check=True)
        result={"status":"ok","checkpoint_exists":ckpt.exists(),"stdout_tail":proc.stdout.strip()[-300:]}
        p=ROOT/"outputs"/"reports"/"curvature_preserving_zeta_training_smoke_test.json"; p.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
    finally:
        if had and backup.exists(): shutil.copy2(backup,ckpt)
if __name__=="__main__": main()
