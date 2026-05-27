from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    cmd=[sys.executable,str(ROOT/"experiments"/"run_zeta_noise_bridge.py"),"--quick","--num-samples","3"]
    proc=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,check=True)
    csv=ROOT/"outputs"/"tables"/"zeta_noise_bridge"/"zeta_noise_bridge.csv"
    result={"status":"ok","csv_exists":csv.exists(),"stdout_tail":proc.stdout.strip()[-300:]}
    p=ROOT/"outputs"/"reports"/"zeta_noise_bridge_smoke_test.json"; p.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
