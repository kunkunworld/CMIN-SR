from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    cmd=[sys.executable,str(ROOT/"experiments"/"run_scale_length_sensitivity.py"),"--quick","--num-samples","2"]
    proc=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,check=True)
    csv=ROOT/"outputs"/"tables"/"finite_sample_identifiability"/"lambda2_recovery_by_scale_range.csv"
    result={"status":"ok","csv_exists":csv.exists(),"stdout_tail":proc.stdout.strip()[-300:]}
    p=ROOT/"outputs"/"reports"/"scale_length_sensitivity_smoke_test.json"; p.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
