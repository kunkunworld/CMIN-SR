from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main():
    cmd = [sys.executable, str(ROOT/"experiments"/"apply_spectral_calibrator_to_cmin_sr.py"), "--num-samples","64","--t-eval","512"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    summary = ROOT/"outputs"/"reports"/"cmin_sr_spectrum_calibrated_eval"/"cmin_sr_spectrum_calibrated_eval_summary.md"
    result = {"status":"ok","summary_exists":summary.exists(),"stdout_tail":proc.stdout.strip()[-300:]}
    p = ROOT/"outputs"/"reports"/"apply_spectral_calibrator_to_cmin_sr_smoke_test.json"; p.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))
if __name__ == "__main__": main()
