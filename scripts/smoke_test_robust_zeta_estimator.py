from __future__ import annotations
import json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/"src"
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from mrw_inverse.models import estimate_robust_zeta
def main():
    out=estimate_robust_zeta(torch.randn(4,512))
    result={"status":"ok","shape":list(out.zeta_robust.shape),"has_nan":bool(torch.isnan(out.zeta_robust).any())}
    p=ROOT/"outputs"/"reports"/"robust_zeta_estimator_smoke_test.json"; p.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
