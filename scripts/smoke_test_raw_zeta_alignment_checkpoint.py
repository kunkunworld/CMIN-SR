from __future__ import annotations
import json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/"src"
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from mrw_inverse.models import CMINSRZetaAlignedModel
def main():
    ckpt=ROOT/"checkpoints"/"cmin"/"cmin_sr_zeta_aligned.pt"
    if not ckpt.exists(): result={"status":"missing_checkpoint"}
    else:
        state=torch.load(ckpt,map_location="cpu"); m=CMINSRZetaAlignedModel(); m.load_state_dict(state["model_state_dict"],strict=False); out=m(torch.randn(2,512)); result={"status":"ok","zeta_shape":list(out["zeta_emp"].shape)}
    p=ROOT/"outputs"/"reports"/"raw_zeta_alignment_checkpoint_smoke_test.json"; p.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
