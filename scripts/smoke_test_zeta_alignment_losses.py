from __future__ import annotations
import json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/"src"
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from mrw_inverse.data import RawZetaAlignmentDatasetConfig, generate_raw_zeta_alignment_dataset
from mrw_inverse.losses import zeta_alignment_loss
from mrw_inverse.models import CMINSRZetaAlignedModel
def main():
    ds=generate_raw_zeta_alignment_dataset(RawZetaAlignmentDatasetConfig(length=256,num_samples=16,seed=2))
    model=CMINSRZetaAlignedModel()
    out=model(torch.tensor(ds["x"],dtype=torch.float32))
    loss=zeta_alignment_loss(out,torch.tensor(ds["process_code"],dtype=torch.long),torch.tensor(ds["zeta_target"],dtype=torch.float32),torch.tensor(ds["zeta_target_mask"],dtype=torch.float32),torch.tensor(ds["zeta_weight_by_q"],dtype=torch.float32),torch.tensor(ds["H_true"],dtype=torch.float32),torch.tensor(ds["lambda2_true"],dtype=torch.float32),torch.tensor(ds["target_tail_instability"],dtype=torch.float32))
    result={"status":"ok","loss":float(loss.total),"isfinite":bool(torch.isfinite(loss.total))}
    p=ROOT/"outputs"/"reports"/"zeta_alignment_losses_smoke_test.json"; p.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
