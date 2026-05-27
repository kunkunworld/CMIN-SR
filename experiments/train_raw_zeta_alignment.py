from __future__ import annotations

import argparse, copy, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from mrw_inverse.data import PROCESS_CODE_TO_NAME, RawZetaAlignmentDatasetConfig, generate_raw_zeta_alignment_dataset
from mrw_inverse.losses import zeta_alignment_loss
from mrw_inverse.models import CMINSRZetaAlignedModel

BASE = ROOT / "checkpoints" / "cmin" / "cmin_sr_v3_synthetic.pt"
CKPT = ROOT / "checkpoints" / "cmin" / "cmin_sr_zeta_aligned.pt"
REPORT_DIR = ROOT / "outputs" / "reports" / "raw_zeta_alignment_training"
TABLE_DIR = ROOT / "outputs" / "tables" / "raw_zeta_alignment_training"
FIG_DIR = ROOT / "outputs" / "figures" / "raw_zeta_alignment_training"

ORDER = ["x","process_code","T","zeta_target","zeta_target_mask","zeta_weight_by_q","H_true","lambda2_true","target_tail_instability","sample_id"]

def _loader(ds, bs, shuffle):
    tensors=[]
    for k in ORDER:
        dtype = torch.long if k in {"process_code","T","sample_id"} else torch.float32
        tensors.append(torch.tensor(ds[k], dtype=dtype))
    return DataLoader(TensorDataset(*tensors), batch_size=bs, shuffle=shuffle)

def _make(n,T,seed): return generate_raw_zeta_alignment_dataset(RawZetaAlignmentDatasetConfig(length=T,num_samples=n,seed=seed))

def _eval(model, loaders, device):
    model.eval(); rows=[]
    with torch.no_grad():
        for T, loader in loaders.items():
            for batch in loader:
                x, code, _T, target, mask, wq, h, lam, tail, sid = [b.to(device) for b in batch]
                out = model(x)
                mae = (torch.abs(out["zeta_emp"]-torch.nan_to_num(target))*mask*wq).sum(dim=1)/(mask*wq).sum(dim=1).clamp_min(1)
                sec = (out["zeta_emp"][:,:-2]-2*out["zeta_emp"][:,1:-1]+out["zeta_emp"][:,2:]).abs().mean(dim=1)
                for i in range(x.shape[0]):
                    rows.append({"T":T,"process_type":PROCESS_CODE_TO_NAME[int(code[i].cpu())],"zeta_mae":float(mae[i].cpu()),"second_diff_norm":float(sec[i].cpu()),"H_proj":float(out["H_proj"][i,0].cpu()),"lambda2_proj":float(out["lambda2_proj"][i,0].cpu()),"H_true":float(h[i,0].cpu()),"lambda2_true":float(lam[i,0].cpu()),"tail_instability":float(out["tail_instability"][i,0].cpu())})
    df=pd.DataFrame(rows)
    proc=df.groupby(["T","process_type"]).mean(numeric_only=True).reset_index()
    return df, proc

def main():
    p=argparse.ArgumentParser(); p.add_argument("--num-train",type=int,default=10000); p.add_argument("--num-val",type=int,default=2000); p.add_argument("--epochs",type=int,default=15); p.add_argument("--batch-size",type=int,default=32); p.add_argument("--lr",type=float,default=5e-4); p.add_argument("--seed",type=int,default=2026); p.add_argument("--device",default="cpu"); p.add_argument("--base-checkpoint",default=str(BASE))
    args=p.parse_args(); REPORT_DIR.mkdir(parents=True,exist_ok=True); TABLE_DIR.mkdir(parents=True,exist_ok=True); FIG_DIR.mkdir(parents=True,exist_ok=True); CKPT.parent.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); device=args.device if args.device!="cpu" and torch.cuda.is_available() else "cpu"
    Ts=[512,1024]; train_loaders={}; val_loaders={}
    for i,T in enumerate(Ts):
        train_loaders[T]=_loader(_make(args.num_train//2,T,args.seed+10*i),args.batch_size,True)
        val_loaders[T]=_loader(_make(args.num_val//2,T,args.seed+100+10*i),args.batch_size,False)
    model=CMINSRZetaAlignedModel().to(device); initialized=False
    bp=Path(args.base_checkpoint)
    if bp.exists():
        state=torch.load(bp,map_location="cpu"); model.load_state_dict(state["model_state_dict"] if isinstance(state,dict) and "model_state_dict" in state else state, strict=False); initialized=True
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    hist=[]; best=None; best_loss=float("inf"); best_proc=None; nan=False
    for ep in range(1,args.epochs+1):
        model.train(); losses=[]
        for T, loader in train_loaders.items():
            for batch in loader:
                x, code, _T, target, mask, wq, h, lam, tail, _sid = [b.to(device) for b in batch]
                out=model(x); loss=zeta_alignment_loss(out,code,target,mask,wq,h,lam,tail).total
                if torch.isnan(loss): nan=True; break
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach().cpu()))
            if nan: break
        _pred, proc=_eval(model,val_loaders,device); val=float(proc["zeta_mae"].mean())
        hist.append({"epoch":ep,"train_loss":float(np.mean(losses)),"val_zeta_mae":val})
        if val<best_loss: best_loss=val; best=copy.deepcopy(model.state_dict()); best_proc=proc.copy()
        if nan: break
    torch.save({"model_state_dict":best,"model_name":"cmin_sr_zeta_aligned","initialized_from_checkpoint":initialized,"config":vars(args)},CKPT)
    pd.DataFrame(hist).to_csv(TABLE_DIR/"train_history.csv",index=False); best_proc.to_csv(TABLE_DIR/"val_by_process.csv",index=False)
    fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True); hdf=pd.DataFrame(hist); ax.plot(hdf["epoch"],hdf["train_loss"],label="train"); ax.plot(hdf["epoch"],hdf["val_zeta_mae"],label="val zeta mae"); ax.legend(); fig.savefig(FIG_DIR/"loss_curve.png",dpi=220); plt.close(fig)
    report=REPORT_DIR/"raw_zeta_alignment_training_summary.md"; report.write_text("\n".join(["# Raw Zeta Alignment Training","",f"- Initialized from checkpoint: `{initialized}`",f"- Checkpoint: `{CKPT.relative_to(ROOT)}`",f"- NaN occurred: `{nan}`","",best_proc.to_csv(index=False)]),encoding="utf-8")
    meta={"checkpoint":str(CKPT.relative_to(ROOT)),"initialized_from_checkpoint":initialized,"nan_occurred":nan,"report":str(report.relative_to(ROOT))}
    (REPORT_DIR/"raw_zeta_alignment_training_summary.json").write_text(json.dumps(meta,indent=2),encoding="utf-8"); print(json.dumps(meta,indent=2))
if __name__=="__main__": main()
