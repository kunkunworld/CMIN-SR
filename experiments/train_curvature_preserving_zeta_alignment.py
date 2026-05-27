from __future__ import annotations

import argparse, copy, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/"src"
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from mrw_inverse.data import PROCESS_CODE_TO_NAME, RawZetaAlignmentDatasetConfig, generate_raw_zeta_alignment_dataset
from mrw_inverse.losses import curvature_preserving_zeta_loss
from mrw_inverse.models import CMINSRZetaAlignedModel

BASES=[ROOT/"checkpoints"/"cmin"/"cmin_sr_zeta_aligned.pt", ROOT/"checkpoints"/"cmin"/"cmin_sr_v3_synthetic.pt", ROOT/"checkpoints"/"cmin"/"cmin_sr_calibrated_synthetic.pt"]
CKPT=ROOT/"checkpoints"/"cmin"/"cmin_sr_zeta_curvature_preserved.pt"
REPORT_DIR=ROOT/"outputs"/"reports"/"curvature_preserving_zeta_training"
TABLE_DIR=ROOT/"outputs"/"tables"/"curvature_preserving_zeta_training"
FIG_DIR=ROOT/"outputs"/"figures"/"curvature_preserving_zeta_training"
ORDER=["x","process_code","T","zeta_target","zeta_target_mask","zeta_weight_by_q","H_true","lambda2_true","target_tail_instability","sample_id"]

def _loader(ds,bs,shuffle):
    ts=[]
    for k in ORDER:
        dtype=torch.long if k in {"process_code","T","sample_id"} else torch.float32
        ts.append(torch.tensor(ds[k],dtype=dtype))
    return DataLoader(TensorDataset(*ts),batch_size=bs,shuffle=shuffle)

def _make(n,T,seed):
    return generate_raw_zeta_alignment_dataset(RawZetaAlignmentDatasetConfig(length=T,num_samples=n,seed=seed,mrw_low_ratio=.12,mrw_mid_high_ratio=.48,fgn_ratio=.22,gaussian_ratio=.13,student_t_ratio=.025,stress_ratio=.025))

def _band(lam):
    if not np.isfinite(lam) or lam == 0: return "non_mrw"
    if lam < .03: return "low"
    if lam < .08: return "medium"
    return "high"

def _eval(model,loaders,device):
    model.eval(); rows=[]
    with torch.no_grad():
        for T,loader in loaders.items():
            for batch in loader:
                x,code,_T,target,mask,wq,h,lam,tail,sid=[b.to(device) for b in batch]
                out=model(x)
                mae=((out["zeta_emp"]-torch.nan_to_num(target)).abs()*mask*wq).sum(dim=1)/(mask*wq).sum(dim=1).clamp_min(1)
                d2=(out["zeta_emp"][:,:-2]-2*out["zeta_emp"][:,1:-1]+out["zeta_emp"][:,2:]).abs().mean(dim=1)
                d3=(out["zeta_emp"][:,:-3]-3*out["zeta_emp"][:,1:-2]+3*out["zeta_emp"][:,2:-1]-out["zeta_emp"][:,3:]).abs().mean(dim=1)
                for i in range(x.shape[0]):
                    l=float(lam[i,0].cpu())
                    rows.append({"T":T,"process_type":PROCESS_CODE_TO_NAME[int(code[i].cpu())],"lambda_band":_band(l),"zeta_mae":float(mae[i].cpu()),"second_diff_norm":float(d2[i].cpu()),"third_diff_norm":float(d3[i].cpu()),"H_proj":float(out["H_proj"][i,0].cpu()),"lambda2_proj":float(out["lambda2_proj"][i,0].cpu()),"H_true":float(h[i,0].cpu()),"lambda2_true":l,"residual_norm":float(out["residual_norm"][i,0].cpu()),"mono_residual_norm":float(out["mono_residual_norm"][i,0].cpu()),"tail_instability":float(out["tail_instability"][i,0].cpu())})
    df=pd.DataFrame(rows); proc=df.groupby(["T","process_type","lambda_band"]).mean(numeric_only=True).reset_index()
    return df,proc

def main():
    p=argparse.ArgumentParser(); p.add_argument("--num-train",type=int,default=10000); p.add_argument("--num-val",type=int,default=2000); p.add_argument("--epochs",type=int,default=10); p.add_argument("--batch-size",type=int,default=32); p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--seed",type=int,default=2026); p.add_argument("--device",default="cpu")
    args=p.parse_args(); REPORT_DIR.mkdir(parents=True,exist_ok=True); TABLE_DIR.mkdir(parents=True,exist_ok=True); FIG_DIR.mkdir(parents=True,exist_ok=True); CKPT.parent.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); device=args.device if args.device!="cpu" and torch.cuda.is_available() else "cpu"
    Ts=[512,1024]; train={}; val={}
    for i,T in enumerate(Ts):
        train[T]=_loader(_make(args.num_train//2,T,args.seed+i*10),args.batch_size,True)
        val[T]=_loader(_make(args.num_val//2,T,args.seed+100+i*10),args.batch_size,False)
    model=CMINSRZetaAlignedModel().to(device); init="none"
    for bp in BASES:
        if bp.exists():
            st=torch.load(bp,map_location="cpu"); model.load_state_dict(st["model_state_dict"] if isinstance(st,dict) and "model_state_dict" in st else st,strict=False); init=str(bp.relative_to(ROOT)); break
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    hist=[]; best=None; best_loss=float("inf"); best_proc=None; nan=False
    for ep in range(1,args.epochs+1):
        model.train(); losses=[]
        for T,loader in train.items():
            for batch in loader:
                x,code,_T,target,mask,wq,h,lam,tail,_sid=[b.to(device) for b in batch]
                out=model(x); loss=curvature_preserving_zeta_loss(out,code,target,mask,wq,h,lam,tail).total
                if torch.isnan(loss): nan=True; break
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach().cpu()))
            if nan: break
        _pred,proc=_eval(model,val,device); score=float(proc["zeta_mae"].mean()+0.2*proc["second_diff_norm"].mean())
        hist.append({"epoch":ep,"train_loss":float(np.mean(losses)),"val_score":score})
        if score<best_loss: best_loss=score; best=copy.deepcopy(model.state_dict()); best_proc=proc.copy()
        if nan: break
    torch.save({"model_state_dict":best,"model_name":"cmin_sr_zeta_curvature_preserved","initialized_from_checkpoint":init,"config":vars(args)},CKPT)
    pd.DataFrame(hist).to_csv(TABLE_DIR/"train_history.csv",index=False); best_proc.to_csv(TABLE_DIR/"val_by_process_band.csv",index=False)
    fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True); h=pd.DataFrame(hist); ax.plot(h["epoch"],h["train_loss"],label="train"); ax.plot(h["epoch"],h["val_score"],label="val"); ax.legend(); fig.savefig(FIG_DIR/"loss_curve.png",dpi=220); plt.close(fig)
    report=REPORT_DIR/"curvature_preserving_zeta_training_summary.md"; report.write_text("\n".join(["# Curvature-Preserving Zeta Training","",f"- Initialized from: `{init}`",f"- Checkpoint: `{CKPT.relative_to(ROOT)}`",f"- NaN occurred: `{nan}`","",best_proc.to_csv(index=False)]),encoding="utf-8")
    meta={"checkpoint":str(CKPT.relative_to(ROOT)),"initialized_from_checkpoint":init,"nan_occurred":nan,"report":str(report.relative_to(ROOT))}
    (REPORT_DIR/"curvature_preserving_zeta_training_summary.json").write_text(json.dumps(meta,indent=2),encoding="utf-8"); print(json.dumps(meta,indent=2))
if __name__=="__main__": main()
