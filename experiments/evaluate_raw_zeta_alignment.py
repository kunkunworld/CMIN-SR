from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/"src"
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from mrw_inverse.data import PROCESS_CODE_TO_NAME, RawZetaAlignmentDatasetConfig, generate_raw_zeta_alignment_dataset
from mrw_inverse.models import CMINSRZetaAlignedModel, SpectralGeometryCalibrator, apply_pretrained_calibrator

CKPT=ROOT/"checkpoints"/"cmin"/"cmin_sr_zeta_aligned.pt"
CAL=ROOT/"checkpoints"/"cmin"/"spectral_geometry_calibrator.pt"
REPORT_DIR=ROOT/"outputs"/"reports"/"raw_zeta_alignment_eval"
TABLE_DIR=ROOT/"outputs"/"tables"/"raw_zeta_alignment_eval"
FIG_DIR=ROOT/"outputs"/"figures"/"raw_zeta_alignment_eval"

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",default=str(CKPT)); p.add_argument("--calibrator",default=str(CAL)); p.add_argument("--num-samples",type=int,default=600); p.add_argument("--t-eval",nargs="*",type=int,default=[256,512,1024,2048]); p.add_argument("--seed",type=int,default=6060); p.add_argument("--device",default="cpu")
    args=p.parse_args(); REPORT_DIR.mkdir(parents=True,exist_ok=True); TABLE_DIR.mkdir(parents=True,exist_ok=True); FIG_DIR.mkdir(parents=True,exist_ok=True)
    if not Path(args.checkpoint).exists():
        out={"status":"missing_checkpoint"}; print(json.dumps(out,indent=2)); return
    device=args.device if args.device!="cpu" and torch.cuda.is_available() else "cpu"
    state=torch.load(args.checkpoint,map_location="cpu"); model=CMINSRZetaAlignedModel(); model.load_state_dict(state["model_state_dict"],strict=False); model.to(device).eval()
    cal=None
    if Path(args.calibrator).exists():
        s=torch.load(args.calibrator,map_location="cpu"); cal=SpectralGeometryCalibrator(); cal.load_state_dict(s["model_state_dict"],strict=False); cal.to(device).eval()
    rows=[]
    with torch.no_grad():
        for ti,T in enumerate(args.t_eval):
            ds=generate_raw_zeta_alignment_dataset(RawZetaAlignmentDatasetConfig(length=T,num_samples=args.num_samples,seed=args.seed+ti*13))
            for i in range(args.num_samples):
                x=torch.tensor(ds["x"][i:i+1],dtype=torch.float32,device=device); out=model(x)
                target=torch.tensor(ds["zeta_target"][i:i+1],dtype=torch.float32,device=device); mask=torch.tensor(ds["zeta_target_mask"][i:i+1],dtype=torch.float32,device=device); wq=torch.tensor(ds["zeta_weight_by_q"][i:i+1],dtype=torch.float32,device=device)
                mae=float(((out["zeta_emp"]-torch.nan_to_num(target)).abs()*mask*wq).sum().cpu()/(mask*wq).sum().clamp_min(1).cpu())
                sec=float((out["zeta_emp"][:,:-2]-2*out["zeta_emp"][:,1:-1]+out["zeta_emp"][:,2:]).abs().mean().cpu())
                proc=PROCESS_CODE_TO_NAME[int(ds["process_code"][i])]
                row={"T":T,"process_type":proc,"zeta_mae":mae,"second_diff_norm":sec,"H_proj":float(out["H_proj"].item()),"lambda2_proj":float(out["lambda2_proj"].item()),"H_true":float(ds["H_true"][i,0]),"lambda2_true":float(ds["lambda2_true"][i,0]),"tail_instability":float(out["tail_instability"].item()),"residual_norm":float(out["residual_norm"].item()),"mono_residual_norm":float(out["mono_residual_norm"].item())}
                if cal is not None:
                    co=apply_pretrained_calibrator(out,cal); row.update({k:float(v.item()) for k,v in co.items()})
                rows.append(row)
    df=pd.DataFrame(rows); df.to_csv(TABLE_DIR/"predictions.csv",index=False)
    proc=df.groupby(["T","process_type"]).mean(numeric_only=True).reset_index(); proc.to_csv(TABLE_DIR/"process_by_T.csv",index=False)
    plot=proc[proc["T"]==1024] if (proc["T"]==1024).any() else proc
    if "p_mrw_cal" in plot.columns:
        fig,ax=plt.subplots(figsize=(8,4.8),constrained_layout=True); focus=plot[plot["process_type"].isin(["MRW","Low-lambda2 MRW","fGn","iid Gaussian","iid Student-t","GARCH(1,1)","Regime-switching Gaussian"])]
        ax.bar(focus["process_type"],focus["p_mrw_cal"]); ax.tick_params(axis="x",rotation=30); ax.set_ylabel("p_MRW_cal"); fig.savefig(FIG_DIR/"pmrw_cal_by_process.png",dpi=220); plt.close(fig)
    report=REPORT_DIR/"raw_zeta_alignment_eval_summary.md"; report.write_text("\n".join(["# Raw Zeta Alignment Evaluation","",proc.to_csv(index=False)]),encoding="utf-8")
    meta={"predictions":str((TABLE_DIR/"predictions.csv").relative_to(ROOT)),"process_by_T":str((TABLE_DIR/"process_by_T.csv").relative_to(ROOT)),"report":str(report.relative_to(ROOT))}
    (REPORT_DIR/"raw_zeta_alignment_eval_summary.json").write_text(json.dumps(meta,indent=2),encoding="utf-8"); print(json.dumps(meta,indent=2))
if __name__=="__main__": main()
