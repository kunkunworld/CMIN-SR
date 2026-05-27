from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"experiments"))
from run_finite_sample_curvature_identifiability import run_grid, summarize

def main():
    p=argparse.ArgumentParser(); p.add_argument("--quick",action="store_true"); p.add_argument("--num-samples",type=int,default=None); p.add_argument("--T-values",default=None); p.add_argument("--qgrid-name",default="Q1"); p.add_argument("--estimator-name",default="structure_trimmed"); p.add_argument("--output-dir",default=None); p.add_argument("--seed",type=int,default=2026)
    a=p.parse_args(); out=Path(a.output_dir) if a.output_dir else ROOT/"outputs"; rd=out/"reports"/"finite_sample_identifiability"; td=out/"tables"/"finite_sample_identifiability"; fd=out/"figures"/"finite_sample_identifiability"; rd.mkdir(parents=True,exist_ok=True); td.mkdir(parents=True,exist_ok=True); fd.mkdir(parents=True,exist_ok=True)
    Ts=[512,1024] if a.quick else [512,1024,2048,4096,8192]; n=a.num_samples or (6 if a.quick else 12)
    scale_sets={"A":(2,4,8,16,32,64),"B":(4,8,16,32,64,128),"C":(2,4,8,16,32,64,128),"D":(2,4,8,16,32,64,128,256)}
    rows=[]
    for name,scales in scale_sets.items():
        df=run_grid([0.4,0.6],[0,0.03,0.10,0.20],Ts,n,(0.5,1,1.5,2,2.5,3),scales,(a.estimator_name,),a.seed)
        s=summarize(df); s["scale_set"]=name; rows.append(s)
    res=pd.concat(rows,ignore_index=True); res.to_csv(td/"lambda2_recovery_by_scale_range.csv",index=False)
    fig,ax=plt.subplots(figsize=(7,4.8),constrained_layout=True)
    for k,g in res.groupby("scale_set"): ax.plot(g["T"],g["lambda2_corr"],marker="o",label=k)
    ax.set_xscale("log",base=2); ax.set_ylabel("lambda2 corr"); ax.set_xlabel("T"); ax.legend(); fig.savefig(fd/"scale_length_lambda2_corr.png",dpi=220); plt.close(fig)
    report=rd/"scale_length_sensitivity_summary.md"; report.write_text("# Scale/Length Sensitivity\n\n"+res.to_csv(index=False),encoding="utf-8")
    meta={"table":str((td/"lambda2_recovery_by_scale_range.csv").relative_to(ROOT)),"report":str(report.relative_to(ROOT))}; (rd/"scale_length_sensitivity_summary.json").write_text(json.dumps(meta,indent=2),encoding="utf-8"); print(json.dumps(meta,indent=2))
if __name__=="__main__": main()
