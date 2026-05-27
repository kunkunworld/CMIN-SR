from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"experiments"))
from run_finite_sample_curvature_identifiability import run_grid, summarize

QGRIDS={"Q1":(0.5,1,1.5,2,2.5,3),"Q2":(0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3),"Q3":(0.5,1,1.5,2),"Q4":(0.25,0.5,0.75,1,1.25,1.5,1.75,2),"Q5":(1,1.5,2,2.5,3)}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--quick",action="store_true"); p.add_argument("--num-samples",type=int,default=None); p.add_argument("--T-values",default=None); p.add_argument("--qgrid-name",default="all"); p.add_argument("--estimator-name",default="structure_trimmed"); p.add_argument("--output-dir",default=None); p.add_argument("--seed",type=int,default=2026)
    a=p.parse_args(); out=Path(a.output_dir) if a.output_dir else ROOT/"outputs"; rd=out/"reports"/"finite_sample_identifiability"; td=out/"tables"/"finite_sample_identifiability"; fd=out/"figures"/"finite_sample_identifiability"; rd.mkdir(parents=True,exist_ok=True); td.mkdir(parents=True,exist_ok=True); fd.mkdir(parents=True,exist_ok=True)
    names=["Q1","Q3"] if a.quick else (list(QGRIDS) if a.qgrid_name=="all" else [a.qgrid_name]); n=a.num_samples or (6 if a.quick else 12); rows=[]
    for name in names:
        df=run_grid([0.4,0.6],[0,0.03,0.10,0.20],[1024,2048] if a.quick else [512,1024,2048],n,QGRIDS[name],(2,4,8,16,32,64),(a.estimator_name,),a.seed)
        s=summarize(df); s["qgrid"]=name; rows.append(s)
    res=pd.concat(rows,ignore_index=True); res.to_csv(td/"lambda2_recovery_by_qgrid.csv",index=False)
    agg=res.groupby("qgrid").mean(numeric_only=True).reset_index()
    fig,ax=plt.subplots(figsize=(6,4.5),constrained_layout=True); ax.bar(agg["qgrid"],agg["lambda2_corr"]); ax.set_ylabel("mean lambda2 corr"); fig.savefig(fd/"qgrid_lambda2_corr.png",dpi=220); plt.close(fig)
    report=rd/"qgrid_sensitivity_summary.md"; report.write_text("# Q-grid Sensitivity\n\n"+res.to_csv(index=False),encoding="utf-8")
    meta={"table":str((td/"lambda2_recovery_by_qgrid.csv").relative_to(ROOT)),"report":str(report.relative_to(ROOT))}; (rd/"qgrid_sensitivity_summary.json").write_text(json.dumps(meta,indent=2),encoding="utf-8"); print(json.dumps(meta,indent=2))
if __name__=="__main__": main()
