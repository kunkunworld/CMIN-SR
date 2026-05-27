from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/"src"
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from mrw_inverse.data.analytic_spectrum_dataset import _diagnostics,_linear_zeta,_mono_fit,_mrw_fit,_mrw_zeta
from mrw_inverse.models import SpectralGeometryCalibrator

CKPT=ROOT/"checkpoints"/"cmin"/"spectral_geometry_calibrator.pt"
REPORT_DIR=ROOT/"outputs"/"reports"/"zeta_noise_bridge"; TABLE_DIR=ROOT/"outputs"/"tables"/"zeta_noise_bridge"; FIG_DIR=ROOT/"outputs"/"figures"/"zeta_noise_bridge"
def _smooth_noise(q,rng,level): return level*(rng.normal()* (q-q.mean()) + rng.normal()*((q-q.mean())**2) + rng.normal(size=q.shape)*0.3)
def _cal(model,z,q,device):
    _hm,zm,mr=_mono_fit(q,z); _h,lp,zr,rr=_mrw_fit(q,z); d=_diagnostics(q,z,zm,zr,mr,rr,lp)
    args=[z[None,:],zm[None,:],zr[None,:],[[mr]],[[rr]],[[d["mrw_vs_mono_gain"]]],[[d["normalized_mrw_gain"]]],[[d["curvature_score"]]],[[d["linearity_score"]]],[[d["boundary_score"]]],[[0.0]]]
    tensors=[torch.tensor(a,dtype=torch.float32,device=device) for a in args]
    with torch.no_grad(): out=model(*tensors)
    return {k:float(v.item()) for k,v in out.items() if k.startswith("p_")}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--quick",action="store_true"); p.add_argument("--num-samples",type=int,default=None); p.add_argument("--T-values",default=""); p.add_argument("--qgrid-name",default="Q1"); p.add_argument("--estimator-name",default=""); p.add_argument("--output-dir",default=None); p.add_argument("--seed",type=int,default=2026)
    a=p.parse_args(); out=Path(a.output_dir) if a.output_dir else ROOT/"outputs"; rd=out/"reports"/"zeta_noise_bridge"; td=out/"tables"/"zeta_noise_bridge"; fd=out/"figures"/"zeta_noise_bridge"; rd.mkdir(parents=True,exist_ok=True); td.mkdir(parents=True,exist_ok=True); fd.mkdir(parents=True,exist_ok=True)
    if not CKPT.exists(): print(json.dumps({"status":"missing_calibrator"},indent=2)); return
    model=SpectralGeometryCalibrator(); model.load_state_dict(torch.load(CKPT,map_location="cpu")["model_state_dict"],strict=False); model.eval(); device="cpu"
    rng=np.random.default_rng(a.seed); q=np.asarray([0.5,1,1.5,2,2.5,3],dtype=np.float32); levels=[0,0.005,0.01,0.02,0.05,0.10]; n=a.num_samples or (20 if a.quick else 100)
    rows=[]
    for level in levels:
        for typ in ["linear_mono","boundary_mrw","curved_mrw"]:
            for _ in range(n):
                H=float(rng.choice([0.4,0.6])); lam=0 if typ=="linear_mono" else (0.02 if typ=="boundary_mrw" else 0.12)
                base=_linear_zeta(q,H) if lam==0 else _mrw_zeta(q,H,lam)
                for noise_kind in ["smooth","highq","jagged","highq_bias"]:
                    noise=_smooth_noise(q,rng,level)
                    if noise_kind=="highq": noise=noise*np.maximum(q-1.5,0)
                    if noise_kind=="jagged": noise=level*rng.normal(size=q.shape)
                    if noise_kind=="highq_bias": noise=-level*np.maximum(q-2.0,0)**2
                    outv=_cal(model,(base+noise).astype(np.float32),q,device); rows.append({"spectrum_type":typ,"noise_level":level,"noise_kind":noise_kind,**outv})
    df=pd.DataFrame(rows); df.to_csv(td/"zeta_noise_bridge.csv",index=False)
    summ=df.groupby(["noise_level","spectrum_type"]).mean(numeric_only=True).reset_index(); summ.to_csv(td/"zeta_noise_bridge_summary_table.csv",index=False)
    pivot=summ.pivot_table(index="noise_level",columns="spectrum_type",values="p_mrw")
    margin=pivot.get("curved_mrw",pd.Series(dtype=float))-pivot.get("linear_mono",pd.Series(dtype=float)); margin.to_csv(td/"separation_margin_vs_noise.csv")
    fig,ax=plt.subplots(figsize=(7,4.8),constrained_layout=True)
    for typ,g in summ.groupby("spectrum_type"): ax.plot(g["noise_level"],g["p_mrw"],marker="o",label=typ)
    ax.set_xlabel("zeta noise level"); ax.set_ylabel("p_MRW"); ax.legend(); fig.savefig(fd/"pmrw_vs_noise.png",dpi=220); fig.savefig(fd/"pmrw_vs_noise.pdf"); plt.close(fig)
    report=rd/"zeta_noise_bridge_summary.md"; report.write_text("# Zeta Noise Bridge\n\n"+summ.to_csv(index=False),encoding="utf-8")
    meta={"table":str((td/"zeta_noise_bridge.csv").relative_to(ROOT)),"report":str(report.relative_to(ROOT))}; (rd/"zeta_noise_bridge_summary.json").write_text(json.dumps(meta,indent=2),encoding="utf-8"); print(json.dumps(meta,indent=2))
if __name__=="__main__": main()
