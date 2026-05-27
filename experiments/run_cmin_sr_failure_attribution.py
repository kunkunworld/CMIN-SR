from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/"src"
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from mrw_inverse.analysis import estimate_curvature_identifiability
from mrw_inverse.data.process_generators import generate_process_sample
from mrw_inverse.data.analytic_spectrum_dataset import _diagnostics,_linear_zeta,_mono_fit,_mrw_fit,_mrw_zeta
from mrw_inverse.models import CMINSRZetaAlignedModel, SpectralGeometryCalibrator

REPORT_DIR=ROOT/"outputs"/"reports"/"cmin_sr_failure_attribution"; TABLE_DIR=ROOT/"outputs"/"tables"/"cmin_sr_failure_attribution"; FIG_DIR=ROOT/"outputs"/"figures"/"cmin_sr_failure_attribution"
CAL=ROOT/"checkpoints"/"cmin"/"spectral_geometry_calibrator.pt"; NEURAL=ROOT/"checkpoints"/"cmin"/"cmin_sr_zeta_curvature_preserved.pt"
def _cal(model,z,q):
    _hm,zm,mr=_mono_fit(q,z); _h,lp,zr,rr=_mrw_fit(q,z); d=_diagnostics(q,z,zm,zr,mr,rr,lp)
    tensors=[torch.tensor(a,dtype=torch.float32) for a in [z[None,:],zm[None,:],zr[None,:],[[mr]],[[rr]],[[d["mrw_vs_mono_gain"]]],[[d["normalized_mrw_gain"]]],[[d["curvature_score"]]],[[d["linearity_score"]]],[[d["boundary_score"]]],[[0.0]]]]
    with torch.no_grad(): out=model(*tensors)
    return {k:float(v.item()) for k,v in out.items() if k.startswith("p_")}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--quick",action="store_true"); p.add_argument("--num-samples",type=int,default=None); p.add_argument("--T-values",default="1024"); p.add_argument("--qgrid-name",default="Q1"); p.add_argument("--estimator-name",default="structure_trimmed"); p.add_argument("--output-dir",default=None); p.add_argument("--seed",type=int,default=2026)
    a=p.parse_args(); out=Path(a.output_dir) if a.output_dir else ROOT/"outputs"; rd=out/"reports"/"cmin_sr_failure_attribution"; td=out/"tables"/"cmin_sr_failure_attribution"; fd=out/"figures"/"cmin_sr_failure_attribution"; rd.mkdir(parents=True,exist_ok=True); td.mkdir(parents=True,exist_ok=True); fd.mkdir(parents=True,exist_ok=True)
    cal=SpectralGeometryCalibrator(); cal.load_state_dict(torch.load(CAL,map_location="cpu")["model_state_dict"],strict=False); cal.eval()
    neural=None
    if NEURAL.exists(): neural=CMINSRZetaAlignedModel(); neural.load_state_dict(torch.load(NEURAL,map_location="cpu")["model_state_dict"],strict=False); neural.eval()
    rng=np.random.default_rng(a.seed); q=np.asarray([0.5,1,1.5,2,2.5,3],dtype=np.float32); n=a.num_samples or (12 if a.quick else 60); rows=[]
    for proc,H,lam in [("fGn",0.6,0.0),("iid Gaussian",0.5,0.0),("MRW",0.6,0.12),("Low-lambda2 MRW",0.6,0.02)]:
        for i in range(n):
            sample=generate_process_sample(proc,1024,rng,h=H if proc!="iid Gaussian" else None,lambda2=lam if proc in {"MRW","Low-lambda2 MRW"} else None,q_grid=q)
            z_analytic=_linear_zeta(q,H) if proc in {"fGn","iid Gaussian"} else _mrw_zeta(q,H,lam)
            for level,z in [("analytic",z_analytic)]:
                rows.append({"process_type":proc,"level":level,"lambda2_true":lam,**_cal(cal,z,q),"zeta_mae":0.0})
            det=estimate_curvature_identifiability(sample.x,q_grid=tuple(q.tolist()),estimators=(a.estimator_name,))[0].zeta_est
            rows.append({"process_type":proc,"level":"deterministic","lambda2_true":lam,**_cal(cal,det,q),"zeta_mae":float(np.mean(np.abs(det-z_analytic)))})
            if neural is not None:
                with torch.no_grad(): nz=neural(torch.tensor(sample.x[None,:],dtype=torch.float32))["zeta_emp"].numpy().reshape(-1)
                rows.append({"process_type":proc,"level":"neural","lambda2_true":lam,**_cal(cal,nz,q),"zeta_mae":float(np.mean(np.abs(nz-z_analytic)))})
    df=pd.DataFrame(rows); df.to_csv(td/"failure_attribution.csv",index=False); summ=df.groupby(["process_type","level"]).mean(numeric_only=True).reset_index(); summ.to_csv(td/"failure_attribution_summary_table.csv",index=False)
    fig,ax=plt.subplots(figsize=(8,4.8),constrained_layout=True); pivot=summ.pivot_table(index="process_type",columns="level",values="p_mrw"); pivot.plot(kind="bar",ax=ax); ax.set_ylabel("p_MRW"); ax.tick_params(axis="x",rotation=25); fig.savefig(fd/"pmrw_by_level.png",dpi=220); fig.savefig(fd/"pmrw_by_level.pdf"); plt.close(fig)
    report=rd/"failure_attribution_summary.md"; report.write_text("# CMIN-SR Failure Attribution\n\n"+summ.to_csv(index=False),encoding="utf-8")
    meta={"table":str((td/"failure_attribution.csv").relative_to(ROOT)),"report":str(report.relative_to(ROOT))}; (rd/"failure_attribution_summary.json").write_text(json.dumps(meta,indent=2),encoding="utf-8"); print(json.dumps(meta,indent=2))
if __name__=="__main__": main()
