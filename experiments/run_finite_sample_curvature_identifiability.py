from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/"src"
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from mrw_inverse.analysis.curvature_identifiability import ESTIMATORS, estimate_curvature_identifiability
from mrw_inverse.data.process_generators import generate_process_sample

REPORT_DIR=ROOT/"outputs"/"reports"/"finite_sample_identifiability"
TABLE_DIR=ROOT/"outputs"/"tables"/"finite_sample_identifiability"
FIG_DIR=ROOT/"outputs"/"figures"/"finite_sample_identifiability"

def _parse_floats(s): return tuple(float(x) for x in s.split(",") if x)
def _parse_ints(s): return tuple(int(x) for x in s.split(",") if x)
def _spearman(a,b): return pd.Series(a).rank().corr(pd.Series(b).rank()) if len(a)>2 else np.nan
def _display_path(path: Path) -> str:
    p = path.resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)

def run_grid(Hs,lams,Ts,n,q_grid,scales,estimators,seed):
    rng=np.random.default_rng(seed); rows=[]
    for T in Ts:
        valid_scales=tuple(s for s in scales if s<T//4)
        for H in Hs:
            for lam in lams:
                for sample_idx in range(n):
                    proc="Low-lambda2 MRW" if lam<0.03 else "MRW"
                    sample=generate_process_sample(proc,T,rng,h=float(H),lambda2=float(lam),q_grid=np.asarray(q_grid,dtype=np.float32))
                    for est in estimate_curvature_identifiability(sample.x,q_grid=q_grid,scales=valid_scales,estimators=estimators):
                        ztrue=(np.asarray(q_grid)*H-0.5*lam*np.asarray(q_grid)*(np.asarray(q_grid)-2.0)).astype(float)
                        rows.append({"T":T,"H_true":H,"lambda2_true":lam,"sample_idx":sample_idx,"estimator":est.estimator,"lambda2_proj":est.lambda2_proj,"H_proj":est.H_proj,"zeta_mae":float(np.mean(np.abs(est.zeta_est-ztrue))),"second_diff_norm":est.second_diff_norm,"curvature_score":est.curvature_score,"mrw_residual_norm":est.mrw_residual_norm,"mono_residual_norm":est.mono_residual_norm,"mrw_vs_mono_gain":est.mrw_vs_mono_gain,"scaling_fit_quality":est.scaling_fit_quality,"high_q_instability":est.high_q_instability,"warning_flags":est.warning_flags})
    return pd.DataFrame(rows)

def summarize(df):
    rows=[]
    for keys,g in df.groupby(["T","estimator"]):
        err=g["lambda2_proj"]-g["lambda2_true"]
        rows.append({"T":keys[0],"estimator":keys[1],"lambda2_mae":float(err.abs().mean()),"lambda2_rmse":float(np.sqrt((err**2).mean())),"lambda2_corr":float(g["lambda2_true"].corr(g["lambda2_proj"])),"lambda2_spearman":float(_spearman(g["lambda2_true"],g["lambda2_proj"])),"high_lambda_detection_rate":float((g[g["lambda2_true"]>=0.06]["lambda2_proj"]>=0.04).mean()) if (g["lambda2_true"]>=0.06).any() else np.nan,"boundary_accuracy":float(((g["lambda2_true"]<0.03)==(g["lambda2_proj"]<0.04)).mean()),"mean_mrw_vs_mono_gain":float(g["mrw_vs_mono_gain"].mean()),"warning_rate":float((g["warning_flags"]>0).mean())})
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--quick",action="store_true"); p.add_argument("--num-samples",type=int,default=None); p.add_argument("--T-values",default=None); p.add_argument("--qgrid-name",default="Q1"); p.add_argument("--estimator-name",default="all"); p.add_argument("--output-dir",default=None); p.add_argument("--seed",type=int,default=2026)
    args=p.parse_args()
    out_root=Path(args.output_dir) if args.output_dir else ROOT/"outputs"
    report_dir=out_root/"reports"/"finite_sample_identifiability"; table_dir=out_root/"tables"/"finite_sample_identifiability"; fig_dir=out_root/"figures"/"finite_sample_identifiability"
    report_dir.mkdir(parents=True,exist_ok=True); table_dir.mkdir(parents=True,exist_ok=True); fig_dir.mkdir(parents=True,exist_ok=True)
    Hs=[0.4,0.6] if args.quick else [0.2,0.4,0.6,0.8]
    lams=[0,0.03,0.10,0.20] if args.quick else [0,0.005,0.01,0.03,0.06,0.10,0.15,0.20]
    Ts=_parse_ints(args.T_values) if args.T_values else ([512,1024,2048] if args.quick else [512,1024,2048,4096])
    n=args.num_samples if args.num_samples is not None else (10 if args.quick else 20)
    q=(0.5,1,1.5,2,2.5,3)
    scales=(2,4,8,16,32,64)
    classical=("structure_aggregated_ols","mfdfa","mfdfa_quadratic","wavelet_leader_haar","wtmm_haar")
    if args.estimator_name=="all":
        estimators=ESTIMATORS
    elif args.estimator_name=="classical":
        estimators=classical
    elif args.estimator_name=="structure":
        estimators=("structure_ols","structure_trimmed","structure_bootstrap","structure_smoothed","structure_aggregated_ols")
    else:
        estimators=(args.estimator_name,)
    df=run_grid(Hs,lams,Ts,n,q,scales,estimators,args.seed)
    summary=summarize(df)
    df.to_csv(table_dir/"sample_level_results.csv",index=False); summary.to_csv(table_dir/"lambda2_recovery_by_T.csv",index=False); summary.groupby("estimator").mean(numeric_only=True).reset_index().to_csv(table_dir/"lambda2_recovery_by_estimator.csv",index=False)
    fig,ax=plt.subplots(figsize=(7,4.8),constrained_layout=True)
    for est,g in summary.groupby("estimator"): ax.plot(g["T"],g["lambda2_corr"],marker="o",label=est)
    ax.set_xscale("log",base=2); ax.set_xlabel("T"); ax.set_ylabel("corr(lambda2_true, lambda2_proj)"); ax.legend(fontsize=8); fig.savefig(fig_dir/"lambda2_corr_vs_T.png",dpi=220); fig.savefig(fig_dir/"lambda2_corr_vs_T.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.8),constrained_layout=True)
    for est,g in summary.groupby("estimator"): ax.plot(g["T"],g["lambda2_mae"],marker="o",label=est)
    ax.set_xscale("log",base=2); ax.set_xlabel("T"); ax.set_ylabel("lambda2 MAE"); ax.legend(fontsize=8); fig.savefig(fig_dir/"lambda2_mae_vs_T.png",dpi=220); fig.savefig(fig_dir/"lambda2_mae_vs_T.pdf"); plt.close(fig)
    report=report_dir/"finite_sample_identifiability_summary.md"; report.write_text("\n".join(["# Finite-Sample Curvature Identifiability","",summary.to_csv(index=False)]),encoding="utf-8")
    meta={"sample_level_results":_display_path(table_dir/"sample_level_results.csv"),"lambda2_recovery_by_T":_display_path(table_dir/"lambda2_recovery_by_T.csv"),"report":_display_path(report)}
    (report_dir/"finite_sample_identifiability_summary.json").write_text(json.dumps(meta,indent=2),encoding="utf-8"); print(json.dumps(meta,indent=2))
if __name__=="__main__": main()
