"""Paired uncertainty for existing q-grid, scale-density and integral-scale controls."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import argparse
import numpy as np
import pandas as pd
from revision.scripts.summarize_corrected_experiments import indices,corr_rows,bounds
from revision.analysis_tools import dump_json,safe_corr


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    base=pd.read_csv(ROOT/'revision/results/corrected_recovery_formal/sample_results.csv')
    base=base[(base.generator=='corrected')&(base.process=='MRW')&(base.H==.5)]
    q=pd.read_csv(ROOT/'revision/results/scope_sensitivity/qgrid_sample_results.csv');q=q[q.process=='MRW']
    fixed=pd.read_csv(ROOT/'revision/results/scope_sensitivity/fixed_integral_scale_samples.csv')
    rows=[]
    for T,reference in base[base.method=='structure_ols'].groupby('T'):
        reference=reference.sort_values('sample_id').reset_index(drop=True)
        ix=indices(reference,2000,seed=93301);truth=reference.lambda2_true.to_numpy()
        pr=reference.lambda2_proj.to_numpy();rboot=corr_rows(truth[ix],pr[ix]);rmae=np.abs(pr-truth)[ix].mean(axis=1)
        candidates=[('qgrid',name,g) for name,g in q[q['T']==T].groupby('qgrid')]
        candidates += [('scales',name,g) for name,g in base[(base['T']==T)&base.method.isin(['structure_ols_dense','structure_ols_wide'])].groupby('method')]
        candidates += [('integral_scale','fixed_L512',fixed[fixed['T']==T])]
        for kind,name,g in candidates:
            if len(g)==0:continue
            if g.sample_id.duplicated().any() or set(g.sample_id)!=set(reference.sample_id):raise AssertionError('Unpaired design')
            g=g.set_index('sample_id').loc[reference.sample_id].reset_index()
            p=g.lambda2_proj.to_numpy();rb=corr_rows(truth[ix],p[ix]);mb=np.abs(p-truth)[ix].mean(axis=1)
            rl,rh=bounds(rb);ml,mh=bounds(mb);dl,dh=bounds(rb-rboot);el,eh=bounds(mb-rmae)
            rows.append(dict(T=T,study=kind,configuration=name,independent_N=len(g),
                lambda2_corr=safe_corr(truth,p),corr_ci_low=rl,corr_ci_high=rh,
                lambda2_mae=np.mean(np.abs(p-truth)),mae_ci_low=ml,mae_ci_high=mh,
                corr_delta_vs_default=safe_corr(truth,p)-safe_corr(truth,pr),corr_delta_ci_low=dl,corr_delta_ci_high=dh,
                mae_delta_vs_default=np.mean(np.abs(p-truth))-np.mean(np.abs(pr-truth)),mae_delta_ci_low=el,mae_delta_ci_high=eh))
        print(f'scope controls T={T} complete',flush=True)
    pd.DataFrame(rows).to_csv(out/'paired_scope_intervals.csv',index=False)
    dump_json(out/'status.json',dict(status='PASS',bootstrap_replicates=2000,
        note='Paired exploratory contrasts, no multiplicity-adjusted claim of best configuration'))


if __name__=='__main__':main()
