"""Controlled spectral perturbations and corrected raw-error overlays, no retraining."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import argparse
import numpy as np
import pandas as pd
import torch
from revision.analysis_tools import Q,FrozenCalibrator,dump_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evaluation-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);cal=FrozenCalibrator()
    rows=[];levels=[0,.005,.01,.02,.05,.1];seeds=[94001,94002,94003]
    covariance=np.exp(-.5*((Q[:,None]-Q[None,:])/.75)**2);root=np.linalg.cholesky(covariance)
    for seed in seeds:
        rng=np.random.default_rng(seed)
        for kind in ['jagged','smooth_correlated','high_q_bending']:
            for i in range(100):
                if kind=='jagged':noise=rng.normal(size=6)
                elif kind=='smooth_correlated':noise=root@rng.normal(size=6)
                else:noise=rng.choice([-1.,1.])*np.maximum(Q-1.5,0)**2
                noise=noise/np.sqrt(np.mean(noise**2))
                for level in levels:
                    scores=[]
                    for family,lam in [('linear',0.),('boundary',.01),('curved',.1)]:
                        target=.5*Q-.5*lam*Q*(Q-2);p=cal.scores(target+level*noise);scores.append(p)
                        rows.append(dict(seed=seed,pair_id=i,perturbation=kind,noise_rmse=level,family=family,**p))
        print(f'noise bridge seed={seed} complete',flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'controlled_sample_results.csv',index=False)
    pairs=df.pivot(index=['seed','pair_id','perturbation','noise_rmse'],columns='family',values='p_mrw').reset_index()
    pairs['curved_minus_linear']=pairs.curved-pairs.linear;pairs.to_csv(out/'paired_gaps.csv',index=False)
    summaries=[]
    for (kind,level),d in pairs.groupby(['perturbation','noise_rmse']):
        d=d.reset_index(drop=True);rng=np.random.default_rng(94011)
        idx=np.concatenate([rng.choice(v,(2000,len(v))) for v in d.groupby('seed').indices.values()],axis=1)
        gap=d.curved_minus_linear.to_numpy();lo,hi=np.quantile(gap[idx].mean(axis=1),[.025,.975])
        summaries.append(dict(perturbation=kind,noise_rmse=level,gap=gap.mean(),ci_low=lo,ci_high=hi,N=len(d)))
    pd.DataFrame(summaries).to_csv(out/'gap_intervals.csv',index=False)
    raw=pd.read_csv(args.evaluation_dir/'results_test_T1024.csv')
    raw=raw[(raw.method.isin(['structure_ols','full_c025','ridge']))&
            (((raw.process=='MRW')&(raw.lambda2_true==.1))|((raw.process=='fGn')&(raw.H==.5)))]
    z=raw[[f'zeta_{i}' for i in range(6)]].to_numpy()
    target=raw.H.to_numpy()[:,None]*Q-.5*raw.lambda2_true.to_numpy()[:,None]*Q*(Q-2)
    raw=raw.copy();raw['zeta_rmse']=np.sqrt(np.mean((z-target)**2,axis=1))
    raw.to_csv(out/'raw_error_overlay_samples.csv',index=False)
    raw.groupby(['method','process','train_seed'])[['zeta_rmse','p_mrw','p_curved']].mean().to_csv(out/'raw_error_overlay.csv')
    dump_json(out/'status.json',dict(status='PASS',controlled_rows=len(df),
        scope='Equal RMSE does not imply equal error direction/distribution; overlay is descriptive, not causal attribution'))


if __name__=='__main__':main()
