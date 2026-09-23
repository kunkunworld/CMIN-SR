"""Paired normalization/input factorial plus corrected canonical MRW recovery.

No training, no historical overwrite. H=.5 is the confirmatory target.
Other H are stored as empirical modulation stress tests, not analytic truth.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import argparse
from datetime import datetime
import time
import numpy as np
import pandas as pd
import torch
from revision.generation import generate_pair,standardize,sample_gaussian
from revision.analysis_tools import (Q,SCALES,structure_zeta,projection_metrics,FrozenCalibrator,
    dump_json,protected_hashes)
from mrw_dl.generation import build_fgn_covariance
from mrw_inverse.analysis.classical_multifractal_estimators import estimate_classical_zeta
from revision.estimators.structure_input_contract import estimate_revision_robust_zeta


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--num-samples',type=int,default=20)
    ap.add_argument('--T-values',type=int,nargs='+',default=[512,1024,2048,4096,8192])
    ap.add_argument('--seeds',type=int,nargs='+',default=[31001,31002,31003,31004,31005])
    ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--skip-classical',action='store_true')
    args=ap.parse_args()
    if args.quick: args.num_samples=2;args.seeds=[31999];args.T_values=[512,1024]
    torch.set_num_threads(1)
    out=args.output_dir or ROOT/'revision/results'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_corrected_recovery')
    out.mkdir(parents=True,exist_ok=False)
    before=protected_hashes();dump_json(out/'protected_before.json',before)
    cal=FrozenCalibrator()
    levels=[0.,.005,.01,.03,.06,.10,.15,.20]
    manifest=dict(**vars(args),lambda2_levels=levels,H_confirmatory=.5,H_stress=[.2,.4,.6,.8],
        q_grid=Q.tolist(),scales=SCALES,integral_scale='min(512,max(64,T//4))',
        canonical_mean='legacy -Var/2 retained; after standardization identical to -Var',
        target='q/2-lambda2*q*(q-2)/2 asymptotic reference, finite discretization not exact',
        inference='frozen models only; old checkpoints evaluated on corrected data are distribution-shift tests',
        calibrator=cal.manifest,bootstrap_replicates_planned=2000,
        selection='No result-dependent selection of seeds, T, lambda or sample count')
    dump_json(out/'manifest.json',manifest)
    rows=[];t0=time.time();count=0
    def record(x,meta,method,z,target):
        d={**meta,'method':method,**projection_metrics(z,target,
            h=meta['H'] if target is not None else None,
            lam=meta['lambda2_true'] if target is not None else None)}
        d.update(cal.scores(z));rows.append(d)
    for T in args.T_values:
        raw=[];metadata=[]
        for es in args.seeds:
            for li,lam in enumerate(levels):
                for i in range(args.num_samples):
                    sd=int(np.random.SeedSequence([es,T,li,i,0]).generate_state(1)[0])
                    pair=generate_pair(T,.5,lam,sd)
                    sid=f'{es}_{T}_MRW_{li}_{i}'
                    base=dict(sample_id=sid,seed=es,data_seed=sd,T=T,process='MRW',H=.5,
                              lambda2_true=lam,L=pair['integral_scale'],target_valid=True)
                    target=Q*.5-.5*lam*Q*(Q-2)
                    for gen in ['legacy','corrected']:
                        x=standardize(pair[gen]);meta={**base,'generator':gen}
                        raw.append(x.astype(np.float32));metadata.append(meta)
                        for oldinput in [False,True]:
                            z=structure_zeta(x,legacy_input=oldinput)
                            record(x,meta,'structure_ols_wrong_input' if oldinput else 'structure_ols',z,target)
                        if gen=='corrected':
                            z=estimate_revision_robust_zeta(torch.tensor(x),input_kind='increments').zeta_robust.numpy()[0]
                            record(x,meta,'structure_trimmed',z,target)
                            if T<=2048 and not args.skip_classical:
                                for method in ['mfdfa','mfdfa_quadratic','wavelet_leader_haar','wtmm_haar']:
                                    z=estimate_classical_zeta(x,tuple(Q),SCALES,method).zeta
                                    record(x,meta,method,z,target)
                            for sn,sc in [('dense',(2,3,4,6,8,12,16,24,32,40,48,64)),
                                          ('wide',(2,4,8,16,32,64,128))]:
                                valid=tuple(s for s in sc if s<T//4)
                                record(x,meta,'structure_ols_'+sn,structure_zeta(x,scales=valid),target)
                    count+=1
            # Independent controls, including same-H fGn and separate Gaussian draws.
            for hi,h in enumerate([.2,.4,.5,.6,.8]):
                for i in range(args.num_samples*2):
                    sd=int(np.random.SeedSequence([es,T,hi,i,1]).generate_state(1)[0])
                    x,_=sample_gaussian(build_fgn_covariance(T,h),np.random.default_rng(sd));x=standardize(x)
                    meta=dict(sample_id=f'{es}_{T}_fGn_{hi}_{i}',seed=es,data_seed=sd,T=T,
                              process='fGn',H=h,lambda2_true=0.,L=0,target_valid=True,generator='corrected')
                    raw.append(x.astype(np.float32));metadata.append(meta)
                    record(x,meta,'structure_ols',structure_zeta(x),Q*h)
            for i in range(args.num_samples*2):
                sd=int(np.random.SeedSequence([es,T,i,2]).generate_state(1)[0])
                x=standardize(np.random.default_rng(sd).normal(size=T))
                meta=dict(sample_id=f'{es}_{T}_Gaussian_{i}',seed=es,data_seed=sd,T=T,
                          process='Gaussian',H=.5,lambda2_true=0.,L=0,target_valid=True,generator='corrected')
                raw.append(x.astype(np.float32));metadata.append(meta)
                record(x,meta,'structure_ols',structure_zeta(x),Q*.5)
            # Minimal all-H modulation check: no asserted theoretical zeta/parameter error.
            for hi,h in enumerate([.2,.4,.6,.8]):
                for li,lam in enumerate([0.,.03,.1,.2]):
                    for i in range(max(2,args.num_samples//4)):
                        sd=int(np.random.SeedSequence([es,T,hi,li,i,3]).generate_state(1)[0])
                        pair=generate_pair(T,h,lam,sd);x=standardize(pair['corrected'])
                        meta=dict(sample_id=f'{es}_{T}_Hstress_{hi}_{li}_{i}',seed=es,data_seed=sd,T=T,
                                  process='fGn_modulated_stress',H=h,lambda2_true=lam,L=pair['integral_scale'],
                                  target_valid=False,generator='corrected')
                        record(x,meta,'structure_ols',structure_zeta(x),None)
            pd.DataFrame(rows).to_csv(out/'sample_results.csv',index=False)
            print(f'T={T} seed={es} MRW pairs={count} rows={len(rows)} seconds={time.time()-t0:.1f}',flush=True)
        np.savez_compressed(out/f'raw_T{T}.npz',x=np.stack(raw))
        pd.DataFrame(metadata).to_csv(out/f'raw_T{T}_manifest.csv',index=False)
    after=protected_hashes();changed=[p for p in before if before[p]!=after.get(p)]
    dump_json(out/'status.json',dict(status='PASS' if not changed else 'FAIL_PROTECTED_FILES',
              rows=len(rows),MRW_pairs=count,protected_count=len(before),changed=changed,seconds=time.time()-t0))
    print(out,flush=True)
    if changed: raise SystemExit(2)

if __name__=='__main__':main()
