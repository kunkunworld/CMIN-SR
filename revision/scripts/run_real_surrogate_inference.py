"""Existing real-data proxy, verified batched implementation, randomization tests."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import argparse
import time
import numpy as np
import pandas as pd
from revision.analysis_tools import Q,SCALES,dump_json,hashfile
from mrw_inverse.proxy import proxy_inverse_estimate


def batched_proxy(x):
    x=np.asarray(x,dtype=float)
    cs=np.c_[np.zeros(len(x)),np.cumsum(x,axis=1)]
    surface=[]
    for s in SCALES:
        a=np.abs(cs[:,s:]-cs[:,:-s])+1e-30
        surface.append(np.log(np.mean(a[:,:,None]**Q,axis=1)+1e-30))
    u=np.log(SCALES);u-=u.mean()
    z=np.einsum('s,bsq->bq',u,np.stack(surface,axis=1))/(u@u)
    X=np.c_[Q,-.5*Q*(Q-2)]
    b=z@np.linalg.pinv(X).T
    lc=np.maximum(b[:,1],0)
    zfit=b[:,0,None]*Q-.5*lc[:,None]*Q*(Q-2)
    ss=((z-z.mean(axis=1,keepdims=True))**2).sum(axis=1)
    r2=np.where(ss<=1e-12,1.,1.-((z-zfit)**2).sum(axis=1)/np.maximum(ss,1e-30))
    logabs=np.log(np.abs(x)+1e-6);logabs-=logabs.mean(axis=1,keepdims=True)
    lags=[1,2,4,8,16,32,64]
    covs=np.array([np.mean(logabs[:,:-k]*logabs[:,k:],axis=1) for k in lags]).T
    u=np.log(lags);u-=u.mean()
    lv=-(covs@u)/(u@u)
    lam=np.clip(.5*lc+.5*np.maximum(lv,0),0,.2)
    h=np.clip(.5*b[:,0]+.5*z[:,3]/2,1e-4,1-1e-4)
    final=h[:,None]*Q-.5*lam[:,None]*Q*(Q-2)
    residual=np.abs(z-final).mean(axis=1)
    score=1/(1+np.exp(-(3.5*(r2-.75)+6*lam+2*np.maximum(lv,0)-4*residual)))
    alpha=np.gradient(final,Q,axis=1)
    width=np.ptp(alpha,axis=1)
    return np.c_[lam,score,width]


def holm(p):
    p=np.asarray(p);idx=np.argsort(p);adj=np.empty(len(p))
    adj[idx]=np.minimum(1,np.maximum.accumulate(p[idx]*(len(p)-np.arange(len(p)))))
    return adj


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--replicates',type=int,default=999)
    ap.add_argument('--seed',type=int,default=61001)
    ap.add_argument('--blocks',type=int,nargs='+',default=[1,16])
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--input-csv',type=Path,default=ROOT/'data/real_processed/all_processed_returns.csv')
    args=ap.parse_args()
    if args.quick:args.replicates=19
    if not args.blocks or len(set(args.blocks))!=len(args.blocks) or any(b<1 or 512%b for b in args.blocks):
        raise ValueError('Unique positive block lengths must divide 512')
    inp=args.input_csv
    if not inp.is_file():
        raise FileNotFoundError('Supply --input-csv with locally prepared decimal factor returns; data are not bundled')
    source=pd.read_csv(inp,parse_dates=['date'])
    required={'date','series_name','return','source_name'}
    if not required.issubset(source.columns):
        raise ValueError(f'Missing columns: {sorted(required-set(source.columns))}')
    source=source[source.series_name.isin(['Mkt-RF','SMB','HML','Mom'])]
    if set(source.series_name)!=set(['Mkt-RF','SMB','HML','Mom']):
        raise ValueError('All four factors are required for the declared multiplicity family')
    if source.groupby('series_name').size().min()<512:
        raise ValueError('Each factor needs at least 512 observations')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    B=args.replicates;rows=[];prov=[];originals=[];repmeans=[];t0=time.time()
    dump_json(out/'manifest.json',dict(replicates=B,seed=args.seed,input_sha256=hashfile(inp),
        window=512,step=512,blocks=args.blocks,endpoints=['proxy_lambda2','proxy_p_MRW'],
        nulls=[f'exchangeability of {b}-day blocks within each window' for b in args.blocks],
        pvalue='min(1,2*min((1+count(sur>=obs))/(B+1),(1+count(sur<=obs))/(B+1)))',
        multiplicity=f'Holm over 4 factors x {len(args.blocks)} nulls x 2 endpoints; each run is a separate declared family',
        caveat='Conditional surrogate test, not a test proving multifractality or MRW; stationarity/exchangeability may fail'))
    # Independently check numerical identity against frozen scalar implementation.
    test=np.random.default_rng(90).normal(0,.01,(8,512))
    ref=np.array([[r.pred_lambda2,r.p_MRW,r.f_alpha_width] for r in map(proxy_inverse_estimate,test)])
    err=float(np.max(np.abs(ref-batched_proxy(test))))
    if err>1e-10:raise RuntimeError(f'Vectorization differs from legacy: {err}')
    for fi,(factor,d) in enumerate(source.groupby('series_name')):
        d=d.sort_values('date')
        if d.date.duplicated().any() or not np.isfinite(d['return']).all():raise ValueError('Invalid input dates/returns')
        n=len(d)//512;x=d['return'].to_numpy()[:n*512].reshape(n,512)
        obs=batched_proxy(x)
        for w in range(n):
            originals.append(dict(factor=factor,window_id=w,start=str(d.date.iloc[w*512].date()),
                end=str(d.date.iloc[(w+1)*512-1].date()),proxy_lambda2=obs[w,0],proxy_p_MRW=obs[w,1],width=obs[w,2]))
        prov.append(dict(factor=factor,source_name=d.source_name.iloc[0],start=d.date.min(),end=d.date.max(),
                         n_observations=len(d),n_windows=n,n_used=n*512,n_discarded=len(d)-n*512,
                         units='decimal returns; local preprocessing percent/100'))
        for block in args.blocks:
            rng=np.random.default_rng(np.random.SeedSequence([args.seed,fi,block]))
            allvals=np.empty((B,n,3))
            for b in range(B):
                ix=np.argsort(rng.random((n,512//block)),axis=1)
                y=np.take_along_axis(x.reshape(n,512//block,block),ix[:,:,None],axis=1).reshape(n,512)
                allvals[b]=batched_proxy(y)
                if (b+1)%200==0:print(f'{factor} block={block} {b+1}/{B} seconds={time.time()-t0:.1f}',flush=True)
            np.savez_compressed(out/f'{factor}_block{block}_surrogates.npz',metrics=allvals,
                                metric_names=['proxy_lambda2','proxy_p_MRW','width'])
            means=allvals.mean(axis=1);oval=obs.mean(axis=0)
            for b in range(B):repmeans.append(dict(factor=factor,block_size=block,replicate=b,
                 proxy_lambda2=means[b,0],proxy_p_MRW=means[b,1],width=means[b,2]))
            for j,name in enumerate(['proxy_lambda2','proxy_p_MRW']):
                upper=(1+np.sum(means[:,j]>=oval[j]))/(B+1)
                lower=(1+np.sum(means[:,j]<=oval[j]))/(B+1)
                gap=obs[:,j]-allvals[:,:,j].mean(axis=0)
                # Descriptive circular-block CI across consecutive nonoverlap windows.
                boot=np.random.default_rng(np.random.SeedSequence([args.seed,fi,block,j,8]))
                starts=boot.integers(0,n,(2000,int(np.ceil(n/4))))
                bix=((starts[:,:,None]+np.arange(4))%n).reshape(2000,-1)[:,:n]
                ci=np.quantile(gap[bix].mean(axis=1),[.025,.975])
                rows.append(dict(factor=factor,block_size=block,metric=name,n_windows=n,B=B,
                    original=oval[j],surrogate_mean=means[:,j].mean(),gap=oval[j]-means[:,j].mean(),
                    gap_ci_low=ci[0],gap_ci_high=ci[1],surrogate_q025=np.quantile(means[:,j],.025),
                    surrogate_q975=np.quantile(means[:,j],.975),p_two_sided=min(1.,2*min(upper,lower))))
            print(f'{factor} block={block} finished',flush=True)
            pd.DataFrame(rows).to_csv(out/'surrogate_tests_partial.csv',index=False)
    df=pd.DataFrame(rows);df['p_holm']=holm(df.p_two_sided)
    df.to_csv(out/'surrogate_tests.csv',index=False)
    pd.DataFrame(prov).to_csv(out/'data_provenance.csv',index=False)
    pd.DataFrame(originals).to_csv(out/'original_windows.csv',index=False)
    pd.DataFrame(repmeans).to_csv(out/'surrogate_replicate_means.csv',index=False)
    dump_json(out/'status.json',dict(status='PASS',scalar_equivalence_max_error=err,
               tests=len(df),replicates=B,input_unchanged=hashfile(inp)==__import__('json').loads((out/'manifest.json').read_text())['input_sha256'],
               seconds=time.time()-t0))
    print(df.to_string(index=False),flush=True)

if __name__=='__main__':main()
