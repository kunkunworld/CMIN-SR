"""Signed-coordinate audit and common-sample tables; reuse frozen R1 predictions."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from common import *
import argparse
import time
import torch
from scipy.stats import pearsonr
from revision.analysis_tools import projection_metrics, structure_zeta
from revision.retraining import make_model
from revision.estimators.structure_input_contract import estimate_revision_robust_zeta
from mrw_inverse.analysis.classical_multifractal_estimators import estimate_classical_zeta

SOURCE = ROOT / 'revision/results/neural_eval_formal'
TRAIN = ROOT / 'revision/results/neural_retrain_formal_attempt3'
METHODS = ['structure_ols','structure_trimmed','mfdfa','mfdfa_quadratic',
           'wavelet_leader_haar','wtmm_haar','ridge','ridge_logvol','full_c025']
AUDIT_METHODS = METHODS + ['nonoverlap_prior','zeta_only_c025']
ZCOL = [f'zeta_{i}' for i in range(6)]


def check_replay(df, x, metadata, T):
    rows=[]; runs=pd.read_csv(TRAIN/'training_runs.csv')
    # Explicit sample IDs, not row position across differently ordered tables.
    take=metadata.groupby(['process','H','lambda2_true'],sort=True).head(1).index.to_numpy()
    for method,ts in [('structure_ols',-1)]+[('full_c025',s) for s in [51001,51002,51003]]:
        g=df[(df.method==method)&(df.train_seed==ts)].set_index('sample_id')
        if method=='structure_ols':
            z=np.stack([structure_zeta(y.astype(float)) for y in x[take]])
        else:
            r=runs[(runs.config==method)&(runs.seed==ts)].iloc[0]
            cp=TRAIN/r.checkpoint
            assert sha(cp)==r.checkpoint_sha256
            model=make_model(ts,.25,'cuda')
            state=torch.load(cp,map_location='cpu')
            model.load_state_dict(state['model_state_dict'],strict=True);model.eval()
            # Match the original device and 64-row batch boundaries for replay.
            with torch.no_grad():
                zs=[model(torch.tensor(x[i:i+64],device='cuda'))['zeta_emp'].cpu().numpy()
                    for i in range(0,len(x),64)]
            z=np.concatenate(zs)[take]
            del model
        target=g.loc[metadata.iloc[take].sample_id,ZCOL].to_numpy()
        error=float(np.max(np.abs(z-target)))
        tolerance=2e-5 if method=='full_c025' else 1e-10
        assert error<tolerance, (method,T,error)
        rows.append(dict(T=T,method=method,train_seed=ts,samples=len(take),max_abs_error=error,tolerance=tolerance))
    return rows


def boot_metrics(g, ix):
    y=g.lambda2_true.to_numpy();s=g.lambda2_signed.to_numpy();c=g.lambda2_proj.to_numpy()
    arrays={'spectrum_mae':g.zeta_mae.to_numpy(),'signed_bias':s-y,'signed_mae':np.abs(s-y),
            'clipped_bias':c-y,'clipped_mae':np.abs(c-y),'negative_fraction':(s<0).astype(float),
            'clipping_mae_change':np.abs(c-y)-np.abs(s-y)}
    p={k:v.mean() for k,v in arrays.items()};b={k:v[ix].mean(axis=1) for k,v in arrays.items()}
    for key,v in [('signed_corr',s),('clipped_corr',c)]:
        p[key]=float(corr(y,v));b[key]=corr(y[ix],v[ix])
        if np.std(y)>1e-14 and np.std(v)>1e-14:
            assert abs(p[key]-pearsonr(y,v).statistic)<1e-12
    return p,b


def summaries(df, T, out):
    intervals=[];perseed=[];paired=[];qrows=[]
    for group in ['MRW_all','MRW_positive','MRW_zero','fGn_H0.5','Gaussian']:
        if group=='MRW_all':d=df[df.process=='MRW']
        elif group=='MRW_positive':d=df[(df.process=='MRW')&(df.lambda2_true>0)]
        elif group=='MRW_zero':d=df[(df.process=='MRW')&(df.lambda2_true==0)]
        elif group=='Gaussian':d=df[df.process=='Gaussian']
        else:d=df[(df.process=='fGn')&(df.H==.5)]
        ref=d[d.method=='structure_ols'].sort_values('sample_id').reset_index(drop=True)
        ix=indices(ref);boots={};points={};seedsets={}
        for method,gm in d.groupby('method',sort=True):
            seeds=sorted(gm.train_seed.unique());seedsets[method]=seeds
            pb=[];bb=[]
            for seed in seeds:
                gs=gm[gm.train_seed==seed].set_index('sample_id').loc[ref.sample_id].reset_index()
                assert len(gs)==len(ref) and not gs.sample_id.duplicated().any()
                p,b=boot_metrics(gs,ix);pb.append(p);bb.append(b)
                perseed.append(dict(T=T,group=group,method=method,train_seed=seed,N=len(gs),**p))
                if group in ['fGn_H0.5','MRW_zero']:
                    w=np.linalg.pinv(np.c_[Q,-.5*Q*(Q-2)])[1]
                    target=gs.H.to_numpy()[:,None]*Q
                    error=(gs[ZCOL].to_numpy()-target).mean(axis=0)
                    assert abs(error@w-p['signed_bias'])<1e-9
                    for j,q in enumerate(Q):
                        qrows.append(dict(T=T,group=group,method=method,train_seed=seed,q=q,
                            mean_zeta_error=error[j],projection_weight=w[j],bias_contribution=error[j]*w[j],
                            signed_bias=p['signed_bias']))
            points[method]={k:np.array([p[k] for p in pb]) for k in pb[0]}
            boots[method]={k:np.stack([b[k] for b in bb]) for k in bb[0]}
            sel=np.random.default_rng(202610077).integers(len(seeds),size=(B,len(seeds)))
            for metric,values in boots[method].items():
                boot=values[sel,np.arange(B)[:,None]].mean(axis=1)
                point=points[method][metric].mean();lo,hi=ci(boot)
                intervals.append(dict(T=T,group=group,method=method,metric=metric,mean=point,
                    ci_low=lo,ci_high=hi,training_seed_sd=np.std(points[method][metric],ddof=1) if len(seeds)>1 else np.nan,
                    training_repeats=len(seeds),independent_N=len(ref),
                    note='Undefined correlation when true curvature is constant' if not np.isfinite(point) else '',
                    interval_scope='stratified test resampling; learned models additionally resample three training/data seeds'))
        if 'full_c025' in boots:
            for baseline in METHODS:
                if baseline not in boots or baseline=='full_c025':continue
                a=boots['full_c025'];b=boots[baseline]
                if len(seedsets[baseline])>1: assert seedsets[baseline]==seedsets['full_c025']
                sel=np.random.default_rng(202610077).integers(3,size=(B,3))
                for metric in ['spectrum_mae','signed_mae','clipped_mae','signed_corr','clipped_corr']:
                    delta=a[metric]-b[metric]
                    vals=delta[sel,np.arange(B)[:,None]].mean(axis=1)
                    lo,hi=ci(vals)
                    paired.append(dict(T=T,group=group,method='full_c025',baseline=baseline,metric=metric,
                        difference=points['full_c025'][metric].mean()-points[baseline][metric].mean(),ci_low=lo,ci_high=hi))
    return intervals,perseed,paired,qrows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--quick',action='store_true')
    args=ap.parse_args();out=args.output
    torch.set_num_threads(1)
    before=start(out,dict(experiment='E2 and E3',quick=args.quick,base=str(SOURCE),
       primary='full_c025',source_status=json.loads((SOURCE/'status.json').read_text()),
       note='Classical missing outputs recomputed on the exact stored neural holdout; no re-training.'))
    t0=time.time();allstats=[];allseed=[];allpaired=[];allq=[];replay=[];sources={}
    try:
        assert json.loads((SOURCE/'status.json').read_text())['status']=='PASS'
        for T in ([512] if args.quick else [512,1024,2048]):
            paths=[SOURCE/f'results_test_T{T}.csv',SOURCE/f'test_T{T}.npz',SOURCE/f'test_T{T}_manifest.csv']
            sources.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
            df=pd.read_csv(paths[0]);x=np.load(paths[1])['x'];meta=pd.read_csv(paths[2])
            assert len(meta)==len(x) and not meta.sample_id.duplicated().any()
            assert df.groupby(['method','train_seed']).sample_id.nunique().eq(len(meta)).all()
            replay+=check_replay(df,x,meta,T)
            if args.quick:
                keep=meta.groupby(['seed','process','H','lambda2_true']).head(1).index
                meta=meta.loc[keep];x=x[keep];df=df[df.sample_id.isin(meta.sample_id)]
            df=df[df.method.isin(AUDIT_METHODS)].copy()
            # Recompute every signed projection from archived spectra, independently
            # of the stored scalar columns before using them in the new summaries.
            signed=df[ZCOL].to_numpy() @ np.linalg.pinv(np.c_[Q,-.5*Q*(Q-2)]).T
            assert np.max(np.abs(signed[:,1]-df.lambda2_signed))<1e-9
            assert np.max(np.abs(np.maximum(signed[:,1],1e-6)-df.lambda2_proj))<1e-9
            missing=[]
            for i,(series,info) in enumerate(zip(x,meta.to_dict('records'))):
                if info['process']!='MRW':continue
                target=Q*info['H']-.5*info['lambda2_true']*Q*(Q-2)
                z=estimate_revision_robust_zeta(torch.tensor(series),input_kind='increments').zeta_robust.numpy()[0]
                candidates=[('structure_trimmed',z)]
                for method in ['wavelet_leader_haar','wtmm_haar']:
                    candidates.append((method,estimate_classical_zeta(series.astype(float),tuple(Q),SCALES,method).zeta))
                for method,z in candidates:
                    missing.append(dict(**info,method=method,train_seed=-1,
                        **projection_metrics(z,target,info['H'],info['lambda2_true'])))
            extra=pd.DataFrame(missing);extra.to_csv(out/f'added_classical_T{T}.csv',index=False)
            df=pd.concat([df,extra],ignore_index=True)
            df.to_csv(out/f'common_sample_predictions_T{T}.csv',index=False)
            stats,perseed,paired,qrows=summaries(df,T,out)
            allstats+=stats;allseed+=perseed;allpaired+=paired;allq+=qrows
            print('signed/common table',T,'MRW N',int((meta.process=='MRW').sum()),'elapsed',round(time.time()-t0),flush=True)
        d=pd.DataFrame(allstats)
        d.to_csv(out/'all_metrics_intervals.csv',index=False)
        pd.DataFrame(allseed).to_csv(out/'metrics_by_training_seed.csv',index=False)
        pd.DataFrame(allpaired).to_csv(out/'paired_method_differences.csv',index=False)
        pd.DataFrame(allq).to_csv(out/'boundary_bias_q_decomposition.csv',index=False)
        pd.DataFrame(replay).to_csv(out/'numerical_replay_checks.csv',index=False)
        primary=d[(d.group=='MRW_all')&d.method.isin(METHODS)]
        table=primary.pivot(index=['T','method','independent_N','training_repeats'],columns='metric',values=['mean','ci_low','ci_high'])
        table.columns=[f'{b}_{a}' for a,b in table.columns]
        table.reset_index().to_csv(out/'common_estimator_table.csv',index=False)
        d[d.method.isin(['structure_ols','ridge','ridge_logvol','full_c025'])&d.metric.isin(
            ['signed_bias','signed_mae','signed_corr','clipped_mae','clipped_corr','negative_fraction','clipping_mae_change'])].to_csv(out/'signed_curvature_companion.csv',index=False)
        dump(out/'input_hashes.json',sources)
        for name,h in sources.items(): assert sha(ROOT/name)==h
        finish(out,before,seconds=time.time()-t0,input_files_unchanged=True,neural_replay_verified=True,
               independent_sample_counts='MRW 800 per T; learned rows have 3 predictions per same sample, not N=2400',
               primary_scope='MRW H=.5; Haar controls explicitly approximate')
    except Exception:
        import traceback
        dump(out/'status.json',dict(status='FAILED',traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
