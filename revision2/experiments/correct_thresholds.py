"""Null-only threshold recalibration followed by sealed independent evaluation."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from common import *
import argparse
import math
import time
import torch
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from revision.retraining import simulate, make_model
from revision.analysis_tools import structure_zeta, projection_metrics, FrozenCalibrator
from revision.generation import generate_pair, standardize
from mrw_inverse.data.process_generators import gen_iid_student_t, gen_garch, gen_regime_switching_gaussian

PROTOCOL=HERE/'THRESHOLD_CORRECTION_PROTOCOL.json'
TRAIN=ROOT/'revision/results/neural_retrain_formal_attempt3'
OLD=ROOT/'revision/results/neural_summary_formal/heldout_roc_thresholds.csv'
SCORES=['p_mrw','lambda2_proj']


def excluded_seeds():
    files=[TRAIN/'data_manifest.csv']
    files+=list((ROOT/'revision/results/neural_eval_formal').glob('*_manifest.csv'))
    files+=list((ROOT/'revision/results/backend_ablation_eval_formal').glob('*_manifest.csv'))
    files+=list((HERE/'results/E4_formal').glob('*_manifest.csv'))
    values=set()
    for p in files:
        header=pd.read_csv(p,nrows=0).columns
        if 'data_seed' in header:
            values.update(int(x) for x in pd.read_csv(p,usecols=['data_seed'],dtype=str).data_seed.dropna())
    return values


def make_data(T,stage,protocol,quick,excluded,seen):
    cfg=protocol['calibration' if stage=='calibrate' else 'test']
    hs=cfg['H' if stage=='calibrate' else 'fGn_H']
    seeds=cfg['data_seeds'][:1] if quick else cfg['data_seeds']
    count=(24 if stage=='calibrate' else 3) if quick else cfg[
        'replicates_per_seed_H' if stage=='calibrate' else 'fGn_replicates_per_seed_H']
    specs=[('fGn',h,0.,count) for h in hs]
    if stage=='test':
        specs += [('MRW',.5,lam,2 if quick else cfg['MRW_replicates_per_seed_lambda2']) for lam in cfg['MRW_lambda2']]
        if T==cfg['stress_length']:
            specs += [(p,np.nan,np.nan,2 if quick else cfg['stress_replicates_per_seed_family']) for p in cfg['stress_families']]
    xs=[];rows=[]
    for seed in seeds:
        for group,(process,h,lam,n) in enumerate(specs):
            for i in range(n):
                sd=int(np.random.SeedSequence([2026100705,1 if stage=='calibrate' else 2,
                    int(quick),seed,T,group,i]).generate_state(1,dtype=np.uint64)[0])
                assert sd not in excluded and sd not in seen
                seen.add(sd);rng=np.random.default_rng(sd)
                if process in ['fGn','MRW']:x=simulate(T,process,h,lam,sd)
                elif process=='Student_t':x=gen_iid_student_t(T,rng)
                elif process=='GARCH':x=gen_garch(T,rng)
                elif process=='iid_Gaussian_mixture':x=gen_regime_switching_gaussian(T,rng)
                elif process=='shuffled_MRW':
                    x=standardize(generate_pair(T,.5,.1,sd)['corrected'])
                    np.random.default_rng(np.random.SeedSequence([sd,1])).shuffle(x)
                else:raise ValueError(process)
                xs.append(np.asarray(x,dtype=np.float32))
                rows.append(dict(sample_id=f'E5_{stage}_{seed}_{T}_{group}_{i}',data_seed=sd,seed=seed,
                                 split=stage,T=T,process=process,H=h,lambda2_true=lam))
    x=np.stack(xs);assert np.isfinite(x).all()
    return x,pd.DataFrame(rows)


def infer(x,meta,input_hashes):
    arrays=[('structure_ols',-1,np.stack([structure_zeta(s.astype(float)) for s in x]))]
    runs=pd.read_csv(TRAIN/'training_runs.csv')
    for seed in [51001,51002,51003]:
        r=runs[(runs.config=='full_c025')&(runs.seed==seed)].iloc[0];cp=TRAIN/r.checkpoint
        assert sha(cp)==r.checkpoint_sha256;input_hashes[str(cp.relative_to(ROOT))]=sha(cp)
        model=make_model(seed,.25,'cuda')
        model.load_state_dict(torch.load(cp,map_location='cpu')['model_state_dict'],strict=True);model.eval()
        with torch.no_grad():
            z=np.concatenate([model(torch.tensor(x[i:i+64],device='cuda'))['zeta_emp'].cpu().numpy()
                              for i in range(0,len(x),64)])
        arrays.append(('full_c025',seed,z));del model
    cal=FrozenCalibrator();input_hashes[cal.manifest['path']]=cal.manifest['sha256'];rows=[]
    for method,seed,zs in arrays:
        assert np.isfinite(zs).all()
        for info,z in zip(meta.to_dict('records'),zs):
            target=Q*info['H']-.5*info['lambda2_true']*Q*(Q-2) if info['process'] in ['fGn','MRW'] else None
            rows.append(dict(**info,method=method,train_seed=seed,
                **projection_metrics(z,target,info['H'],info['lambda2_true']),**cal.scores(z)))
    return pd.DataFrame(rows)


def rank_threshold(values,alpha):
    v=np.sort(np.asarray(values,dtype=float));n=len(v)
    rank=math.ceil((n+1)*(1-alpha))
    if rank>n:raise ValueError('Calibration set too small for requested tail level')
    return float(v[rank-1]),rank


def calibrate(df,alpha):
    groups=[];frozen=[]
    for (T,method,seed),g in df.groupby(['T','method','train_seed']):
        assert set(g.process)=={'fGn'}
        for score in SCORES:
            cuts=[]
            for h,d in g.groupby('H'):
                threshold,rank=rank_threshold(d[score],alpha);cuts.append(threshold)
                groups.append(dict(T=T,method=method,train_seed=seed,score=score,H=h,N=len(d),
                    order_statistic_rank=rank,threshold=threshold,validation_tail_rate=np.mean(d[score]>threshold)))
            threshold=max(cuts)
            frozen.append(dict(T=int(T),method=method,train_seed=int(seed),score=score,threshold=threshold,
                threshold_hex=threshold.hex(),comparison='strictly greater than',
                source='maximum of finite-sample-adjusted per-H null quantiles',alpha=alpha))
    return groups,frozen


def resample(g):
    groups=g.reset_index(drop=True).groupby(['seed','process','H','lambda2_true'],dropna=False).indices
    rng=np.random.default_rng(2026100717)
    return np.concatenate([rng.choice(v,size=(B,len(v)),replace=True) for v in groups.values()],axis=1)


def wilson(k,n):
    p=k/n;z=norm.ppf(.975);den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,center-half),min(1,center+half)


def summarize(df,frozen):
    old=pd.read_csv(OLD,float_precision='round_trip');per=[];combined=[];paired=[];aucrows=[]
    cuts={(r['T'],r['method'],r['train_seed'],r['score']):float.fromhex(r['threshold_hex']) for r in frozen}
    for (T,method),d in df.groupby(['T','method']):
        groups=[(f'fGn_H{h:g}',(d.process=='fGn')&(d.H==h)) for h in sorted(d.loc[d.process=='fGn','H'].unique())]
        groups += [(f'MRW_lambda2_{l:g}',(d.process=='MRW')&(d.lambda2_true==l)) for l in sorted(d.loc[d.process=='MRW','lambda2_true'].unique())]
        groups += [('MRW_significant',(d.process=='MRW')&(d.lambda2_true>=.06))]
        groups += [(p,d.process==p) for p in ['Student_t','GARCH','iid_Gaussian_mixture','shuffled_MRW'] if (d.process==p).any()]
        for name,mask in groups:
            dd=d[mask];seeds=sorted(dd.train_seed.unique())
            for score in SCORES:
                points={};boots={}
                for seed in seeds:
                    g=dd[dd.train_seed==seed].sort_values('sample_id').reset_index(drop=True);ix=resample(g)
                    r=old[(old['T']==T)&(old.method==method)&(old.train_seed==seed)&(old.score==score)].iloc[0]
                    oldcut=float(r.threshold);cut=cuts[T,method,seed,score]
                    val=g[score].to_numpy(dtype=float)
                    # Preserve the original >= rule exactly; the new rank rule is >.
                    dec={'original':val>=oldcut,'multiH':val>cut}
                    for rule,a in dec.items():
                        p=float(a.mean());boot=a[ix].mean(axis=1);lo,hi=ci(boot);wl,wh=wilson(int(a.sum()),len(a))
                        points[seed,rule]=p;boots[seed,rule]=boot
                        per.append(dict(T=T,method=method,train_seed=seed,group=name,score=score,rule=rule,
                            rate=p,positive_count=int(a.sum()),independent_N=len(a),ci_low=lo,ci_high=hi,
                            wilson_low=wl,wilson_high=wh,threshold=cut if rule=='multiH' else oldcut,
                            threshold_hex=(cut if rule=='multiH' else oldcut).hex(),comparison='>' if rule=='multiH' else '>='))
                selection=np.random.default_rng(2026100718).integers(len(seeds),size=(B,len(seeds)))
                for rule in ['original','multiH']:
                    a=np.stack([boots[s,rule] for s in seeds]);vals=a[selection,np.arange(B)[:,None]].mean(axis=1)
                    lo,hi=ci(vals)
                    combined.append(dict(T=T,method=method,group=name,score=score,rule=rule,
                        mean=np.mean([points[s,rule] for s in seeds]),ci_low=lo,ci_high=hi,
                        independent_N=len(g),training_repeats=len(seeds)))
                delta=np.stack([boots[s,'multiH']-boots[s,'original'] for s in seeds])
                vals=delta[selection,np.arange(B)[:,None]].mean(axis=1);lo,hi=ci(vals)
                paired.append(dict(T=T,method=method,group=name,score=score,
                    difference=np.mean([points[s,'multiH']-points[s,'original'] for s in seeds]),ci_low=lo,ci_high=hi))
        for seed in sorted(d.train_seed.unique()):
            for h in sorted(d.loc[d.process=='fGn','H'].unique()):
                pos=d[(d.train_seed==seed)&(d.process=='MRW')&(d.lambda2_true>=.06)]
                neg=d[(d.train_seed==seed)&(d.process=='fGn')&(d.H==h)]
                for score in SCORES:
                    aucrows.append(dict(T=T,method=method,train_seed=seed,H_null=h,score=score,
                        AUROC=roc_auc_score(np.r_[np.ones(len(pos)),np.zeros(len(neg))],np.r_[pos[score],neg[score]]),
                        positive_N=len(pos),negative_N=len(neg),positive_H=.5,
                        note='Unchanged scores: AUROC is identical before and after threshold correction'))
    return pd.DataFrame(per),pd.DataFrame(combined),pd.DataFrame(paired),pd.DataFrame(aucrows)


def assess(summary,per,protocol):
    p=protocol['predeclared_acceptance'];d=summary[(summary.method=='full_c025')&(summary.score=='p_mrw')]
    names=[f'fGn_H{h:g}' for h in protocol['calibration']['H']]
    null=d[(d.rule=='multiH')&d.group.isin(names)]
    individual=per[(per.method=='full_c025')&(per.score=='p_mrw')&(per.rule=='multiH')&per.group.isin(names)]
    pos=d[(d.rule=='multiH')&(d.group=='MRW_significant')]
    old=d[(d.rule=='original')&(d.group=='MRW_significant')].set_index('T')
    loss=[old.loc[r.T,'mean']-r.mean for r in pos.itertuples()]
    interpolation=d[(d.rule=='multiH')&d.group.isin(['fGn_H0.3','fGn_H0.7'])]
    checks=dict(max_mean_FPR=float(null['mean'].max()),max_checkpoint_FPR=float(individual.rate.max()),
                min_mean_TPR=float(pos['mean'].min()),max_mean_TPR_loss=float(max(loss)),
                max_interpolation_FPR=float(interpolation['mean'].max()))
    passes=dict(mean_FPR=checks['max_mean_FPR']<=p['max_training_mean_fGn_FPR_at_calibration_H'],
        checkpoint_FPR=checks['max_checkpoint_FPR']<=p['max_individual_checkpoint_fGn_FPR_at_calibration_H'],
        TPR=checks['min_mean_TPR']>=p['min_training_mean_TPR_lambda2_at_least_0_06'],
        TPR_loss=checks['max_mean_TPR_loss']<=p['max_training_mean_TPR_drop_vs_old_threshold'],
        interpolation=checks['max_interpolation_FPR']<=.10)
    return dict(observed=checks,checks=passes,all_predeclared_checks_passed=all(passes.values()),
                scope='Tested fGn H grid, fixed generator/settings; no continuous-H or process-family guarantee')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['calibrate','test'])
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--calibration-dir',type=Path)
    ap.add_argument('--quick',action='store_true');args=ap.parse_args();out=args.output
    torch.set_num_threads(1);protocol=json.loads(PROTOCOL.read_text());t0=time.time()
    before=start(out,dict(experiment='E5',stage=args.stage,quick=args.quick,
        correction_protocol_sha256=sha(PROTOCOL),script_sha256=sha(Path(__file__)),network_changed=False))
    try:
        excluded=excluded_seeds();seen=set();sources={str(OLD.relative_to(ROOT)):sha(OLD)};frozen=None
        if args.stage=='test':
            assert args.calibration_dir is not None
            cal=args.calibration_dir.resolve()
            manifest=json.loads((cal/'manifest.json').read_text())
            assert manifest['details']['correction_protocol_sha256']==sha(PROTOCOL)
            assert manifest['details']['script_sha256']==sha(Path(__file__))
            status=json.loads((cal/'status.json').read_text());assert status['status']=='COMPLETED'
            assert manifest['details']['quick']==args.quick
            path=cal/'frozen_thresholds.json';assert sha(path)==status['thresholds_sha256']
            sources[str(path.relative_to(ROOT))]=sha(path);frozen=json.loads(path.read_text())
            for p in cal.glob('raw_T*_manifest.csv'):
                excluded.update(int(x) for x in pd.read_csv(p,dtype={'data_seed':str}).data_seed)
        allpred=[]
        for T in ([512] if args.quick else protocol['lengths']):
            x,meta=make_data(T,args.stage,protocol,args.quick,excluded,seen)
            np.savez_compressed(out/f'raw_T{T}.npz',x=x)
            meta.to_csv(out/f'raw_T{T}_manifest.csv',index=False)
            df=infer(x,meta,sources);df.to_csv(out/f'predictions_T{T}.csv',index=False);allpred.append(df)
            print(args.stage,'T',T,'N',len(meta),'seconds',round(time.time()-t0),flush=True)
        df=pd.concat(allpred,ignore_index=True)
        if args.stage=='calibrate':
            groups,frozen=calibrate(df,protocol['alpha'])
            pd.DataFrame(groups).to_csv(out/'calibration_quantiles_by_H.csv',index=False)
            dump(out/'frozen_thresholds.json',frozen);pd.DataFrame(frozen).to_csv(out/'frozen_thresholds.csv',index=False)
            extras=dict(thresholds_sha256=sha(out/'frozen_thresholds.json'),test_generated=False)
        else:
            per,summary,paired,aucs=summarize(df,frozen)
            for name,table in [('rates_by_checkpoint',per),('rates_summary',summary),('paired_rate_changes',paired),('unchanged_auc_by_checkpoint',aucs)]:
                table.to_csv(out/f'{name}.csv',index=False)
            verdict=assess(summary,per,protocol);dump(out/'predeclared_assessment.json',verdict)
            extras=dict(thresholds_unchanged=True,assessment=verdict)
        for p,h in sources.items():assert sha(ROOT/p)==h
        dump(out/'input_hashes.json',sources)
        finish(out,before,seconds=time.time()-t0,independent_new_N=len(seen),data_seed_overlap=0,**extras)
    except Exception:
        import traceback
        dump(out/'status.json',dict(status='FAILED',traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
