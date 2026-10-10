"""Independent nonwhite fGn controls with R1 models and thresholds frozen."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from common import *
import argparse
import time
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
from revision.retraining import simulate, make_model, prior_and_ols
from revision.analysis_tools import projection_metrics, FrozenCalibrator
from revision.linear_control import logvol_features

TRAIN=ROOT/'revision/results/neural_retrain_formal_attempt3'
SOURCE=ROOT/'revision/results/neural_eval_formal'
THRESHOLD=ROOT/'revision/results/neural_summary_formal/heldout_roc_thresholds.csv'
METHODS=['structure_ols','ridge','ridge_logvol','full_c025']
SEEDS=[51001,51002,51003]


def holdout(T,quick):
    xs=[];meta=[]
    for seed in ([202610081] if quick else range(202610081,202610086)):
        for hi,h in enumerate([.2,.4,.5,.6,.8]):
            for i in range(2 if quick else 40):
                sd=int(np.random.SeedSequence([20261007,seed,T,hi,i]).generate_state(1)[0])
                xs.append(simulate(T,'fGn',h,0.,sd))
                meta.append(dict(sample_id=f'R2_fgn_{T}_{seed}_{hi}_{i}',data_seed=sd,seed=seed,
                                 T=T,process='fGn',H=h,lambda2_true=0.,split='R2_test'))
    return torch.tensor(np.stack(xs)),pd.DataFrame(meta)


def auc(pos,neg):
    n=np.sort(neg)
    return float(np.mean((np.searchsorted(n,pos,'left')+np.searchsorted(n,pos,'right'))/(2*len(n))))


def wilson(k,n):
    z=norm.ppf(.975);p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,center-half),min(1,center+half)


def summarize(df,old,thresholds):
    per=[];combined=[]
    for (T,h,method),group in df.groupby(['T','H','method']):
        bs={};ps={};seeds=sorted(group.train_seed.unique())
        for ts in seeds:
            g=group[group.train_seed==ts].sort_values('sample_id').reset_index(drop=True)
            p=old[(old.method==method)&(old.train_seed==ts)&(old.process=='MRW')&
                  (old.lambda2_true>=.06)].sort_values('sample_id').reset_index(drop=True)
            assert len(p)==400
            ni=indices(g);pi=indices(p,seed=202610080)
            ps[ts]={};bs[ts]={}
            for score in ['lambda2_proj','p_mrw']:
                tr=thresholds[(thresholds.T_length==T)&(thresholds.method==method)&
                              (thresholds.train_seed==ts)&(thresholds.score==score)]
                assert len(tr)==1
                threshold=float(tr.iloc[0].threshold)
                neg=g[score].to_numpy();pos=p[score].to_numpy()
                decision=neg>=threshold
                a=auc(pos,neg)
                assert abs(a-roc_auc_score(np.r_[np.ones(len(pos)),np.zeros(len(neg))],np.r_[pos,neg]))<1e-12
                values={'FPR':decision.mean(),'AUROC':a,'TPR':np.mean(pos>=threshold)}
                boots={'FPR':decision[ni].mean(axis=1),
                       'TPR':(pos[pi]>=threshold).mean(axis=1),
                       'AUROC':np.array([auc(pos[j],neg[k]) for j,k in zip(pi,ni)])}
                wlo,whi=wilson(int(decision.sum()),len(neg))
                for metric,value in values.items():
                    lo,hi=ci(boots[metric]);key=(score,metric)
                    ps[ts][key]=value;bs[ts][key]=boots[metric]
                    per.append(dict(T=T,H_null=h,method=method,train_seed=ts,score=score,metric=metric,
                        value=value,ci_low=lo,ci_high=hi,threshold=threshold,
                        FPR_wilson_low=wlo if metric=='FPR' else np.nan,
                        FPR_wilson_high=whi if metric=='FPR' else np.nan,
                        negative_N=len(neg),positive_N=len(pos),positive_H=.5,
                        interval_scope='conditional on one frozen checkpoint and validation threshold'))
            for metric,arr in [('spectrum_mae',g.zeta_mae.to_numpy()),
                               ('signed_bias',g.lambda2_signed.to_numpy()),
                               ('signed_mae',np.abs(g.lambda2_signed.to_numpy())),
                               ('negative_fraction',(g.lambda2_signed.to_numpy()<0).astype(float))]:
                key=('spectrum',metric);ps[ts][key]=arr.mean();bs[ts][key]=arr[ni].mean(axis=1)
        selection=np.random.default_rng(202610077).integers(len(seeds),size=(B,len(seeds)))
        for (score,metric) in ps[seeds[0]]:
            a=np.stack([bs[s][score,metric] for s in seeds])
            boots=a[selection,np.arange(B)[:,None]].mean(axis=1);lo,hi=ci(boots)
            combined.append(dict(T=T,H_null=h,method=method,score=score,metric=metric,
                mean=np.mean([ps[s][score,metric] for s in seeds]),ci_low=lo,ci_high=hi,
                training_repeats=len(seeds),negative_N=len(g),positive_N=400,
                positive_H=.5,threshold_source='R1 independent H=.5 validation; unchanged',
                note='AUROC contrasts H=.5 MRW with fGn at stated H; not same-H MRW for H!=.5'))
    return per,combined


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--quick',action='store_true')
    args=ap.parse_args();out=args.output;torch.set_num_threads(1);t0=time.time()
    before=start(out,dict(experiment='E4',quick=args.quick,seeds=list(range(202610081,202610086)),
        N_per_H=200,H=[.2,.4,.5,.6,.8],threshold_sha256=sha(THRESHOLD),
        primary='full_c025',threshold_retuning=False,MRW_positive_H=.5))
    try:
        thresholds=pd.read_csv(THRESHOLD).rename(columns={'T':'T_length'})
        sources={str(THRESHOLD.relative_to(ROOT)):sha(THRESHOLD)}
        for f in [TRAIN/'data_manifest.csv',*SOURCE.glob('*_manifest.csv')]:
            sources[str(f.relative_to(ROOT))]=sha(f)
        excluded=set()
        for f in [TRAIN/'data_manifest.csv',*SOURCE.glob('*_manifest.csv')]:
            excluded.update(pd.read_csv(f).data_seed.tolist())
        cal=FrozenCalibrator();runs=pd.read_csv(TRAIN/'training_runs.csv');allper=[];allcombined=[];newseeds=set()
        sources[cal.manifest['path']]=cal.manifest['sha256']
        for T in ([512] if args.quick else [512,1024,2048]):
            x,meta=holdout(T,args.quick)
            assert not meta.data_seed.duplicated().any()
            assert not (set(meta.data_seed)&(excluded|newseeds))
            newseeds.update(meta.data_seed)
            np.savez_compressed(out/f'fgn_T{T}.npz',x=x.numpy())
            meta.to_csv(out/f'fgn_T{T}_manifest.csv',index=False)
            ref=make_model(90001,0.,'cuda');prior,ols=prior_and_ols(ref,x,'cuda');del ref
            arrays=[('structure_ols',-1,ols)];lv=logvol_features(x)
            for seed in SEEDS:
                for method,path,feat in [('ridge',TRAIN/f'seed_{seed}/ridge.npz',np.c_[ols,prior]),
                    ('ridge_logvol',SOURCE/f'ridge_logvol_seed{seed}.npz',np.c_[ols,prior,lv])]:
                    sources[str(path.relative_to(ROOT))]=sha(path)
                    st=np.load(path);z=((feat-st['mean'])/st['scale'])@st['coef'].T+st['intercept']
                    arrays.append((method,seed,z))
                r=runs[(runs.config=='full_c025')&(runs.seed==seed)].iloc[0];cp=TRAIN/r.checkpoint
                assert sha(cp)==r.checkpoint_sha256;sources[str(cp.relative_to(ROOT))]=sha(cp)
                model=make_model(seed,.25,'cuda');model.load_state_dict(torch.load(cp,map_location='cpu')['model_state_dict'],strict=True)
                model.eval()
                with torch.no_grad():
                    z=np.concatenate([model(x[i:i+64].cuda())['zeta_emp'].cpu().numpy() for i in range(0,len(x),64)])
                arrays.append(('full_c025',seed,z));del model
            rows=[]
            for method,seed,zs in arrays:
                assert np.isfinite(zs).all()
                for info,z in zip(meta.to_dict('records'),zs):
                    rows.append(dict(**info,method=method,train_seed=seed,
                        **projection_metrics(z,Q*info['H'],info['H'],0.),**cal.scores(z)))
            df=pd.DataFrame(rows);df.to_csv(out/f'predictions_T{T}.csv',index=False)
            oldpath=SOURCE/f'results_test_T{T}.csv';sources[str(oldpath.relative_to(ROOT))]=sha(oldpath)
            per,combined=summarize(df,pd.read_csv(oldpath),thresholds);allper+=per;allcombined+=combined
            print('nonwhite fGn',T,'N',len(meta),'elapsed',round(time.time()-t0),flush=True)
        pd.DataFrame(allper).to_csv(out/'threshold_and_auc_by_seed.csv',index=False)
        pd.DataFrame(allcombined).to_csv(out/'nonwhite_summary.csv',index=False)
        dump(out/'input_hashes.json',sources)
        for path,h in sources.items():assert sha(ROOT/path)==h
        finish(out,before,seconds=time.time()-t0,independent_new_N=len(newseeds),training_validation_test_overlap=0,
               frozen_thresholds=True,unchanged_input_hashes=True,calibrator=cal.manifest)
    except Exception:
        import traceback
        dump(out/'status.json',dict(status='FAILED',traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
