"""Matched uncertainty summaries; training repetitions are not new test samples."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from revision.analysis_tools import safe_corr,dump_json

B=2000


def strata_indices(meta,seed=99221):
    rng=np.random.default_rng(seed)
    groups=meta.reset_index(drop=True).groupby(['seed','process','H','lambda2_true'],sort=True).indices
    return np.concatenate([rng.choice(ids,(B,len(ids)),replace=True) for ids in groups.values()],axis=1)


def correlations(y,p):
    a=y-y.mean(axis=-1,keepdims=True);b=p-p.mean(axis=-1,keepdims=True)
    den=np.sqrt((a*a).sum(axis=-1)*(b*b).sum(axis=-1))
    return np.divide((a*b).sum(axis=-1),den,out=np.full_like(den,np.nan),where=den>1e-14)


def interval(x):
    vals=x[np.isfinite(x)]
    return np.percentile(vals,[2.5,97.5]) if len(vals) else [np.nan,np.nan]


def hierarchy(values,seed=99):
    # values[training_repeat, bootstrap_repeat]; seed resampling does not create test N.
    r=len(values);sel=np.random.default_rng(seed).integers(r,size=(B,r))
    return values[sel,np.arange(B)[:,None]].mean(axis=1)


def metric_vector(df):
    d={k:df[k].mean() for k in ['zeta_mae','lambda2_mae','H_mae','d2_error','second_diff_norm',
        'third_diff_norm','lambda2_proj','lambda2_signed','mrw_residual','mono_residual',
        'p_mrw','p_curved','p_mono','p_scaling','p_boundary']}
    d['lambda2_corr']=safe_corr(df.lambda2_true,df.lambda2_proj)
    d['lambda2_signed_corr']=safe_corr(df.lambda2_true,df.lambda2_signed)
    d['lambda2_signed_mae']=float((df.lambda2_signed-df.lambda2_true).abs().mean())
    d['lambda2_signed_bias']=float((df.lambda2_signed-df.lambda2_true).mean())
    d['negative_coordinate_fraction']=float((df.lambda2_signed<0).mean())
    d['lambda_floor_fraction']=float((df.lambda2_proj<=1.000001e-6).mean())
    d['lambda2_bias']=float((df.lambda2_proj-df.lambda2_true).mean())
    d['independent_N']=len(df)
    return d


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evaluation-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
    source=args.evaluation_dir;out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    if json.loads((source/'status.json').read_text())['status']!='PASS':raise SystemExit('Evaluation not PASS')
    test=pd.concat([pd.read_csv(p) for p in source.glob('results_test_T*.csv')],ignore_index=True)
    val=pd.concat([pd.read_csv(p) for p in source.glob('results_threshold_validation_T*.csv')],ignore_index=True)
    rows=[];cirows=[];paired=[];rocrows=[];roccurves=[];powerrows=[]
    for T,dt in test.groupby('T'):
        dt=dt.copy();dt['group']=np.where(dt.process=='MRW','MRW_'+dt.lambda_band,
                                      np.where(dt.process=='fGn','fGn_H'+dt.H.astype(str),'Gaussian'))
        for (method,seed,group),g in dt.groupby(['method','train_seed','group']):
            rows.append(dict(T=T,method=method,train_seed=seed,group=group,**metric_vector(g)))
        for (method,seed),g in dt[dt.process=='MRW'].groupby(['method','train_seed']):
            rows.append(dict(T=T,method=method,train_seed=seed,group='MRW_all',**metric_vector(g)))
        for group in ['MRW_all','fGn_same_H','Gaussian']:
            g=dt[dt.process=='MRW'] if group=='MRW_all' else dt[(dt.process=='fGn')&(dt.H==.5)] if group=='fGn_same_H' else dt[dt.process=='Gaussian']
            ref=g[g.method=='structure_ols'].sort_values('sample_id').reset_index(drop=True)
            idx=strata_indices(ref);truth=ref.lambda2_true.to_numpy();ref_metrics=metric_vector(ref)
            ref_boot={k:ref[k].to_numpy()[idx].mean(axis=1) for k in ['zeta_mae','lambda2_mae','H_mae','d2_error','p_mrw','p_curved','p_mono']}
            ref_boot['lambda2_corr']=correlations(truth[idx],ref.lambda2_proj.to_numpy()[idx])
            method_boots={};method_points={}
            for method,gm in g.groupby('method'):
                boots={k:[] for k in ref_boot};points={k:[] for k in ref_boot}
                for seed,gs in gm.groupby('train_seed'):
                    gs=gs.set_index('sample_id').loc[ref.sample_id].reset_index()
                    for k in boots:
                        if k=='lambda2_corr':
                            b=correlations(truth[idx],gs.lambda2_proj.to_numpy()[idx]);p=safe_corr(truth,gs.lambda2_proj)
                        else:b=gs[k].to_numpy()[idx].mean(axis=1);p=float(gs[k].mean())
                        boots[k].append(b);points[k].append(p)
                method_boots[method]={k:np.array(v) for k,v in boots.items()}
                method_points[method]={k:np.array(v) for k,v in points.items()}
                for metric in boots:
                    p=np.array(points[metric]);values=hierarchy(np.array(boots[metric]));lo,hi=interval(values)
                    point=float(p.mean()) if np.isfinite(p).all() else np.nan
                    cirows.append(dict(T=T,group=group,method=method,metric=metric,mean=point,
                        training_seed_sd=float(p.std(ddof=1)) if len(p)>1 and np.isfinite(p).all() else np.nan,
                        ci_low=lo,ci_high=hi,training_repeats=len(p),independent_test_N=len(ref)))
                    plo,phi=interval(values-ref_boot[metric]);paired.append(dict(T=T,group=group,method=method,
                        baseline='structure_ols',metric=metric,delta=point-ref_metrics[metric],ci_low=plo,ci_high=phi))
            for baseline in ['ridge','ridge_logvol','nonoverlap_prior','mfdfa','mfdfa_quadratic','zeta_only_c025']:
                if baseline not in method_boots:continue
                for metric,values in method_boots['full_c025'].items():
                    other=method_boots[baseline][metric]
                    # Matched training seeds for ridge/loss ablation; singleton deterministic baseline broadcasts.
                    delta=hierarchy(values-other);lo,hi=interval(delta)
                    point=np.mean(method_points['full_c025'][metric]-method_points[baseline][metric])
                    paired.append(dict(T=T,group=group,method='full_c025',baseline=baseline,
                                       metric=metric,delta=point,ci_low=lo,ci_high=hi))
        # Operational classification: same-H fGn null; significant MRW alternative only.
        d=dt[((dt.process=='MRW')&(dt.lambda2_true>=.06))|((dt.process=='fGn')&(dt.H==.5))]
        v=val[val['T']==T]
        v=v[((v.process=='MRW')&(v.lambda2_true>=.06))|((v.process=='fGn')&(v.H==.5))]
        ref=d[d.method=='structure_ols'].sort_values('sample_id').reset_index(drop=True);idx=strata_indices(ref)
        labels=(ref.process=='MRW').to_numpy();yb=labels[idx];npos=yb.sum(axis=1);nneg=(~yb).sum(axis=1)
        for (method,ts),g in d.groupby(['method','train_seed']):
            g=g.set_index('sample_id').loc[ref.sample_id].reset_index()
            vg=v[(v.method==method)&(v.train_seed==ts)]
            for score in ['lambda2_proj','p_mrw']:
                null=vg.loc[vg.process=='fGn',score].to_numpy()
                candidates=np.r_[np.unique(null),np.nextafter(np.max(null),np.inf)]
                threshold=next(t for t in candidates if np.mean(null>=t)<=.05)
                s=g[score].to_numpy();pred=s>=threshold;sample=s[idx]
                ranks=rankdata(sample,axis=1,method='average')
                aucboot=((ranks*yb).sum(axis=1)-npos*(npos+1)/2)/(npos*nneg)
                auc_lo,auc_hi=interval(aucboot)
                tpr_boot=((sample>=threshold)&yb).sum(axis=1)/npos
                fpr_boot=((sample>=threshold)&~yb).sum(axis=1)/nneg
                tl,th=interval(tpr_boot);fl,fh=interval(fpr_boot)
                rocrows.append(dict(T=T,method=method,train_seed=ts,score=score,threshold=threshold,
                    validation_FPR=float(np.mean(null>=threshold)),AUROC=roc_auc_score(labels,s),
                    AUROC_ci_low=auc_lo,AUROC_ci_high=auc_hi,TPR=float(pred[labels].mean()),
                    FPR=float(pred[~labels].mean()),TPR_ci_low=tl,TPR_ci_high=th,FPR_ci_low=fl,FPR_ci_high=fh,
                    positive_N=int(labels.sum()),negative_N=int((~labels).sum()),ci_scope='data conditional on one trained checkpoint'))
                fpr,tpr,thresholds=roc_curve(labels,s)
                roccurves.extend(dict(T=T,method=method,train_seed=ts,score=score,FPR=a,TPR=b,threshold=c)
                                 for a,b,c in zip(fpr,tpr,thresholds))
                allg=dt[(dt.method==method)&(dt.train_seed==ts)]
                for (process,h,lam),cell in allg.groupby(['process','H','lambda2_true']):
                    rate=(cell[score]>=threshold).astype(float).to_numpy()
                    rng=np.random.default_rng(99223);local=cell.reset_index(drop=True)
                    ix=np.concatenate([rng.choice(v,(B,len(v))) for v in local.groupby('seed').indices.values()],axis=1)
                    lo,hi=interval(rate[ix].mean(axis=1))
                    powerrows.append(dict(T=T,method=method,train_seed=ts,score=score,threshold=threshold,
                        process=process,H=h,lambda2_true=lam,positive_decision_rate=rate.mean(),
                        ci_low=lo,ci_high=hi,N=len(cell),threshold_null='same-H fGn validation only'))
        print(f'summary T={T} complete',flush=True)
    byseed=pd.DataFrame(rows);byseed.to_csv(out/'metrics_by_training_seed.csv',index=False)
    byseed[['T','method','train_seed','group','lambda2_mae','lambda2_signed_mae','lambda2_bias',
            'lambda2_signed_bias','lambda2_corr','lambda2_signed_corr','negative_coordinate_fraction',
            'lambda_floor_fraction','independent_N']].to_csv(out/'clipping_by_seed.csv',index=False)
    byseed.groupby(['T','method','group']).agg({k:['mean','std'] for k in ['zeta_mae','lambda2_mae','lambda2_corr','H_mae','p_mrw','p_curved','p_mono','p_boundary']}).to_csv(out/'metrics_mean_sd.csv')
    ci=pd.DataFrame(cirows);ci.to_csv(out/'metrics_intervals.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'paired_method_effects.csv',index=False)
    pd.DataFrame(rocrows).to_csv(out/'heldout_roc_thresholds.csv',index=False)
    pd.DataFrame(roccurves).to_csv(out/'roc_curves.csv',index=False)
    pd.DataFrame(powerrows).to_csv(out/'power_by_lambda_and_null_H.csv',index=False)
    methods=['structure_ols','nonoverlap_prior','mfdfa','mfdfa_quadratic','ridge','ridge_logvol','full_c025','zeta_only_c025']
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for method in methods:
        for ax,metric,label in zip(axes,['lambda2_mae','zeta_mae'],['Intermittency MAE','Scaling-function MAE']):
            p=ci[(ci.group=='MRW_all')&(ci.method==method)&(ci.metric==metric)].sort_values('T')
            ax.plot(p['T'],p['mean'],marker='o',label=method)
            ax.fill_between(p['T'],p.ci_low,p.ci_high,alpha=.1);ax.set_xlabel('Sample length');ax.set_ylabel(label)
    axes[1].legend(fontsize=7)
    fig.savefig(out/'neural_vs_classical.png',dpi=220);fig.savefig(out/'neural_vs_classical.pdf');plt.close(fig)
    text=['# Corrected-data neural comparison','',
          'Primary model: full_c025. All configurations/seeds reported; no test winner selection.',
          'Intervals use paired stratified test resampling and training-seed resampling (only 3 training seeds).',
          'Training seeds also change training data; SD is not initialization-only variation.',
          'ROC intervals are conditional on individual checkpoints. Thresholds use independent validation.',
          'MRW targets are audited at H=.5 only; T=2048 is length extrapolation.', '',
          '## Primary results at T=1024',
          ci[(ci['T']==1024)&ci.method.isin(methods)&ci.metric.isin(['lambda2_mae','lambda2_corr','zeta_mae'])].to_csv(index=False)]
    (out/'summary.md').write_text('\n'.join(text),encoding='utf-8')
    dump_json(out/'status.json',dict(status='PASS',bootstrap_repeats=B,scope='Estimation, not proof of mechanism'))


if __name__=='__main__':main()
