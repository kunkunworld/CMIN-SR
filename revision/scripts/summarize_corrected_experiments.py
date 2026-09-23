"""Fixed-design stratified intervals, clipping audit, ROC and paired effects."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from revision.analysis_tools import safe_corr,dump_json


def indices(df,B,seed=99101,strata=('seed','lambda2_true')):
    rng=np.random.default_rng(seed);df=df.reset_index(drop=True)
    return np.concatenate([rng.choice(np.asarray(ix),size=(B,len(ix)))
                           for ix in df.groupby(list(strata),dropna=False).indices.values()],axis=1)


def corr_rows(x,y):
    xc=x-x.mean(axis=-1,keepdims=True);yc=y-y.mean(axis=-1,keepdims=True)
    den=np.sqrt((xc*xc).sum(axis=-1)*(yc*yc).sum(axis=-1))
    return np.divide((xc*yc).sum(axis=-1),den,out=np.full_like(den,np.nan),where=den>1e-14)


def bounds(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    return np.quantile(a,[.025,.975]) if len(a) else [np.nan,np.nan]


def combined(base,neural):
    d=pd.read_csv(base/'sample_results.csv');d['pipeline']=d['method']
    n=pd.read_csv(neural/'frozen_encoder_results.csv')
    n['pipeline']=n['method']+'_c'+n['coefficient'].map(lambda x:f'{x:g}')
    return pd.concat([d,n],ignore_index=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--bootstrap',type=int,default=2000)
    args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    b=ROOT/'revision/results'
    d=combined(b/'corrected_recovery_formal',b/'frozen_encoder_formal')
    val=combined(b/'corrected_recovery_validation',b/'frozen_encoder_validation')
    B=args.bootstrap;rec=[];byseed=[];clip=[]
    for key,g in d[d.process=='MRW'].groupby(['T','generator','pipeline']):
        T,gen,method=key;g=g.sort_values('sample_id').reset_index(drop=True)
        ix=indices(g,B);truth=g.lambda2_true.to_numpy();pred=g.lambda2_proj.to_numpy()
        r=safe_corr(truth,pred);rb=corr_rows(truth[ix],pred[ix]);rl,rh=bounds(rb)
        err=np.abs(truth-pred);lo,hi=bounds(err[ix].mean(axis=1))
        seedrs=[]
        for s,h in g.groupby('seed'):
            sr=safe_corr(h.lambda2_true,h.lambda2_proj);seedrs.append(sr)
            byseed.append(dict(T=T,generator=gen,pipeline=method,seed=s,n=len(h),r=sr,
                               lambda2_mae=h.lambda2_mae.mean(),zeta_mae=h.zeta_mae.mean()))
        validseed=np.asarray(seedrs)[np.isfinite(seedrs)]
        rec.append(dict(T=T,generator=gen,pipeline=method,n=len(g),n_seeds=g.seed.nunique(),
                        r=r,r_ci_low=rl,r_ci_high=rh,bootstrap_r_defined=int(np.isfinite(rb).sum()),
                        r_seed_mean=np.mean(validseed) if len(validseed) else np.nan,
                        r_seed_sd=np.std(validseed,ddof=1) if len(validseed)>1 else np.nan,
                        spearman=float(spearmanr(truth,pred).statistic) if np.std(pred)>1e-12 else np.nan,
                        lambda2_mae=err.mean(),lambda2_mae_ci_low=lo,lambda2_mae_ci_high=hi,
                        zeta_mae=g.zeta_mae.mean(),H_mae=g.H_mae.mean(),d2_error=g.d2_error.mean(),
                        p_mrw=g.p_mrw.mean(),p_curved=g.p_curved.mean(),negative_fraction=g.negative_signed.mean()))
        for col in ['lambda2_signed','lambda2_nonnegative','lambda2_proj']:
            y=g[col].to_numpy()
            clip.append(dict(T=T,generator=gen,pipeline=method,coordinate=col,n=len(g),
                              r=safe_corr(truth,y),mae=np.abs(y-truth).mean(),bias=np.mean(y-truth),
                              std=y.std(),negative_fraction=np.mean(y<0),floor_fraction=np.mean(y<=1.0001e-6),
                              correlation_status='defined' if y.std()>1e-12 else 'undefined_constant_output'))
    pd.DataFrame(rec).to_csv(out/'recovery_with_intervals.csv',index=False)
    pd.DataFrame(byseed).to_csv(out/'recovery_by_seed.csv',index=False)
    pd.DataFrame(clip).to_csv(out/'clipping_audit.csv',index=False)
    # Process/band means and sample-level uncertainty conditional on each frozen pipeline.
    dc=d[d.generator=='corrected'].copy()
    dc['band']=np.where(dc.process=='MRW',np.where(dc.lambda2_true<.03,'low',np.where(dc.lambda2_true<.08,'medium','high')),'control')
    prows=[]
    for key,g in dc.groupby(['T','process','H','band','pipeline']):
        if key[1]=='fGn_modulated_stress':continue
        ix=indices(g.reset_index(drop=True),B)
        for metric in ['p_mrw','p_curved','p_mono','p_boundary','p_scaling','zeta_mae','lambda2_proj','second_diff_norm','gain']:
            y=g[metric].to_numpy();lo,hi=bounds(y[ix].mean(axis=1))
            seedmeans=g.groupby('seed')[metric].mean()
            prows.append(dict(T=key[0],process=key[1],H=key[2],band=key[3],pipeline=key[4],metric=metric,
                              n=len(g),mean=np.mean(y),ci_low=lo,ci_high=hi,seed_sd=seedmeans.std()))
    pd.DataFrame(prows).to_csv(out/'process_diagnostics_intervals.csv',index=False)
    # Paired changes, same sample ID, in total MAE and lambda correlation.
    paired=[]
    for T,g in d[(d.process=='MRW')&(d['method']=='structure_ols')].groupby('T'):
        p=g.pivot(index=['sample_id','seed','lambda2_true'],columns='generator',values=['lambda2_proj','lambda2_mae']).reset_index()
        meta=pd.DataFrame({'seed':p['seed'],'lambda2_true':p['lambda2_true']});ix=indices(meta,B)
        true=p['lambda2_true'].to_numpy();old=p[('lambda2_proj','legacy')].to_numpy();new=p[('lambda2_proj','corrected')].to_numpy()
        rd=corr_rows(true[ix],new[ix])-corr_rows(true[ix],old[ix]);lo,hi=bounds(rd)
        diff=np.abs(new-true)-np.abs(old-true);ml,mh=bounds(diff[ix].mean(axis=1))
        paired.append(dict(T=T,n=len(p),r_change=safe_corr(true,new)-safe_corr(true,old),r_change_low=lo,r_change_high=hi,
                           mae_change=diff.mean(),mae_change_low=ml,mae_change_high=mh))
    pd.DataFrame(paired).to_csv(out/'paired_generator_effect.csv',index=False)
    # ROC is explicitly scores, not posterior calibration. Threshold uses only validation negatives.
    roc=[];curves=[]
    for key,g0 in dc.groupby(['T','pipeline']):
        T,pipeline=key
        for neg in ['fGn','Gaussian']:
            select=lambda f:f[((f.process=='MRW')&(f.lambda2_true>=.06))|((f.process==neg)&(f.H==.5))]
            g=select(g0).sort_values('sample_id').reset_index(drop=True)
            vg=select(val[(val['T']==T)&(val.pipeline==pipeline)&(val.generator=='corrected')])
            if g.empty or vg.empty or g.process.nunique()!=2:continue
            y=(g.process=='MRW').to_numpy().astype(int);vy=(vg.process=='MRW').to_numpy().astype(int)
            ix=indices(g.assign(label=y),B,strata=('seed','label','lambda2_true'))
            for score in ['lambda2_proj','p_mrw','native_p_mrw','legacy_feature_p_mrw_cal']:
                if score not in g or g[score].isna().any() or vg[score].isna().any():continue
                scores=g[score].to_numpy();vs=vg[score].to_numpy()
                if not np.isfinite(scores).all():raise ValueError('Nonfinite score')
                negval=np.sort(vs[vy==0]);threshold=negval[int(np.ceil(.95*len(negval)))-1]
                decision=scores>threshold
                auc=roc_auc_score(y,scores)
                # Pairwise comparison avoids ties being treated as strictly ordered.
                aucb=[]
                for batch in np.array_split(ix,20):
                    for ii in batch:aucb.append(roc_auc_score(y[ii],scores[ii]))
                lo,hi=bounds(aucb)
                tprb=np.array([decision[ii][y[ii]==1].mean() for ii in ix]);fprb=np.array([decision[ii][y[ii]==0].mean() for ii in ix])
                tl,th=bounds(tprb);fl,fh=bounds(fprb)
                roc.append(dict(T=T,pipeline=pipeline,negative=neg,score=score,n_positive=int(y.sum()),n_negative=int((1-y).sum()),
                    auc=auc,auc_low=lo,auc_high=hi,threshold=threshold,threshold_operator='>',
                    validation_n_negative=len(negval),validation_FPR=np.mean(negval>threshold),
                    test_TPR=decision[y==1].mean(),TPR_low=tl,TPR_high=th,
                    test_FPR=decision[y==0].mean(),FPR_low=fl,FPR_high=fh))
                fpr,tpr,thresholds=roc_curve(y,scores)
                curves.extend([dict(T=T,pipeline=pipeline,negative=neg,score=score,FPR=f,TPR=t,threshold=v)
                               for f,t,v in zip(fpr,tpr,thresholds)])
    pd.DataFrame(roc).to_csv(out/'heldout_roc_thresholds.csv',index=False)
    pd.DataFrame(curves).to_csv(out/'roc_curves.csv',index=False)
    # Figures are new revision assets, never manuscript replacements.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    r=pd.DataFrame(rec);fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for gen,color in [('legacy','#777777'),('corrected','#146b8c')]:
        s=r[(r.generator==gen)&(r.pipeline=='structure_ols')].sort_values('T')
        axes[0].errorbar(s['T'],s.r,yerr=[s.r-s.r_ci_low,s.r_ci_high-s.r],marker='o',color=color,label=gen)
        axes[1].errorbar(s['T'],s.lambda2_mae,yerr=[s.lambda2_mae-s.lambda2_mae_ci_low,s.lambda2_mae_ci_high-s.lambda2_mae],marker='o',color=color)
    for ax in axes:ax.set_xscale('log',base=2);ax.set_xlabel('Length T');ax.grid(alpha=.15)
    axes[0].set_ylabel('Curvature correlation');axes[1].set_ylabel('Curvature MAE');axes[0].legend()
    fig.savefig(out/'generator_correction_recovery.png',dpi=200);plt.close(fig)
    s=r[(r['T']==1024)&(r.generator=='corrected')&r.pipeline.isin(['structure_ols','v3_c0.25','aligned_c0.25','curvature_preserved_c0.25'])]
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');ax.bar(s.pipeline,s.r,color=['#146b8c','#a86e41','#7b657e','#4b8063'])
    ax.set_ylabel('Curvature correlation, T=1024');ax.tick_params(axis='x',rotation=15)
    fig.savefig(out/'frozen_checkpoint_attribution.png',dpi=200);plt.close(fig)
    dump_json(out/'status.json',dict(status='PASS',bootstrap=B,recovery_rows=len(rec),roc_rows=len(roc),
        interval_scope='stratified by seed/lambda; conditional data sampling uncertainty, NOT training uncertainty',
        confirmatory_scope='H=.5 MRW; same-H negatives; corrected generator',
        warning='Old neural checkpoints evaluated under distribution shift; coefficient ablation inference-only'))
    print(r[(r.pipeline=='structure_ols')].to_string(index=False),flush=True)

if __name__=='__main__':main()
