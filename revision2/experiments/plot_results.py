"""Plot existing R2 result tables without modifying predictions or the paper."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=HERE/'figures';out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.facecolor':'white','font.family':'DejaVu Sans'})
    labels={'structure_ols':'OLS','structure_trimmed':'Trimmed SF','mfdfa':'MFDFA (linear)',
        'mfdfa_quadratic':'MFDFA (quadratic)','wavelet_leader_haar':'Haar leader approx.',
        'wtmm_haar':'Haar WTMM-style','ridge':'Ridge','ridge_logvol':'Ridge + log-vol.',
        'full_c025':'Neural correction'}
    d=pd.read_csv(HERE/'results/E2_E3_formal/all_metrics_intervals.csv')
    order=list(labels);fig,axes=plt.subplots(1,3,figsize=(10.3,4.2),layout='constrained')
    for ax,metric,title in zip(axes,['spectrum_mae','clipped_mae','clipped_corr'],
                              ['Spectrum MAE','Clipped curvature MAE','Clipped curvature correlation']):
        for i,method in enumerate(order):
            r=d[(d['T']==1024)&(d.group=='MRW_all')&(d.method==method)&(d.metric==metric)].iloc[0]
            color='#b43d40' if method=='full_c025' else '#16777e' if method=='ridge_logvol' else '#4d555d'
            ax.errorbar(r['mean'],i,xerr=[[r['mean']-r.ci_low],[r.ci_high-r['mean']]],
                        fmt='s' if method=='full_c025' else 'o',color=color,ms=4,capsize=2)
        ax.set(yticks=range(len(order)),yticklabels=[labels[k] for k in order] if ax==axes[0] else [],
               xlabel=title,ylim=(len(order)-.5,-.5))
        ax.grid(axis='x',alpha=.2)
    savefig(fig,out,'common_estimators_T1024');plt.close(fig)
    small=['structure_ols','ridge','ridge_logvol','full_c025']
    fig,axes=plt.subplots(1,3,figsize=(10.3,3.4),layout='constrained')
    for ax,T in zip(axes,[512,1024,2048]):
        for j,(metric,color,mark,name) in enumerate([('signed_mae','#985127','o','Signed'),('clipped_mae','#166b91','s','Clipped')]):
            g=d[(d['T']==T)&(d.group=='MRW_all')&(d.metric==metric)].set_index('method').loc[small]
            ax.errorbar(np.arange(4)+(j-.5)*.15,g['mean'],
                        yerr=[g['mean']-g.ci_low,g.ci_high-g['mean']],fmt=mark,color=color,label=name,capsize=2)
        ax.set(xticks=range(4),xticklabels=['OLS','Ridge','Ridge+LV','Neural'],title=f'T = {T}',
               ylabel='Curvature MAE' if ax==axes[0] else '',ylim=(0,.041))
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(frameon=False);savefig(fig,out,'signed_vs_clipped');plt.close(fig)
    d=pd.read_csv(HERE/'results/E4_formal/nonwhite_summary.csv')
    fig,axes=plt.subplots(2,2,figsize=(9,6.3),layout='constrained')
    for i,score in enumerate(['p_mrw','lambda2_proj']):
        for j,metric in enumerate(['FPR','AUROC']):
            ax=axes[i,j]
            for T,color,mark in [(512,'#985127','o'),(1024,'#166b91','s'),(2048,'#3b7653','^')]:
                g=d[(d.method=='full_c025')&(d.score==score)&(d.metric==metric)&(d['T']==T)].sort_values('H_null')
                ax.plot(g.H_null,g['mean'],marker=mark,color=color,label=f'T = {T}')
                ax.fill_between(g.H_null,g.ci_low,g.ci_high,color=color,alpha=.12)
            if metric=='FPR':ax.axhline(.05,color='black',ls='--',lw=.8)
            ax.set(xlabel='fGn H (null)',ylabel=metric,
                   title=('MRW score' if score=='p_mrw' else 'Projected curvature'),
                   xticks=[.2,.4,.5,.6,.8],ylim=(-.02,1.02) if metric=='FPR' else (.90,1.005))
            ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False);savefig(fig,out,'nonwhite_fgn_threshold_transfer');plt.close(fig)
    dump(out/'figure_sources.json',{
        'common_estimators_T1024':'results/E2_E3_formal/all_metrics_intervals.csv: MRW_all, T=1024',
        'signed_vs_clipped':'results/E2_E3_formal/all_metrics_intervals.csv: MRW_all',
        'nonwhite_fgn_threshold_transfer':'results/E4_formal/nonwhite_summary.csv: full_c025',
        'intervals':'95% bootstrap intervals; no point estimates or thresholds changed',
        'sample_count':'800 MRW per length; 200 independent fGn per H/length; three frozen neural predictions per sample',
        'auc_warning':'All positive MRW samples use H=.5; H!=.5 nulls are not same-H comparisons',
        'FPR_warning':'Frozen H=.5 validation thresholds; dashed line is nominal validation rate, not a universal guarantee'})


if __name__=='__main__':main()
