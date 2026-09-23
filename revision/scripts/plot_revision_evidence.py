"""Figures from audited revision CSVs only; never modifies manuscript assets."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
    out=a.output_dir;out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    def save(fig,name):
        fig.savefig(out/f'{name}.png',dpi=240,bbox_inches='tight');fig.savefig(out/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    base=ROOT/'revision/results';ci=pd.read_csv(base/'neural_summary_formal/metrics_intervals.csv')
    labels={'structure_ols':'Overlapping OLS','ridge':'Linear spectral correction','ridge_logvol':'Linear + log-volatility',
            'full_c025':'Neural: full loss','zeta_only_c025':'Neural: zeta-only'}
    fig,axs=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for j,(metric,ylabel) in enumerate([('zeta_mae','Scaling-function MAE'),('lambda2_mae','Intermittency MAE'),('lambda2_corr','Intermittency correlation')]):
        for method,label in labels.items():
            d=ci[(ci.group=='MRW_all')&(ci.method==method)&(ci.metric==metric)].sort_values('T')
            axs[j].plot(d['T'],d['mean'],'o-',label=label);axs[j].fill_between(d['T'],d.ci_low,d.ci_high,alpha=.12)
        axs[j].set_xlabel('Sample length');axs[j].set_ylabel(ylabel)
    axs[0].legend(fontsize=7);save(fig,'corrected_estimation_comparison')
    by=pd.read_csv(base/'neural_summary_formal/metrics_by_training_seed.csv')
    d=by[(by['T']==1024)&(by.method=='full_c025')]
    order=['fGn_H0.5','Gaussian','MRW_low','MRW_medium','MRW_high']
    fig,ax=plt.subplots(figsize=(8,3.6),layout='constrained');x=np.arange(len(order))
    for j,metric in enumerate(['p_curved','p_mrw','p_mono','p_boundary']):
        means=d.groupby('group')[metric].mean().reindex(order);sd=d.groupby('group')[metric].std().reindex(order)
        ax.errorbar(x+(j-1.5)*.14,means,yerr=sd,fmt='o',capsize=3,label=metric)
    ax.set_xticks(x,['fGn H=0.5','Gaussian','MRW low','MRW medium','MRW high']);ax.set_ylabel('Diagnostic score');ax.set_ylim(0,1.05)
    ax.legend(ncol=4,fontsize=8);save(fig,'primary_neural_geometry_scores')
    fig,axs=plt.subplots(1,2,figsize=(8,3.5),layout='constrained')
    for ax,metric in zip(axs,['lambda2_mae','zeta_mae']):
        for seed,g in by[(by['T']==1024)&(by.group=='MRW_all')&by.method.isin(['full_c0125','full_c025','full_c05'])].groupby('train_seed'):
            g=g.set_index('method').loc[['full_c0125','full_c025','full_c05']]
            ax.plot([.125,.25,.5],g[metric],marker='o',label=f'seed {seed}')
        ax.set_xlabel('Correction coefficient');ax.set_ylabel(metric.replace('_',' '))
    axs[0].legend(fontsize=8);save(fig,'matched_coefficient_ablation')
    noise=pd.read_csv(base/'corrected_noise_bridge/gap_intervals.csv')
    fig,ax=plt.subplots(figsize=(6,3.5),layout='constrained')
    for name,g in noise.groupby('perturbation'):
        ax.plot(g.noise_rmse,g.gap,marker='o',label=name.replace('_',' '));ax.fill_between(g.noise_rmse,g.ci_low,g.ci_high,alpha=.15)
    ax.set_xlabel('Added spectral noise RMSE');ax.set_ylabel('MRW-minus-linear compatibility gap');ax.legend(fontsize=8);save(fig,'controlled_spectral_noise')
    real=pd.concat([pd.read_csv(base/'real_surrogate_formal/surrogate_tests.csv'),pd.read_csv(base/'real_surrogate_block_sensitivity/surrogate_tests.csv')])
    fig,axs=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for ax,metric in zip(axs,['proxy_lambda2','proxy_p_MRW']):
        for name,g in real[real.metric==metric].groupby('factor'):
            g=g.sort_values('block_size');ax.errorbar(g.block_size,g.gap,yerr=[g.gap-g.gap_ci_low,g.gap_ci_high-g.gap],fmt='o-',capsize=2,label=name)
        ax.axhline(0,color='black',lw=.7);ax.set_xlabel('Shuffle block size (1 = full shuffle)');ax.set_ylabel(f'Original minus surrogate: {metric}')
    axs[0].legend(fontsize=8);save(fig,'surrogate_direction_sensitivity')
    sources=['neural_summary_formal','corrected_noise_bridge','real_surrogate_formal','real_surrogate_block_sensitivity']
    (out/'figure_manifest.md').write_text('# Revision figures\n\nSources are under revision/results/:\n\n'+'\n'.join('- '+s for s in sources)+
        '\n\nEstimation bands: paired test + three-training-seed bootstrap. Score bars: SD across three training seeds, not confidence intervals.\n'
        'Surrogate intervals: descriptive circular-block bootstrap across windows; not the surrogate null quantiles.\n'
        'No figure proves an MRW mechanism. No manuscript figure was replaced.\n',encoding='utf-8')


if __name__=='__main__':main()
