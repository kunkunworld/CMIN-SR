"""Readable paired effects without conflating target fit and raw discrimination."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    source = ROOT/'revision/results/backend_ablation_summary_formal'; out = source/'figures'
    ci = pd.read_csv(source/'metrics_intervals.csv'); effects = pd.read_csv(source/'paired_mlp_effects.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    plt.rcParams.update({'font.size': 10})
    g = ci[(ci.scope == 'analytic_target_agreement') & (ci.setting == 'all')].set_index('method').loc[['selected_linear', 'mlp']]
    axes[0].errorbar(g['mean'].to_numpy(), [1, 0], xerr=np.vstack([g['mean']-g.ci_low, g.ci_high-g['mean']]),
        fmt='o', capsize=4, color='#327c9b')
    axes[0].set_yticks([1, 0], ['Affine', 'MLP']); axes[0].set_ylim(-.7, 1.7)
    axes[0].set_xlim(0, .033); axes[0].set_xlabel('Prescribed-target MSE (lower is better)')
    axes[0].set_title('(a) Analytic target agreement', fontsize=11)
    for y, row in zip([1, 0], g.itertuples()):
        axes[0].annotate(f'{row.mean:.5f}', (row.mean, y), xytext=(0, 12), textcoords='offset points', ha='center', fontsize=9)
    settings = ['raw|frozen_neural_seed51001|T1024|score_mrw', 'raw|structure_ols|T1024|score_mrw']
    e = effects[(effects.scope == 'binary_discrimination') & (effects.metric == 'AUROC') & (effects.baseline == 'selected_linear')].set_index('setting').loc[settings]
    axes[1].axvline(0, color='.5', linewidth=1, linestyle='--')
    axes[1].errorbar(e.delta.to_numpy(), [1, 0], xerr=np.vstack([e.delta-e.ci_low, e.ci_high-e.delta]), fmt='o', capsize=4, color='#b84942')
    axes[1].set_yticks([1, 0], ['Frozen neural spectrum', 'OLS spectrum'])
    axes[1].set_ylim(-.7, 1.7); axes[1].set_xlim(-.03, .005)
    axes[1].set_xlabel('MLP minus affine AUROC (T = 1024)')
    axes[1].set_title('(b) Raw MRW vs fGn discrimination', fontsize=11)
    for ax in axes:
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(out/'backend_fit_vs_transfer.png', dpi=220); fig.savefig(out/'backend_fit_vs_transfer.pdf'); plt.close(fig)
    (out/'figure_manifest.md').write_text('# E8 figure sources\n\n'
        '- backend_fit_vs_transfer.png/pdf: metrics_intervals.csv (analytic overall target MSE), paired_mlp_effects.csv (T1024 p_MRW AUROC delta). Horizontal bars are paired/hierarchical 95% intervals. Target agreement is not physical calibration.\n'
        '- backend_raw_discrimination.png/pdf: metrics_intervals.csv, frozen frontend, T512/1024/2048; absolute AUROC/TPR on full 0-1 axes intentionally show ceiling behavior.\n'
        '- backend_ood_decisions.png/pdf: metrics_intervals.csv, conditional frozen frontend and fixed clean validation thresholds, 250 independent samples per stress family, three backend seeds. No retuning on stress.\n'
        '- All plots use existing sample predictions; no new fitting or altered experimental numbers.\n', encoding='utf-8')


if __name__ == '__main__': main()
