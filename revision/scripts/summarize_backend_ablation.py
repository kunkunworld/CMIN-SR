"""Paired uncertainty and operating-point analysis for E8; no test-set tuning."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import argparse
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from revision.backend_ablation import null_threshold, SCORES
from revision.analysis_tools import dump_json, hashfile


def binary_subset(df):
    if df.domain.iloc[0] == 'analytic':
        pos = df.family.eq('curved_mrw'); neg = df.family.isin(['linear_mono', 'noisy_mono'])
    else:
        pos = df.family.eq('MRW') & df.lambda2_true.ge(.06)
        neg = df.family.eq('fGn') & df.H.eq(.5)
    d = df[pos | neg].copy(); d['positive'] = pos[pos | neg].astype(int)
    return d


def add_selected(df, selections):
    parts = [df]
    for seed, method in selections.items():
        d = df[(df.backend_seed == seed) & (df.method == method)].copy()
        d['method'] = 'selected_linear'; parts.append(d)
    return pd.concat(parts, ignore_index=True)


def bootstrap_ids(df, B, seed=128001):
    # Analytic H/lambda are continuous draws: conditioning on exact values would
    # create singleton strata and falsely erase sampling uncertainty.
    keys = ['eval_seed', 'family']
    if df.domain.iloc[0] == 'raw': keys += ['H', 'lambda2_true']
    groups = df.reset_index(drop=True).groupby(keys, dropna=False, sort=True).indices
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(v, (B, len(v)), replace=True) for v in groups.values()], axis=1)


def auc_boot(scores, labels, ids):
    # Chunked rank-sum AUROC: average ranks explicitly handle tied scores.
    out = []
    for start in range(0, len(ids), 100):
        ii = ids[start:start+100]; yy = labels[ii]; p = yy.sum(axis=1); n = yy.shape[1]-p
        ranks = rankdata(scores[ii], axis=1)
        out.append(((ranks*yy).sum(axis=1)-p*(p+1)/2)/(p*n))
    return np.concatenate(out)


def hierarchy(values, seed=128002):
    values = np.asarray(values); B = values.shape[1]; r = len(values)
    take = np.random.default_rng(seed).integers(r, size=(B, r))
    return values[take, np.arange(B)[:, None]].mean(axis=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--quick', action='store_true')
    ap.add_argument('--training-dir', type=Path, required=True)
    ap.add_argument('--evaluation-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True); args = ap.parse_args()
    train = args.training_dir; source = args.evaluation_dir; out = args.output_dir
    out.mkdir(parents=True, exist_ok=False); t0 = time.time(); B = 100 if args.quick else 2000
    assert json.loads((source/'status.json').read_text())['status'] == 'PASS'
    evaluation_manifest = json.loads((source/'manifest.json').read_text())
    frontend_tag = f"frozen_neural_seed{evaluation_manifest['frontend_seed']}"
    selections = {int(p.parent.name.split('_')[-1]): json.loads(p.read_text())['variant'] for p in train.glob('seed_*/selected_linear.json')}
    files = pd.read_csv(source/'prediction_manifest.csv'); parts = []
    for r in files.itertuples():
        assert hashfile(source/r.file) == r.sha256
        parts.append(pd.read_csv(source/r.file))
    df = add_selected(pd.concat(parts, ignore_index=True), selections)
    val = df[df.split.isin(['validation', 'threshold_validation'])]
    test = df[df.split == 'test']; stress = df[df.split == 'stress']; unseen = df[df.split == 'unseen']
    rows = []; thresholds = {}; keys = ['domain', 'upstream', 'T', 'method', 'backend_seed']
    for group, g in test.groupby(keys):
        domain, upstream, T, method, seed = group
        v = val[(val.domain == domain) & (val.upstream == upstream) & (val['T'] == T) & (val.method == method) & (val.backend_seed == seed)]
        b = binary_subset(g); vb = binary_subset(v)
        for score in ['score_mrw', 'score_curved']:
            th = null_threshold(vb.loc[vb.positive == 0, score]); thresholds[(*group, score)] = th
            labels = b.positive.to_numpy(); values = b[score].to_numpy(); decision = values > th
            row = dict(zip(keys, group)); row.update(score=score, threshold=th,
                AUROC=roc_auc_score(labels, values), TPR=decision[labels == 1].mean(), FPR=decision[labels == 0].mean(),
                validation_FPR=float((vb.loc[vb.positive == 0, score] > th).mean()),
                positive_N=int(labels.sum()), negative_N=int((labels == 0).sum()))
            assert row['validation_FPR'] <= .05 + 1e-12
            rows.append(row)
    roc = pd.DataFrame(rows); roc.to_csv(out/'roc_operating_points_by_seed.csv', index=False)
    roc.groupby(['domain', 'upstream', 'T', 'method', 'score'])[['AUROC', 'TPR', 'FPR']].agg(['mean', 'std']).to_csv(out/'roc_summary.csv')

    analytic = test[test.domain == 'analytic']
    msecols = ['soft_target_mse'] + ['sq_error_'+k for k in SCORES]
    analytic.groupby(['method', 'backend_seed'])[msecols].mean().to_csv(out/'analytic_error_by_seed.csv')
    analytic.groupby(['method', 'backend_seed', 'family'])[msecols+SCORES].mean().to_csv(out/'analytic_by_family_seed.csv')
    raw = test[test.domain == 'raw'].copy()
    raw['group'] = raw.family
    mrw = raw.family.eq('MRW')
    raw.loc[mrw, 'group'] = np.where(raw.loc[mrw, 'lambda2_true'] < .03, 'MRW_low', np.where(raw.loc[mrw, 'lambda2_true'] < .08, 'MRW_medium', 'MRW_high'))
    fgn = raw.family.eq('fGn'); raw.loc[fgn, 'group'] = 'fGn_H'+raw.loc[fgn, 'H'].astype(str)
    raw.groupby(['upstream', 'T', 'method', 'backend_seed', 'group'])[SCORES+['score_mrw']].mean().to_csv(out/'raw_scores_by_group_seed.csv')
    raw.groupby(['upstream', 'T', 'method', 'backend_seed', 'family', 'H', 'lambda2_true'])[SCORES].mean().to_csv(out/'raw_lambda_sweep.csv')

    pressure = []
    for domain_df, domain, refdomain, refup in [(stress, 'raw_stress', 'raw', None), (unseen, 'unseen_algebraic', 'analytic', 'analytic')]:
        for group, g in domain_df.groupby(['upstream', 'T', 'method', 'backend_seed', 'family']):
            upstream, T, method, seed, family = group
            for score in ['score_mrw', 'score_curved']:
                th = thresholds[(refdomain, refup or upstream, T, method, seed, score)]
                pressure.append(dict(domain=domain, upstream=upstream, T=T, method=method, backend_seed=seed,
                    family=family, score=score, threshold=th, mean_score=g[score].mean(),
                    positive_rate=(g[score] > th).mean(), independent_N=len(g)))
    pressure = pd.DataFrame(pressure); pressure.to_csv(out/'stress_and_unseen_by_seed.csv', index=False)
    pressure.groupby(['domain', 'upstream', 'T', 'family', 'method', 'score'])[['mean_score', 'positive_rate']].agg(['mean', 'std']).to_csv(out/'stress_and_unseen_summary.csv')

    intervals = []; effects = []

    def record_comparison(scope, name, metric, points, boots, N):
        for method in points:
            h = hierarchy(np.asarray(boots[method])); lo, hi = np.quantile(h, [.025, .975])
            p = np.asarray(points[method])
            intervals.append(dict(scope=scope, setting=name, metric=metric, method=method,
                mean=p.mean(), seed_sd=p.std(ddof=1) if len(p)>1 else np.nan,
                ci_low=lo, ci_high=hi, independent_N=N, backend_repeats=len(p)))
        for base in ['selected_linear', 'rule_signed_curvature', 'rule_curvature_gain_fit']:
            if base not in points: continue
            a = np.asarray(boots['mlp']); b = np.asarray(boots[base])
            # Matched seed resampling for learned comparators; singleton rule broadcasts.
            delta = hierarchy(a-b); lo, hi = np.quantile(delta, [.025, .975])
            effects.append(dict(scope=scope, setting=name, metric=metric, baseline=base,
                delta=np.mean(points['mlp'])-np.mean(points[base]), ci_low=lo, ci_high=hi, independent_N=N))

    methods = ['mlp', 'selected_linear', 'rule_signed_curvature', 'rule_curvature_gain_fit']
    for family in ['all'] + sorted(analytic.family.unique()):
        a = analytic if family == 'all' else analytic[analytic.family == family]
        ref = a[(a.method == 'mlp') & (a.backend_seed == a.loc[a.method == 'mlp', 'backend_seed'].min())].sort_values('sample_id').reset_index(drop=True)
        ids = bootstrap_ids(ref, B); points = {}; boots = {}
        for method in ['mlp', 'selected_linear']:
            points[method] = []; boots[method] = []
            for seed, g in a[a.method == method].groupby('backend_seed'):
                values = g.set_index('sample_id').loc[ref.sample_id, 'soft_target_mse'].to_numpy()
                points[method].append(values.mean()); boots[method].append(values[ids].mean(axis=1))
        record_comparison('analytic_target_agreement', family, 'soft_target_mse', points, boots, len(ref))

    for (domain, upstream, T), dataset in test.groupby(['domain', 'upstream', 'T']):
        binary = binary_subset(dataset)
        ref = binary[(binary.method == 'mlp') & (binary.backend_seed == binary.loc[binary.method == 'mlp', 'backend_seed'].min())].sort_values('sample_id').reset_index(drop=True)
        ids = bootstrap_ids(ref, B); y = ref.positive.to_numpy()
        for score in ['score_mrw', 'score_curved']:
            points = {k: {} for k in ['AUROC', 'TPR', 'FPR']}; boots = {k: {} for k in points}
            for method in methods:
                for k in points: points[k][method] = []; boots[k][method] = []
                for seed, g in binary[binary.method == method].groupby('backend_seed'):
                    values = g.set_index('sample_id').loc[ref.sample_id, score].to_numpy()
                    threshold = thresholds[(domain, upstream, T, method, seed, score)]
                    decision = values > threshold; yy = y[ids]
                    bauc = auc_boot(values, y, ids)
                    btp = ((decision[ids]*yy).sum(axis=1)/yy.sum(axis=1))
                    bfp = ((decision[ids]*(1-yy)).sum(axis=1)/(1-yy).sum(axis=1))
                    for k, point, b in [('AUROC', roc_auc_score(y, values), bauc), ('TPR', decision[y == 1].mean(), btp), ('FPR', decision[y == 0].mean(), bfp)]:
                        points[k][method].append(point); boots[k][method].append(b)
            for k in points: record_comparison('binary_discrimination', f'{domain}|{upstream}|T{T}|{score}', k, points[k], boots[k], len(ref))
        print(f'paired ROC intervals {domain} {upstream} T={T}, elapsed={time.time()-t0:.0f}s', flush=True)

    for (upstream, family), d in stress.groupby(['upstream', 'family']):
        ref = d[(d.method == 'mlp') & (d.backend_seed == d.loc[d.method == 'mlp', 'backend_seed'].min())].sort_values('sample_id').reset_index(drop=True)
        ids = bootstrap_ids(ref, B); points = {}; boots = {}
        for method in methods:
            points[method] = []; boots[method] = []
            for seed, g in d[d.method == method].groupby('backend_seed'):
                values = g.set_index('sample_id').loc[ref.sample_id, 'score_mrw'].to_numpy()
                decision = values > thresholds[('raw', upstream, 1024, method, seed, 'score_mrw')]
                points[method].append(decision.mean()); boots[method].append(decision[ids].mean(axis=1))
        record_comparison('OOD_operating_point', upstream+'|'+family, 'positive_rate', points, boots, len(ref))
    ci = pd.DataFrame(intervals); eff = pd.DataFrame(effects)
    ci.to_csv(out/'metrics_intervals.csv', index=False); eff.to_csv(out/'paired_mlp_effects.csv', index=False)
    figdir = out/'figures'; figdir.mkdir()
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = dict(mlp='#b84942', selected_linear='#327c9b', rule_signed_curvature='#5a8757', rule_curvature_gain_fit='#937344')
    labels = dict(mlp='MLP', selected_linear='Selected affine', rule_signed_curvature='Signed curvature', rule_curvature_gain_fit='Curvature + gain + fit')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for ax, metric in zip(axes, ['AUROC', 'TPR']):
        for method in methods:
            g = ci[(ci.scope == 'binary_discrimination') & (ci.metric == metric) & (ci.method == method) & ci.setting.str.contains(frontend_tag, regex=False) & ci.setting.str.endswith('score_mrw')].copy()
            g['T'] = g.setting.str.extract(r'\|T(\d+)\|').astype(int); g = g.sort_values('T')
            ax.errorbar(g['T'].to_numpy(), g['mean'].to_numpy(), yerr=np.vstack([g['mean']-g.ci_low, g.ci_high-g['mean']]), marker='o', capsize=3, color=colors[method], label=labels[method])
        ax.set_xlabel('Sample length'); ax.set_ylabel(metric); ax.set_ylim(0, 1.02)
    axes[0].legend(fontsize=8); fig.savefig(figdir/'backend_raw_discrimination.png', dpi=220); fig.savefig(figdir/'backend_raw_discrimination.pdf'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 3.8), constrained_layout=True)
    families = ['Student_t', 'GARCH', 'iid_Gaussian_mixture', 'shuffled_MRW']
    for j, method in enumerate(methods):
        g = ci[(ci.scope == 'OOD_operating_point') & (ci.method == method) & ci.setting.str.startswith(frontend_tag)].copy()
        g['family'] = g.setting.str.split('|').str[-1]; g = g.set_index('family').loc[families]
        ax.errorbar(np.arange(4)+(j-1.5)*.12, g['mean'].to_numpy(), yerr=np.vstack([g['mean']-g.ci_low, g.ci_high-g['mean']]), fmt='o', capsize=3, color=colors[method], label=labels[method])
    ax.set_xticks(range(4), ['Student-t', 'GARCH', 'iid variance mixture', 'Shuffled MRW']); ax.set_ylabel('Positive decision rate'); ax.set_ylim(-.03, 1.05)
    ax.legend(fontsize=8, ncol=2); fig.savefig(figdir/'backend_ood_decisions.png', dpi=220); fig.savefig(figdir/'backend_ood_decisions.pdf'); plt.close(fig)
    (out/'summary.md').write_text('# E8 backend ablation\n\n'
        'See paired_mlp_effects.csv and metrics_intervals.csv for paired 95% intervals.\n'
        'Intervals resample evaluation strata and backend training seeds, conditional on one frozen upstream network and fixed validation thresholds.\n'
        'Soft-target MSE measures agreement with constructed diagnostic semantics, not physical truth or posterior calibration.\n'
        'Rules are ranking statistics; their numerical magnitude is not comparable to a neural score.\n'
        'Legacy MLP is descriptive only. The two affine variants use different preprocessing; selected_linear is chosen on validation only.\n\n'
        +ci[(ci.scope == 'analytic_target_agreement') & (ci.setting == 'all')].to_csv(index=False), encoding='utf-8')
    dump_json(out/'status.json', dict(status='PASS', bootstrap_draws=B, paired_effect_rows=len(eff),
        intervals_rows=len(ci), linear_selection=selections, elapsed_seconds=time.time()-t0,
        input_predictions_verified=True, test_threshold_tuning=False))


if __name__ == '__main__': main()
