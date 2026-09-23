"""Matched, label-blind spectral features and backend ablation helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch import nn
from mrw_inverse.data.analytic_spectrum_dataset import (
    AnalyticSpectrumDatasetConfig, generate_analytic_spectrum_dataset,
    _mono_fit, _mrw_fit, _diagnostics,
)
from mrw_inverse.models.spectral_geometry_calibrator import SpectralGeometryCalibrator
from mrw_inverse.losses.spectrum_space_calibration_losses import spectrum_space_calibration_loss
from revision.analysis_tools import Q
from revision.retraining import finite, simulate
from revision.generation import generate_pair, standardize
from mrw_inverse.data.process_generators import gen_iid_student_t, gen_garch, gen_regime_switching_gaussian

SCORES = ['p_scaling', 'p_curved', 'p_mono', 'p_mrw', 'p_boundary']
FEATURE_KEYS = ['zeta_input', 'zeta_mono', 'zeta_mrw', 'mono_residual_norm',
    'mrw_residual_norm', 'mrw_vs_mono_gain', 'normalized_mrw_gain',
    'curvature_score', 'linearity_score', 'boundary_score']
VARIANTS = ['linear_standard', 'linear_layernorm', 'mlp']


def spectrum_features(zs):
    zs = np.asarray(zs, dtype=np.float32)
    finite('spectrum features input', zs)
    q = Q.astype(np.float32)
    rows = []
    for z in zs:
        _, zm, rm = _mono_fit(q, z)
        _, lam, zr, rr = _mrw_fit(q, z)
        d = _diagnostics(q, z, zm, zr, rm, rr, lam)
        rows.append(np.r_[z, zm, zr, rm, rr, d['mrw_vs_mono_gain'],
            d['normalized_mrw_gain'], d['curvature_score'], d['linearity_score'],
            d['boundary_score'], 0.])
    x = np.asarray(rows, dtype=np.float32)
    finite('features', x)
    assert x.shape == (len(zs), 26) and np.all(x[:, -1] == 0)
    return x


def analytic_data(n, seed, split):
    ds = generate_analytic_spectrum_dataset(AnalyticSpectrumDatasetConfig(num_samples=n, seed=seed))
    x = np.concatenate([ds[k] for k in FEATURE_KEYS] + [np.zeros((n, 1))], axis=1).astype(np.float32)
    y = np.concatenate([ds['target_' + k] for k in SCORES], axis=1)
    meta = pd.DataFrame(dict(sample_id=[f'analytic_{split}_{seed}_{i}' for i in range(n)],
        eval_seed=seed, family=ds['spectrum_type'], H=ds['H_true'].ravel(),
        lambda2_true=ds['lambda2_true'].ravel(), split=split, domain='analytic', T=0))
    np.testing.assert_allclose(x, spectrum_features(ds['zeta_input']), atol=2e-6, rtol=2e-5)
    return dict(x=x, y=y, code=ds['spectrum_code'], lam=ds['lambda2_true']), meta


class AffineBackend(nn.Module):
    def __init__(self, variant, mean, scale):
        super().__init__()
        self.variant = variant
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32))
        self.norm = nn.LayerNorm(26) if variant == 'linear_layernorm' else nn.Identity()
        self.linear = nn.Linear(26, 7)

    def forward(self, x):
        x = (x - self.mean) / self.scale if self.variant == 'linear_standard' else self.norm(x)
        return torch.sigmoid(self.linear(x))


def make_backend(variant, x, seed):
    torch.manual_seed(seed)
    if variant == 'mlp':
        # Exact existing architecture; no new hidden layers or diagnostic outputs.
        return SpectralGeometryCalibrator().net
    mean = np.mean(x, axis=0); scale = np.std(x, axis=0)
    scale[scale < 1e-7] = 1.
    return AffineBackend(variant, mean, scale)


def forward_scores(model, variant, x):
    y = model(x)
    if variant in ['mlp', 'legacy_mlp']:
        y = torch.sigmoid(y)
    finite('backend output', y)
    return y[:, :5]


def existing_loss(p, data, ids):
    outputs = {k: p[:, j:j+1] for j, k in enumerate(SCORES)}
    targets = [data['y'][ids, j:j+1] for j in range(5)]
    result = spectrum_space_calibration_loss(outputs, data['code'][ids], data['lam'][ids], *targets)
    for name in ['total', 'l_scores', 'l_rank', 'l_mono', 'l_curved', 'l_caution', 'l_boundary']:
        finite('loss ' + name, getattr(result, name))
    return result.total


def predict(model, variant, x, batch=1024):
    model.eval(); parts = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            parts.append(forward_scores(model, variant, torch.as_tensor(x[i:i+batch], dtype=torch.float32)).numpy())
    return np.concatenate(parts)


def rule_scores(x):
    matrix = np.c_[Q, -.5 * Q * (Q - 2)]
    signed = np.linalg.lstsq(matrix, x[:, :6].T, rcond=None)[0][1]
    gain_rule = x[:, 22] * x[:, 20] * .1 / (.1 + x[:, 19])
    return {'rule_signed_curvature': signed, 'rule_curvature_gain_fit': gain_rule}


def null_threshold(values, alpha=.05):
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values):
        raise ValueError('Empty validation null')
    # Strict score > threshold; ties can only reduce empirical FPR.
    k = int(np.floor(alpha * len(values)))
    return float(values[-k-1])


def clean_raw(T, split, quick=False):
    seeds = range(124001, 124006) if split == 'test' else range(125001, 125004)
    if quick: seeds = list(seeds)[:1]
    n = 2 if quick else (20 if split == 'test' else 10)
    rows = []; xs = []
    for seed in seeds:
        specs = [('MRW', .5, lam, n) for lam in [0, .005, .01, .03, .06, .1, .15, .2]]
        specs += [('fGn', .5, 0., 2*n), ('Gaussian', .5, 0., 2*n)]
        specs += [('fGn', h, 0., max(1, n//2)) for h in [.2, .4, .6, .8]]
        for group, (process, h, lam, count) in enumerate(specs):
            for i in range(count):
                sd = int(np.random.SeedSequence([81260916, seed, T, group, i]).generate_state(1, dtype=np.uint64)[0])
                xs.append(simulate(T, process, h, lam, sd))
                rows.append(dict(sample_id=f'backend_{split}_{seed}_{T}_{group}_{i}', eval_seed=seed,
                    data_seed=sd, family=process, H=h, lambda2_true=lam, split=split,
                    domain='raw', T=T))
    return np.stack(xs), pd.DataFrame(rows)


def stress_raw(quick=False):
    T = 1024; xs = []; rows = []
    seeds = [126001] if quick else range(126001, 126006)
    for seed in seeds:
        for group, family in enumerate(['Student_t', 'GARCH', 'iid_Gaussian_mixture', 'shuffled_MRW']):
            for i in range(2 if quick else 50):
                sd = int(np.random.SeedSequence([81260916, seed, T, group, i]).generate_state(1, dtype=np.uint64)[0])
                rng = np.random.default_rng(sd)
                if family == 'Student_t': x = gen_iid_student_t(T, rng)
                elif family == 'GARCH': x = gen_garch(T, rng)
                elif family == 'iid_Gaussian_mixture': x = gen_regime_switching_gaussian(T, rng)
                else:
                    x = standardize(generate_pair(T, .5, .1, sd)['corrected'])
                    np.random.default_rng(np.random.SeedSequence([sd, 1])).shuffle(x)
                xs.append(x.astype(np.float32))
                rows.append(dict(sample_id=f'backend_stress_{seed}_{group}_{i}', eval_seed=seed,
                    data_seed=sd, family=family, H=np.nan, lambda2_true=np.nan,
                    split='stress', domain='raw_stress', T=T))
    return np.stack(xs), pd.DataFrame(rows)


def unseen_curves(quick=False):
    rows = []; zs = []
    for seed in ([127001] if quick else range(127001, 127004)):
        rng = np.random.default_rng(seed)
        for family in ['quartic_departure', 'localized_cubic_kink', 'qgrid_ripple', 'convex_invalid']:
            for i in range(2 if quick else 100):
                h = rng.uniform(.2, .8); lam = rng.uniform(.06, .2); a = rng.uniform(.01, .08)
                base = Q*h - .5*lam*Q*(Q-2)
                if family == 'quartic_departure': z = base-a*(Q**4-8*Q)/9
                elif family == 'localized_cubic_kink': z = base-a*np.maximum(Q-1.5, 0)**3
                elif family == 'qgrid_ripple': z = base+a*np.sin(3.7*Q)
                else: z = Q*h+.5*lam*Q*(Q-2)
                zs.append(z)
                rows.append(dict(sample_id=f'backend_unseen_{seed}_{family}_{i}', eval_seed=seed,
                    family=family, H=h, lambda2_true=np.nan, lambda2_base=lam,
                    perturbation=a, split='unseen', domain='unseen_algebraic', T=0))
    return spectrum_features(zs), pd.DataFrame(rows)
