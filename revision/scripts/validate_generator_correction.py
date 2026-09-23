"""Covariance gates before any interpretation of corrected MRW samples."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import argparse
import hashlib
import json
from datetime import datetime
import numpy as np
import pandas as pd
from mrw_dl.generation import (build_fgn_covariance, build_log_covariance,
    sample_stationary_gaussian_from_toeplitz)
from revision.generation import sample_gaussian, embedding, generate_pair, standardize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-samples', type=int, default=256)
    ap.add_argument('--output-dir', type=Path)
    args = ap.parse_args()
    out = args.output_dir or ROOT / 'revision/e0_provenance' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_generator_fixed')
    out.mkdir(parents=True, exist_ok=False)
    src = ROOT / 'src/mrw_dl/generation.py'
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    rows, gates, embeds = [], [], []
    for T in [512, 1024, 4096]:
        covs = {'white': np.r_[1., np.zeros(T-1)],
                **{f'fgn_{h}': build_fgn_covariance(T, h) for h in [.2, .5, .8]},
                'omega': build_log_covariance(T, .1, min(512, T//4))}
        for kind, cov in covs.items():
            _, meta = embedding(cov)
            embeds.append(dict(T=T, kind=kind, **meta))
            vals = []
            for i in range(args.num_samples):
                seed = 870000 + i
                x, _ = sample_gaussian(cov, np.random.default_rng(seed))
                vals.append([float(np.mean(x*x))] + [float(np.mean(x[:-k]*x[k:])) for k in [1,2,8,32]])
            vals = np.asarray(vals)
            for j, lag in enumerate([0,1,2,8,32]):
                mean = vals[:,j].mean()
                se = vals[:,j].std(ddof=1)/np.sqrt(len(vals))
                # A generous implementation gate, not a performance CI.
                ok = abs(mean-cov[lag]) <= max(6*se, .025*cov[0])
                gates.append(bool(ok))
                rows.append(dict(T=T,kind=kind,lag=lag,target=cov[lag],observed=mean,
                                 standard_error=se,ratio_to_variance=mean/cov[0],passed=ok,n=len(vals)))
            print(T, kind, 'variance ratio', vals[:,0].mean()/cov[0], flush=True)
    # Algebraic comparison to old sampler, exact same innovations.
    cov = build_fgn_covariance(1024, .8)
    old = sample_stationary_gaussian_from_toeplitz(cov,np.random.default_rng(1))
    new,_ = sample_gaussian(cov,np.random.default_rng(1))
    exact = bool(np.allclose(new, old*np.sqrt(2048),atol=1e-13,rtol=1e-13))
    pair = generate_pair(1024,.5,.1,90001)
    mean_invariant = bool(np.allclose(standardize(pair['corrected']),standardize(pair['canonical']),atol=1e-13))
    zero = generate_pair(1024,.5,0.,90001)
    zero_equal = bool(np.allclose(standardize(zero['legacy']),standardize(zero['corrected']),atol=1e-13))
    pd.DataFrame(rows).to_csv(out/'covariance_acceptance.csv',index=False)
    pd.DataFrame(embeds).to_csv(out/'embedding_diagnostics.csv',index=False)
    status = dict(status='PASS' if all(gates) and exact and mean_invariant and zero_equal else 'FAIL',
                  covariance_gates_passed=sum(gates),covariance_gates_total=len(gates),
                  exact_sqrt_m_correction=exact,mean_convention_standardization_invariant=mean_invariant,
                  lambda_zero_standardized_invariant=zero_equal,num_samples=args.num_samples,
                  legacy_source_hash=before,legacy_unchanged=before==hashlib.sha256(src.read_bytes()).hexdigest())
    (out/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(out, json.dumps(status),flush=True)
    if status['status']!='PASS': raise SystemExit(2)

if __name__=='__main__': main()
