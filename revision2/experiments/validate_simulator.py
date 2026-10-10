"""Covariance gates, independent samplers, and long-series spectrum validation."""
from common import *
import argparse
import importlib
import time
import types
import warnings
from scipy.linalg import toeplitz, cholesky
from revision.generation import generate_pair, sample_gaussian, standardize, embedding
from revision.retraining import sample_fgn_full_embedding
from mrw_dl.generation import build_log_covariance, build_fgn_covariance


def reference():
    path = HERE / 'vendor' / 'pymultifracs'
    metadata = json.loads((path / 'provenance.json').read_text())
    for f in metadata['files']:
        assert sha(path / f['local_name']) == f['sha256']
    pkg = types.ModuleType('r2_independent_simulator'); pkg.__path__ = [str(path)]
    sys.modules[pkg.__name__] = pkg
    return importlib.import_module(pkg.__name__ + '.mrw'), importlib.import_module(pkg.__name__ + '.pzutils')


def covariance_check(out, public, quick):
    n = 1024; R = 64 if quick else 512
    lags = [0, 1, 2, 8, 32, 128, 255, 256, 512]
    cases = [(f'omega_lam{v}', build_log_covariance(n, v, 256)) for v in [.01, .1, .2]]
    cases += [(f'fgn_H{h}', build_fgn_covariance(n, h)) for h in [.2, .5, .8]]
    rows = []; diagnostics = []
    for ci_, (name, cov) in enumerate(cases):
        for method in ['corrected_fft', 'independent_public_fft']:
            vals = []
            for k in range(R):
                seed = int(np.random.SeedSequence([202610071, ci_, k, method == 'corrected_fft']).generate_state(1)[0])
                rng = np.random.default_rng(seed)
                if method == 'corrected_fft':
                    if name.startswith('fgn'):
                        x = sample_fgn_full_embedding(n, float(name.split('H')[1]), rng)
                    else:
                        x, meta = sample_gaussian(cov, rng)
                        if k == 0: diagnostics.append(dict(case=name, **meta))
                else:
                    z = rng.normal(size=(2*n-2, 1)) + 1j*rng.normal(size=(2*n-2, 1))
                    with warnings.catch_warnings():
                        warnings.simplefilter('error')
                        x = public.gaussian_cme(cov, n, 1, z).ravel()
                vals.append([np.mean(x*x) if lag == 0 else np.mean(x[:-lag]*x[lag:]) for lag in lags])
            vals = np.asarray(vals)
            for j, lag in enumerate(lags):
                mean = vals[:, j].mean(); se = vals[:, j].std(ddof=1)/np.sqrt(R)
                tolerance = max(6*se, .025*cov[0])
                rows.append(dict(case=name, method=method, lag=lag, target=cov[lag],
                    observed=mean, se=se, ci_low=mean-1.96*se, ci_high=mean+1.96*se,
                    relative_to_variance=(mean-cov[lag])/cov[0], gate_tolerance=tolerance,
                    passed=bool(abs(mean-cov[lag]) <= tolerance), independent_N=R))
        print('covariance', name, 'done', flush=True)
    pd.DataFrame(rows).to_csv(out/'covariance_validation.csv', index=False)
    dump(out/'embedding_diagnostics.json', diagnostics)
    # Independent dense reference, with lower-triangular orientation explicit.
    cov = build_log_covariance(128, .1, 64); C = toeplitz(cov)
    lower = cholesky(C, lower=True)
    factor_error = float(np.max(np.abs(lower @ lower.T-C)))
    R = 1024 if quick else 8192
    rng = np.random.default_rng(202610072)
    chol = lower @ rng.normal(size=(128, R))
    fft = np.stack([sample_gaussian(cov, rng)[0] for _ in range(R)], axis=1)
    d = dict(cholesky_factor_max_error=factor_error,
        cholesky_empirical_covariance_rmse=float(np.sqrt(np.mean((chol@chol.T/R-C)**2))),
        corrected_fft_empirical_covariance_rmse=float(np.sqrt(np.mean((fft@fft.T/R-C)**2))),
        empirical_cross_covariance_rmse=float(np.sqrt(np.mean((chol@chol.T/R-fft@fft.T/R)**2))),
        independent_N=R, within_roundoff_factorization=bool(factor_error < 1e-12))
    dump(out/'independent_cholesky.json', d)
    # Exact distribution invariants, unaffected by empirical fitting choices.
    p = generate_pair(1024, .5, .1, 202610073)
    invariant = bool(np.allclose(standardize(p['corrected']),standardize(p['canonical']),atol=1e-13))
    dump(out/'mean_convention.json', dict(standardized_equivalent=invariant,
         canonical_mean='-Var(omega)', R1_mean='-Var(omega)/2',
         implication='Fixed discretization: constant amplitude only; not a claim of equal unstandardized variance.'))
    assert all(row['passed'] for row in rows), 'Covariance gate failed; stop before spectrum comparisons'
    assert invariant and factor_error < 1e-12
    return rows


def long_validation(out, public, quick):
    protocol = json.loads((HERE/'protocol.json').read_text())['long_validation']
    lengths = [16384] if quick else protocol['lengths']
    levels = [0., .1] if quick else protocol['lambda2']
    seeds = protocol['seeds'][:1] if quick else protocol['seeds']
    reps = 2 if quick else protocol['replicates_per_seed']
    windows = protocol['scale_windows']; sc = sorted(set(sum(windows.values(), [])))
    curves = []; mrows = []; covrows = []; t0 = time.time()
    for method in ['corrected_fft', 'independent_public_fft']:
        for n in lengths:
            if method == 'independent_public_fft' and n != max(lengths): continue
            for li, lam in enumerate(levels):
                allmom = []
                count = reps if method == 'corrected_fft' else max(2, reps//2)
                for seed in seeds:
                    for k in range(count):
                        sd = int(np.random.SeedSequence([seed,n,li,k,method=='corrected_fft']).generate_state(1)[0])
                        if method == 'corrected_fft':
                            p = generate_pair(n, .5, lam, sd, integral_scale=4096)
                            dx = p['canonical']
                            om = p['omega_centered']
                            for lag in [0,1,4,16,64,256,1024,4096]:
                                covrows.append(dict(method=method,T=n,lambda2=lam,seed=seed,replicate=k,lag=lag,
                                    target=p['omega_covariance'][lag],
                                    empirical=np.mean(om*om) if lag==0 else np.mean(om[:-lag]*om[lag:])))
                        else:
                            np.random.seed(sd)
                            with warnings.catch_warnings():
                                warnings.simplefilter('error')
                                path = public.mrw(n, .5, np.sqrt(lam), L=4096, method='cme')
                            dx = np.diff(path, prepend=0.)
                        # Per-realization normalization cancels in slopes. Pooled moments
                        # are not used, because public mrw applies sample mean removal.
                        mom = moments(standardize(dx), sc)
                        allmom.append((seed,k,mom))
                        for si,s in enumerate(sc):
                            for qi,q in enumerate(Q):
                                mrows.append(dict(method=method,T=n,H=.5,lambda2=lam,seed=seed,
                                    replicate=k,scale=s,q=q,moment=mom[si,qi]))
                        for name, ss in windows.items():
                            z = slopes(mom[[sc.index(s) for s in ss]], ss)
                            target = .5*Q - .5*lam*Q*(Q-2)
                            for qi,q in enumerate(Q):
                                curves.append(dict(method=method,T=n,H=.5,L=4096,lambda2=lam,
                                    seed=seed,replicate=k,scale_window=name,q=q,zeta=z[qi],target=target[qi],
                                    bias=z[qi]-target[qi]))
                pd.DataFrame(curves).to_csv(out/'long_zeta_replicates.csv',index=False)
                print('long',method,n,lam,'N=',len(allmom),'seconds',round(time.time()-t0),flush=True)
    pd.DataFrame(mrows).to_csv(out/'long_structure_moments.csv',index=False)
    pd.DataFrame(covrows).to_csv(out/'long_covariance_replicates.csv',index=False)
    df = pd.DataFrame(curves); summary=[]
    for keys,g in df.groupby(['method','T','H','L','lambda2','scale_window','q'],sort=True):
        vals=g.zeta.to_numpy(); groups=g.reset_index(drop=True).groupby('seed').indices
        rng=np.random.default_rng(202610079)
        ix=np.concatenate([rng.choice(ids,(B,len(ids)),replace=True) for ids in groups.values()],axis=1)
        lo,hi=ci(vals[ix].mean(axis=1))
        summary.append(dict(zip(['method','T','H','L','lambda2','scale_window','q'],keys),
             mean_zeta=vals.mean(),ci_low=lo,ci_high=hi,target=g.target.iloc[0],
             bias=vals.mean()-g.target.iloc[0],independent_N=len(vals)))
    pd.DataFrame(summary).to_csv(out/'long_zeta_summary.csv',index=False)
    # Pure fGn: validate multiple H without assigning invalid fractional-MRW labels.
    frows=[]; n=max(lengths)
    for hi,h in enumerate([.2,.5,.8]):
        for seed in seeds:
            for k in range(reps):
                sd=int(np.random.SeedSequence([seed,n,hi,k,8]).generate_state(1)[0])
                x=sample_fgn_full_embedding(n,h,np.random.default_rng(sd))
                mom=moments(standardize(x),windows['interior'])
                z=slopes(mom,windows['interior'])
                for qi,q in enumerate(Q): frows.append(dict(T=n,H=h,seed=seed,replicate=k,q=q,zeta=z[qi],target=q*h,bias=z[qi]-q*h))
        print('fGn long',h,'done',flush=True)
    pd.DataFrame(frows).to_csv(out/'long_fgn_replicates.csv',index=False)
    return df,pd.DataFrame(summary)


def plot(out,cov,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    d=pd.DataFrame(cov)
    for method,mark in [('corrected_fft','o'),('independent_public_fft','x')]:
        g=d[(d['case']=='omega_lam0.1')&(d.method==method)]
        axes[0].errorbar(g.lag+1,g.observed,yerr=1.96*g.se,fmt=mark,ms=4,label=method)
    axes[0].plot(g.lag+1,g.target,'k--',label='Specified covariance')
    axes[0].set(xscale='log',xlabel='Lag + 1',ylabel='Log-volatility covariance')
    axes[0].legend(fontsize=7)
    d=summary[(summary['T']==summary['T'].max())&(summary.scale_window=='interior')]
    for lam in sorted(d.lambda2.unique()):
        g=d[(d.lambda2==lam)&(d.method=='corrected_fft')]
        axes[1].plot(g.q,g.mean_zeta-g.target,'o-',label=f'lambda2={lam:g}')
        axes[1].fill_between(g.q,g.ci_low-g.target,g.ci_high-g.target,alpha=.12)
    axes[1].axhline(0,color='black',ls='--',lw=.8)
    axes[1].set(xlabel='Moment order q',ylabel='Mean exponent minus reference')
    axes[1].legend(fontsize=7)
    savefig(fig,out,'simulator_validation');plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true');ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output
    before=start(out,dict(experiment='E1',quick=args.quick,reference='pinned pymultifracs CME',
        covariance_gate='max(6 independent-realization SE, 2.5% marginal variance); numerical gate, not a fit claim'))
    t0=time.time()
    try:
        mrw,gaussian=reference()
        cov=covariance_check(out,gaussian,args.quick)
        df,summary=long_validation(out,mrw,args.quick)
        plot(out,cov,summary)
        finish(out,before,seconds=time.time()-t0,covariance_gates_passed=True,
               scientific_scope='MRW H=.5 plus pure fGn controls; finite-grid spectrum discrepancy retained')
    except Exception:
        import traceback
        dump(out/'status.json',dict(status='FAILED',traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
