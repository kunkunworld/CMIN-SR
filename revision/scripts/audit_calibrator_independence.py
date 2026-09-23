"""R2-4: separate within-family sanity, oracle input, and unseen curve stress."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import argparse,json
import numpy as np
import pandas as pd
import torch
from mrw_inverse.data.analytic_spectrum_dataset import AnalyticSpectrumDatasetConfig,generate_analytic_spectrum_dataset
from revision.analysis_tools import FrozenCalibrator,Q,projection_metrics,dump_json,hashfile


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);cal=FrozenCalibrator()
    dump_json(out/'manifest.json',dict(seeds=[91001,91002,91003],n_per_training_family=100,
        n_per_unseen_shape=100,calibrator=cal.manifest,script_sha256=hashfile(Path(__file__)),
        thresholds=[.5,.8],scope='No classifier retraining; stress curves do not claim stochastic realizability',
        oracle_channel='Historical class-defined target_tail_instability; not observable at inference'))
    rows=[]
    keys=['zeta_input','zeta_mono','zeta_mrw','mono_residual_norm','mrw_residual_norm',
          'mrw_vs_mono_gain','normalized_mrw_gain','curvature_score','linearity_score','boundary_score']
    for seed in [91001,91002,91003]:
        ds=generate_analytic_spectrum_dataset(AnalyticSpectrumDatasetConfig(num_samples=800,seed=seed))
        parts=[torch.tensor(ds[k],dtype=torch.float32) for k in keys]
        tail=torch.tensor(ds['target_tail_instability'],dtype=torch.float32)
        with torch.no_grad():
            oracle=cal.model(*parts,tail);blind=cal.model(*parts,torch.zeros_like(tail))
        for i in range(800):
            row=dict(seed=seed,sample_id=f'{seed}_familiar_{i}',family=ds['spectrum_type'][i],
                     evaluation='same_training_family',tail_oracle=float(tail[i]),
                     lambda2_true=float(ds['lambda2_true'][i]),H=float(ds['H_true'][i]))
            for k in ['p_scaling','p_curved','p_mono','p_mrw','p_boundary']:
                row[k]=float(blind[k][i]);row[k+'_oracle_tail']=float(oracle[k][i])
            # Verify the harmonized raw adapter against the actual training features.
            ref=cal.scores(ds['zeta_input'][i],q=Q.astype(np.float32))
            for k in ref:np.testing.assert_allclose(ref[k],row[k],atol=2e-5,rtol=2e-5)
            rows.append(row)
        rng=np.random.default_rng(seed)
        for family in ['quartic_departure','localized_cubic_kink','qgrid_ripple','convex_invalid']:
            for i in range(100):
                h=rng.uniform(.2,.8);lam=rng.uniform(.06,.2);a=rng.uniform(.01,.08)
                base=Q*h-.5*lam*Q*(Q-2)
                if family=='quartic_departure':z=base-a*(Q**4-8*Q)/9
                elif family=='localized_cubic_kink':z=base-a*np.maximum(Q-1.5,0)**3
                elif family=='qgrid_ripple':z=base+a*np.sin(3.7*Q)
                else:z=Q*h+.5*lam*Q*(Q-2)
                d=projection_metrics(z)
                row=dict(seed=seed,sample_id=f'{seed}_{family}_{i}',family=family,
                    evaluation='unseen_algebraic_shape',H=h,lambda2_base=lam,perturbation=a,
                    tail_oracle=0.,**d,**cal.scores(z))
                rows.append(row)
        print(f'calibrator audit seed={seed} complete',flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'sample_results.csv',index=False)
    scores=['p_scaling','p_curved','p_mono','p_mrw','p_boundary']
    byseed=df.groupby(['evaluation','family','seed'])[scores+['p_mrw_oracle_tail']].mean().reset_index()
    byseed.to_csv(out/'by_seed.csv',index=False)
    summary=df.groupby(['evaluation','family'])[scores+['p_mrw_oracle_tail']].mean().reset_index()
    summary['fraction_pmrw_ge_08']=df.assign(accept=df.p_mrw>=.8).groupby(['evaluation','family']).accept.mean().values
    summary.to_csv(out/'summary.csv',index=False)
    (out/'summary.md').write_text('# Interpretation-layer independence audit\n\n'
        'Same-family holdouts test interpolation, not independent scientific validation.\n'
        'The historical tail input is generated from the class and is an oracle input.\n'
        'Unseen algebraic curves have no asserted stochastic ground truth; high compatibility\n'
        'cannot establish an MRW mechanism. Threshold fractions are not real-data error rates.\n\n'
        +summary.to_csv(index=False),encoding='utf-8')
    dump_json(out/'status.json',dict(status='PASS',rows=len(df),training_feature_adapter_verified=True,
        scope='Engineering audit; high compatibility on non-MRW shapes must not be hidden'))


if __name__=='__main__':main()
