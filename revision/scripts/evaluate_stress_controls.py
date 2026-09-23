"""Frozen pipelines on existing stress families; no fabricated clean targets."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import argparse
import numpy as np
import pandas as pd
import torch
from mrw_inverse.data.process_generators import gen_iid_student_t,gen_garch,gen_regime_switching_gaussian
from mrw_inverse.models.zeta_aligned_encoder import CMINSRZetaAlignedModel
from mrw_inverse.models.spectral_representation_model import CMINSRv3Model
from revision.analysis_tools import (FrozenCalibrator,structure_zeta,projection_metrics,dump_json,hashfile)
from revision.generation import generate_pair,standardize


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--num-samples',type=int,default=50);args=ap.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    device='cuda' if torch.cuda.is_available() else 'cpu';cal=FrozenCalibrator();models={}
    for name,file in [('v3','cmin_sr_v3_synthetic.pt'),('curvature_preserved','cmin_sr_zeta_curvature_preserved.pt')]:
        model=CMINSRv3Model() if name=='v3' else CMINSRZetaAlignedModel()
        p=ROOT/'checkpoints/cmin'/file;model.load_state_dict(torch.load(p,map_location='cpu')['model_state_dict'],strict=True)
        models[name]=model.to(device).eval()
    dump_json(out/'manifest.json',dict(n_per_seed_T_family=args.num_samples,seeds=list(range(31001,31006)),
        T_values=[512,1024,2048],calibrator=cal.manifest,
        regime_correction='Existing Regime-switching Gaussian actually has iid state labels, not Markov regimes; renamed iid_Gaussian_mixture',
        targets='No theoretical zeta/parameter labels for stress processes or shuffled MRW'))
    rows=[]
    for T in [512,1024,2048]:
        inputs=[];metas=[]
        for seed in range(31001,31006):
            for fi,process in enumerate(['Student_t','GARCH','iid_Gaussian_mixture','shuffled_MRW']):
                for i in range(args.num_samples):
                    sd=int(np.random.SeedSequence([seed,T,fi,i,7]).generate_state(1)[0]);rng=np.random.default_rng(sd)
                    if process=='Student_t':x=gen_iid_student_t(T,rng)
                    elif process=='GARCH':x=gen_garch(T,rng)
                    elif process=='iid_Gaussian_mixture':x=gen_regime_switching_gaussian(T,rng)
                    else:
                        p=generate_pair(T,.5,.1,sd);x=standardize(p['corrected']);rng.shuffle(x)
                    inputs.append(x.astype(np.float32));metas.append(dict(T=T,seed=seed,data_seed=sd,
                        sample_id=f'{seed}_{T}_{process}_{i}',process=process))
        xs=np.stack(inputs)
        np.savez_compressed(out/f'stress_inputs_T{T}.npz',x=xs)
        pd.DataFrame(metas).to_csv(out/f'stress_manifest_T{T}.csv',index=False)
        for i,x in enumerate(xs):
            z=structure_zeta(x.astype(float));rows.append({**metas[i],'pipeline':'structure_ols',
                **projection_metrics(z),**cal.scores(z)})
        for name,model in models.items():
            for start in range(0,len(xs),32):
                with torch.no_grad():pred=model(torch.tensor(xs[start:start+32],device=device))
                zetas=pred['zeta_emp'].cpu().numpy()
                for j,z in enumerate(zetas):
                    row={**metas[start+j],'pipeline':name,**projection_metrics(z),**cal.scores(z)}
                    row.update({f'native_{key}':float(pred[key][j].item()) for key in ['p_mrw','p_curved','p_scaling','tail_instability']})
                    rows.append(row)
        print('stress T=',T,flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'stress_sample_results.csv',index=False)
    metrics=['p_mrw','p_curved','p_mono','p_scaling','p_boundary','native_p_mrw','native_tail_instability']
    df.groupby(['T','process','pipeline','seed'])[metrics].mean().to_csv(out/'stress_by_seed.csv')
    df.groupby(['T','process','pipeline'])[metrics].agg(['mean','std']).to_csv(out/'stress_summary.csv')
    dump_json(out/'status.json',dict(status='PASS',rows=len(df),independent_sequences=len(df)//3))

if __name__=='__main__':main()
