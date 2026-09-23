"""Frozen checkpoint attribution and inference-only correction-scale ablation."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import argparse
import time
import numpy as np
import pandas as pd
import torch
from revision.analysis_tools import Q,FrozenCalibrator,projection_metrics,dump_json,hashfile
from mrw_inverse.models.spectral_representation_model import CMINSRv3Model
from mrw_inverse.models.zeta_aligned_encoder import CMINSRZetaAlignedModel


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--T-values',type=int,nargs='+',default=[512,1024,2048])
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--quick',action='store_true')
    args=ap.parse_args();torch.set_num_threads(1)
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    paths={'v3':'cmin_sr_v3_synthetic.pt','aligned':'cmin_sr_zeta_aligned.pt',
           'curvature_preserved':'cmin_sr_zeta_curvature_preserved.pt'}
    cal=FrozenCalibrator();meta={};models={}
    for name,file in paths.items():
        p=ROOT/'checkpoints/cmin'/file;state=torch.load(p,map_location='cpu')
        m=CMINSRv3Model() if name=='v3' else CMINSRZetaAlignedModel()
        m.load_state_dict(state['model_state_dict'],strict=True);m.to(args.device).eval();models[name]=m
        meta[name]=dict(path=str(p.relative_to(ROOT)),sha256=hashfile(p),config=state.get('config'),strict=True)
    dump_json(out/'manifest.json',dict(checkpoints=meta,calibrator=cal.manifest,
        coefficients=[0.,.125,.25,.5],coefficient_scope='curvature_preserved inference perturbation only; NOT matched retraining',
        data_scope='Legacy and corrected samples paired; corrected-data performance is distribution shift, not retrained accuracy',
        device=args.device,input_dir=str(args.input_dir),T_values=args.T_values))
    rows=[];t0=time.time()
    for T in args.T_values:
        data=np.load(args.input_dir/f'raw_T{T}.npz')['x']
        manifest=pd.read_csv(args.input_dir/f'raw_T{T}_manifest.csv')
        if args.quick:data=data[:16];manifest=manifest.iloc[:16]
        for name,model in models.items():
            for start in range(0,len(data),32):
                x=torch.tensor(data[start:start+32],device=args.device)
                with torch.no_grad():
                    pred=model(x)
                    prior=model.structure_branch(x)['empirical_zeta_estimate'].cpu().numpy()
                    ze=pred['zeta_emp'].cpu().numpy()
                    native={k:pred[k].cpu().numpy().ravel() for k in ['p_scaling','p_curved','p_mrw','p_mono','tail_instability']}
                    # Keep old MAE feature path separately to quantify preprocessing mismatch.
                    oldcal=cal.model(pred['zeta_emp'].cpu(),pred['zeta_mono'].cpu(),pred['zeta_mrw'].cpu(),
                        pred['mono_residual_norm'].cpu(),pred['residual_norm'].cpu(),
                        pred['mrw_vs_mono_gain'].cpu(),pred['normalized_mrw_gain'].cpu(),
                        pred['curvature_score'].cpu(),pred['linearity_score'].cpu(),
                        pred.get('boundary_score',pred['boundary_mrw_score']).cpu(),pred['tail_instability'].cpu())
                coefs=[0.,.125,.25,.5] if name=='curvature_preserved' else [.25]
                for j in range(len(x)):
                    base=manifest.iloc[start+j].to_dict();h=float(base['H']);lam=float(base['lambda2_true'])
                    target=Q*h-.5*lam*Q*(Q-2) if base['process']=='MRW' else Q*h
                    for c in coefs:
                        z=prior[j]+(c/.25)*(ze[j]-prior[j])
                        row={**base,'method':name,'coefficient':c,**projection_metrics(z,target,h,lam),**cal.scores(z)}
                        if c==.25:
                            row.update({f'native_{k}':float(v[j]) for k,v in native.items()})
                            row['legacy_feature_p_mrw_cal']=float(oldcal['p_mrw'][j].item())
                        rows.append(row)
            pd.DataFrame(rows).to_csv(out/'frozen_encoder_results.csv',index=False)
            print(T,name,'samples',len(data),'seconds',round(time.time()-t0,1),flush=True)
    unchanged=all(hashfile(ROOT/v['path'])==v['sha256'] for v in meta.values())
    dump_json(out/'status.json',dict(status='PASS' if unchanged else 'FAIL',rows=len(rows),
                                   checkpoints_unchanged=unchanged,seconds=time.time()-t0))

if __name__=='__main__':main()
