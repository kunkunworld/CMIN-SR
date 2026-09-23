"""Evaluate the unchanged historical loss at exact analytic spectra."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import argparse
import numpy as np
import pandas as pd
import torch
from mrw_inverse.models import MRWProjection,MonofractalProjection
from mrw_inverse.losses import curvature_preserving_zeta_loss
from mrw_inverse.data import PROCESS_NAME_TO_CODE
from revision.analysis_tools import Q,dump_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False);q=torch.tensor(Q,dtype=torch.float64)[None,:];rows=[]
    for lam in [0.,.01,.03,.06,.1,.15,.2]:
        z=.5*q-.5*lam*q*(q-2);mrw=MRWProjection()(z);mono=MonofractalProjection()(z)
        alpha=torch.gradient(z,spacing=(q[0],),dim=1)[0];f=1+q*alpha-z
        tail=((f.max(dim=1,keepdim=True).values-f.min(dim=1,keepdim=True).values-.3)/1.5).clamp(0,1)
        pred=dict(zeta_emp=z,lambda2_proj=mrw.lambda2_proj,H_proj=mrw.H_proj,
                  residual_norm=mrw.residual_norm,mono_residual_norm=mono.mono_residual_norm,
                  tail_instability=tail,zeta_uncertainty=(z-mrw.zeta_mrw_proj).abs().mean(dim=1,keepdim=True))
        loss=curvature_preserving_zeta_loss(pred,torch.tensor([PROCESS_NAME_TO_CODE['MRW']]),z,
            torch.ones_like(z),torch.ones_like(z),torch.tensor([[.5]]),torch.tensor([[lam]],dtype=torch.float64),torch.zeros_like(tail))
        rows.append(dict(lambda2=lam,discrete_f_width=float(torch.ptp(f)) if hasattr(torch,'ptp') else float(f.max()-f.min()),
                         tail_indicator=float(tail),**{k:float(v) for k,v in vars(loss).items()}))
    df=pd.DataFrame(rows);df.to_csv(out/'loss_at_analytic_truth.csv',index=False)
    (out/'interpretation.md').write_text('# Historical loss semantics\n\n'
        'The tail indicator is computed from finite-q Legendre-spectrum variation, not from\n'
        'an independent tail/moment-stability measurement. It can be positive for an exact\n'
        'clean curved MRW spectrum. A zero tail target therefore adds a regularization penalty\n'
        'at some analytic truths. This alone does not prove the full loss optimum is biased,\n'
        'because the curvature/projection L1 terms also act. Do not claim this penalty measures\n'
        'true heavy-tail instability. The predeclared zeta-only ablation tests the aggregate\n'
        'effect of the auxiliary terms; it cannot isolate the tail term causally.\n\n'+df.to_csv(index=False),encoding='utf-8')
    dump_json(out/'status.json',{'status':'PASS','scope':'Exact-spectrum semantic check; existing loss unchanged'})


if __name__=='__main__':main()
