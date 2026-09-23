"""Corrected-data training helpers; no modification to historical architecture."""
from __future__ import annotations
import random
import numpy as np
import pandas as pd
import torch
from mrw_dl.generation import build_fgn_covariance
from mrw_inverse.models import CMINSRZetaAlignedModel
from mrw_inverse.data import PROCESS_NAME_TO_CODE
from mrw_inverse.losses import curvature_preserving_zeta_loss
from revision.generation import generate_pair, sample_gaussian, standardize
from revision.analysis_tools import Q, structure_zeta

CONFIGS = {'full_c0125':(.125,'full'), 'full_c025':(.25,'full'),
           'full_c05':(.5,'full'), 'zeta_only_c025':(.25,'zeta_only')}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def finite(name, value):
    ok = torch.isfinite(value).all().item() if isinstance(value, torch.Tensor) else np.isfinite(value).all()
    if not ok: raise FloatingPointError(f'Nonfinite {name}')


def make_model(seed, coefficient, device):
    set_seed(seed)
    model = CMINSRZetaAlignedModel().to(device)
    torch.nn.init.zeros_(model.zeta_head[-1].weight)
    torch.nn.init.zeros_(model.zeta_head[-1].bias)
    # Multiplication before the historical fixed .25 implements c without a new layer.
    model.zeta_head.register_forward_hook(lambda module, args, output: output * (coefficient / .25))
    for name, p in model.named_parameters():
        if name.startswith(('p_', 'stability_head')): p.requires_grad_(False)
    return model


def simulate(T, process, h, lam, sd):
    if process == 'MRW':
        if h != .5: raise ValueError('Unvalidated MRW target outside H=.5')
        x = generate_pair(T,h,lam,sd)['corrected']
    elif process == 'fGn':
        x = sample_fgn_full_embedding(T,h,np.random.default_rng(sd))
    elif process == 'Gaussian': x = np.random.default_rng(sd).normal(size=T)
    else: raise ValueError(process)
    return standardize(x).astype(np.float32)


def sample_fgn_full_embedding(n, h, rng):
    """Include covariance at lag n rather than zeroing the embedding midpoint.

    The first n-by-n covariance block is the requested Toeplitz covariance.
    A non-PSD embedding still raises; no eigenvalue clipping beyond roundoff.
    """
    cov=build_fgn_covariance(n+1,h)
    column=np.r_[cov,cov[1:n][::-1]]
    eig=np.fft.fft(column).real
    tol=1e-10*max(1.,float(np.abs(eig).max()))
    if eig.min() < -tol: raise ValueError('Non-PSD full fGn embedding')
    w=rng.normal(size=2*n)+1j*rng.normal(size=2*n)
    return (np.sqrt(2*n)*np.fft.ifft(np.sqrt(np.maximum(eig,0))*w).real)[:n]


def make_training(n, T, seed, split):
    xs=[]; targets=[]; codes=[]; rows=[]
    for i in range(n):
        sd=int(np.random.SeedSequence([602619,split,seed,T,i,0]).generate_state(1)[0])
        parameter_seed=int(np.random.SeedSequence([602619,split,seed,T,i,1]).generate_state(1)[0])
        rng=np.random.default_rng(parameter_seed); bucket=i%10
        if bucket<6:
            lo,hi=[(0,.03),(.03,.08),(.08,.2)][bucket//2]
            h=.5;lam=float(rng.uniform(lo,hi));process='MRW'
            code=PROCESS_NAME_TO_CODE['Low-lambda2 MRW' if lam<.03 else 'MRW']
        else:
            h=float(rng.uniform(.1,.9)) if bucket<8 else .5;lam=0.
            process='Gaussian' if bucket==9 else 'fGn'
            code=PROCESS_NAME_TO_CODE['iid Gaussian' if process=='Gaussian' else 'fGn']
        xs.append(simulate(T,process,h,lam,sd)); targets.append(Q*h-.5*lam*Q*(Q-2));codes.append(code)
        rows.append(dict(sample_id=f'{split}_{seed}_{T}_{i}',data_seed=sd,parameter_seed=parameter_seed,seed=seed,split=split,
                         T=T,process=process,H=h,lambda2_true=lam))
    meta=pd.DataFrame(rows)
    data={'x':torch.tensor(np.stack(xs)), 'target':torch.tensor(np.stack(targets),dtype=torch.float32),
          'code':torch.tensor(codes), 'h':torch.tensor(meta.H.to_numpy(),dtype=torch.float32)[:,None],
          'lam':torch.tensor(meta.lambda2_true.to_numpy(),dtype=torch.float32)[:,None]}
    return data,meta


def loss_for(out,batch,mode):
    for k in ['zeta_emp','lambda2_proj','H_proj','residual_norm','mono_residual_norm',
              'tail_instability','zeta_uncertainty']: finite(k,out[k])
    target=batch['target'];finite('target',target)
    loss=curvature_preserving_zeta_loss(out,batch['code'],target,torch.ones_like(target),
        torch.ones_like(target),batch['h'],batch['lam'],torch.zeros_like(batch['lam']))
    value=loss.total if mode=='full' else 2*loss.l_zeta
    finite('loss',value)
    return value


def prior_and_ols(model, x, device, bs=64):
    model.eval(); pieces=[]
    with torch.no_grad():
        for start in range(0,len(x),bs):
            pieces.append(model.structure_branch(x[start:start+bs].to(device))['empirical_zeta_estimate'].cpu().numpy())
    prior=np.concatenate(pieces)
    ols=np.stack([structure_zeta(y.numpy().astype(float)) for y in x])
    finite('prior',prior);finite('ols',ols)
    return prior,ols
