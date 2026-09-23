"""Revision-only metrics with explicit coordinate and feature conventions."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from mrw_inverse.data.analytic_spectrum_dataset import _mono_fit, _mrw_fit, _diagnostics
from mrw_inverse.models.spectral_geometry_calibrator import SpectralGeometryCalibrator

ROOT = Path(__file__).resolve().parents[1]
Q = np.array([.5,1.,1.5,2.,2.5,3.])
SCALES = (2,4,8,16,32,64)


def hashfile(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def protected_hashes():
    paths = set()
    for folder, patterns in [('src',['*.py']),('checkpoints/cmin',['*.pt'])]:
        for pattern in patterns:
            paths.update((ROOT/folder).rglob(pattern))
    return {str(p.relative_to(ROOT)):hashfile(p) for p in sorted(paths)}


def structure_zeta(x, q=Q, scales=SCALES, legacy_input=False):
    path = np.asarray(x) if legacy_input else np.cumsum(x)
    sq = np.array([np.mean(np.maximum(np.abs(path[s:]-path[:-s]),1e-30)[:,None]**q,axis=0) for s in scales])
    u = np.log(scales); u = u-u.mean()
    y = np.log(sq)
    z = (u[:,None]*(y-y.mean(axis=0))).sum(axis=0)/(u*u).sum()
    return z


def projection_metrics(z, target=None, h=None, lam=None, q=Q):
    z = np.asarray(z,dtype=float)
    if not np.isfinite(z).all(): raise ValueError('Nonfinite spectrum')
    X = np.c_[q, -.5*q*(q-2)]
    hs,ls = np.linalg.lstsq(X,z,rcond=None)[0]
    # Historical MRWProjection decoder: H clipping and positive numerical floor.
    hp = np.clip(hs,1e-4,1-1e-4); lp = max(ls,1e-6)
    hm = np.clip(q@z/(q@q),0,1)
    r = float(np.mean(np.abs(z-X@np.array([hp,lp]))))
    rm = float(np.mean(np.abs(z-q*hm)))
    d = dict(H_signed=hs,lambda2_signed=ls,lambda2_nonnegative=max(ls,0.),
             H_proj=hp,lambda2_proj=lp,H_mono=hm,mrw_residual=r,mono_residual=rm,
             gain=rm-r,negative_signed=bool(ls<0),second_diff_norm=float(np.mean(np.abs(np.diff(z,n=2)))),
             third_diff_norm=float(np.mean(np.abs(np.diff(z,n=3)))))
    d.update({f'zeta_{i}':float(v) for i,v in enumerate(z)})
    if target is not None:
        d.update(zeta_mae=float(np.mean(np.abs(z-target))),
                 d2_error=float(np.mean(np.abs(np.diff(z,n=2)-np.diff(target,n=2)))))
    else: d.update(zeta_mae=np.nan,d2_error=np.nan)
    d['H_mae'] = abs(hp-h) if h is not None else np.nan
    d['lambda2_mae'] = abs(lp-lam) if lam is not None else np.nan
    return d


class FrozenCalibrator:
    def __init__(self, checkpoint=None):
        path = Path(checkpoint or os.environ.get('CMINSR_GEOMETRY_CHECKPOINT') or
                    ROOT/'checkpoints/cmin/spectral_geometry_calibrator.pt').resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f'Geometry checkpoint not found: {path}. Run '
                'experiments/train_spectral_geometry_calibrator.py first, or set '
                'CMINSR_GEOMETRY_CHECKPOINT to a trusted existing checkpoint. '
                'No random-weight fallback is used.')
        state = torch.load(path,map_location='cpu')
        self.model = SpectralGeometryCalibrator().eval()
        self.model.load_state_dict(state['model_state_dict'],strict=True)
        checkpoint_location = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        self.manifest = dict(path=checkpoint_location,sha256=hashfile(path),strict_loading=True,
                             feature_protocol='analytic_training_RMSE_diagnostics',tail_input=0,
                             note='Geometry-only comparison: no process label or tail oracle supplied')

    def scores(self,z,q=Q):
        _,zm,rm = _mono_fit(q,z)
        _,lam,zr,rr = _mrw_fit(q,z)
        diag = _diagnostics(q,z,zm,zr,rm,rr,lam)
        t = lambda x:torch.tensor(np.asarray(x).reshape(1,-1),dtype=torch.float32)
        with torch.no_grad():
            out=self.model(t(z),t(zm),t(zr),t(rm),t(rr),t(diag['mrw_vs_mono_gain']),
                           t(diag['normalized_mrw_gain']),t(diag['curvature_score']),
                           t(diag['linearity_score']),t(diag['boundary_score']),t(0.))
        return {k:float(out[k].item()) for k in ['p_scaling','p_curved','p_mono','p_mrw','p_boundary']}


def safe_corr(x,y):
    x,y=np.asarray(x),np.asarray(y)
    if len(x)<3 or np.std(x)<1e-12 or np.std(y)<1e-12: return np.nan
    return float(np.corrcoef(x,y)[0,1])


def dump_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
