"""Explicit input adaptation for the historical robust structure estimator.

This correction changes input preparation only. Trimming, scale-subset
uncertainty, smoothing and projection retain their historical definitions.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

from mrw_inverse.analysis.curvature_identifiability import (
    CurvatureEstimate,
    ROBUST_STRUCTURE_ESTIMATORS,
    _d2_np,
    _d3_np,
    _project,
    _smooth_zeta,
)
from mrw_inverse.models.robust_zeta_estimator import RobustZetaOutput, estimate_robust_zeta

PROTOCOL = "revision_explicit_input_contract_v1"
Q_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
SCALES = (2, 4, 8, 16, 32, 64)


def estimate_revision_robust_zeta(
    x: torch.Tensor,
    *,
    input_kind: Literal["increments", "path"],
    q_grid: tuple[float, ...] = Q_GRID,
    scales: tuple[int, ...] = SCALES,
    n_bootstrap: int = 12,
    smooth: bool = False,
) -> RobustZetaOutput:
    """Adapt increments once, or accept an existing path without integrating.

    Paths have the same N samples as cumsum(increments); no origin is prepended.
    Consequently the N-a differences match the existing empirical_spectrum.py.
    This intentionally does not change the historical endpoint convention.
    """
    if input_kind not in {"increments", "path"}:
        raise ValueError("input_kind must be explicitly 'increments' or 'path'")
    if not isinstance(x, torch.Tensor) or x.dtype not in (torch.float32, torch.float64):
        raise TypeError("x must be a float32 or float64 torch tensor")
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 10:
        raise ValueError("Expected nonempty [batch, time] input with at least 10 points")
    if not torch.isfinite(x).all():
        raise ValueError("Nonfinite input is rejected, not silently replaced")
    if len(q_grid) < 3 or any(not math.isfinite(q) or q <= 0 for q in q_grid):
        raise ValueError("At least three finite positive q values are required")
    if any(b <= a for a, b in zip(q_grid, q_grid[1:])):
        raise ValueError("q_grid must be strictly increasing")
    if len(scales) < 4 or any(type(s) is not int or s <= 0 for s in scales):
        raise ValueError("At least four positive integer scales are required")
    if any(b <= a for a, b in zip(scales, scales[1:])):
        raise ValueError("scales must be strictly increasing")
    if any(s >= x.shape[1] // 2 for s in scales):
        raise ValueError("Every requested scale must be below time_length//2; select valid scales explicitly")
    if type(n_bootstrap) is not int or n_bootstrap < 2:
        raise ValueError("At least two scale-subset replicates are required")
    path = torch.cumsum(x, dim=1) if input_kind == "increments" else x
    if not torch.isfinite(path).all():
        raise ValueError("Cumulative path overflowed")
    return estimate_robust_zeta(path, q_grid=q_grid, scales=scales,
                                n_bootstrap=n_bootstrap, smooth=smooth)


def estimate_revision_curvature(
    x: torch.Tensor,
    *,
    input_kind: Literal["increments", "path"],
    q_grid: tuple[float, ...] = Q_GRID,
    scales: tuple[int, ...] = SCALES,
    estimators: tuple[str, ...] = ROBUST_STRUCTURE_ESTIMATORS,
) -> list[CurvatureEstimate]:
    """Return the existing projection schema for one corrected input series.

    Classical methods must still receive increments via their original APIs;
    this adapter never passes a cumulative path to a mixed estimator list.
    """
    if not estimators or set(estimators) - set(ROBUST_STRUCTURE_ESTIMATORS):
        raise ValueError("This adapter supports only the four historical robust structure variants")
    if x.ndim not in (1, 2) or (x.ndim == 2 and x.shape[0] != 1):
        raise ValueError("Projection comparison requires exactly one series")
    base = estimate_revision_robust_zeta(x, input_kind=input_kind,
                                         q_grid=q_grid, scales=scales, smooth=False)
    zetas = {
        "structure_ols": base.zeta_ols,
        "structure_trimmed": base.zeta_robust,
        # Kept explicitly as a historical alias, not a second independent estimate.
        "structure_bootstrap": base.zeta_robust,
        "structure_smoothed": _smooth_zeta(base.zeta_robust),
    }
    result = []
    for name in estimators:
        z = zetas[name]
        proj = _project(z, q_grid)
        zn = z.detach().cpu().numpy().reshape(-1)
        std = base.zeta_bootstrap_std.detach().cpu().numpy().reshape(-1)
        result.append(CurvatureEstimate(
            estimator=name, zeta_est=zn,
            zeta_std=std if name in {"structure_bootstrap", "structure_smoothed"} else std * 0,
            H_proj=proj["H_proj"], lambda2_proj=proj["lambda2_proj"],
            H_mono=proj["H_mono"], mrw_residual_norm=proj["mrw_residual_norm"],
            mono_residual_norm=proj["mono_residual_norm"], mrw_vs_mono_gain=proj["mrw_vs_mono_gain"],
            curvature_score=proj["curvature_score"], second_diff_norm=_d2_np(zn),
            third_diff_norm=_d3_np(zn), scaling_fit_quality=float(base.scaling_fit_r2.item()),
            high_q_instability=float(base.high_q_instability.item()),
            warning_flags=float(base.warning_flags.item()),
        ))
    return result
