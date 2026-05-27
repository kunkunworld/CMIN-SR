from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mrw_inverse.models import MonofractalProjection, MRWProjection, compute_curvature_diagnostics, estimate_robust_zeta
from .classical_multifractal_estimators import estimate_classical_zeta


ESTIMATORS = (
    "structure_ols",
    "structure_trimmed",
    "structure_bootstrap",
    "structure_smoothed",
    "structure_aggregated_ols",
    "mfdfa",
    "mfdfa_quadratic",
    "wavelet_leader_haar",
    "wtmm_haar",
)
ROBUST_STRUCTURE_ESTIMATORS = ("structure_ols", "structure_trimmed", "structure_bootstrap", "structure_smoothed")
CLASSICAL_ESTIMATORS = ("structure_aggregated_ols", "mfdfa", "mfdfa_quadratic", "wavelet_leader_haar", "wtmm_haar")


@dataclass
class CurvatureEstimate:
    estimator: str
    zeta_est: np.ndarray
    zeta_std: np.ndarray
    H_proj: float
    lambda2_proj: float
    H_mono: float
    mrw_residual_norm: float
    mono_residual_norm: float
    mrw_vs_mono_gain: float
    curvature_score: float
    second_diff_norm: float
    third_diff_norm: float
    scaling_fit_quality: float
    high_q_instability: float
    warning_flags: float


def _smooth_zeta(z: torch.Tensor) -> torch.Tensor:
    if z.shape[1] < 3:
        return z
    return torch.cat([z[:, :1], 0.25 * z[:, :-2] + 0.5 * z[:, 1:-1] + 0.25 * z[:, 2:], z[:, -1:]], dim=1)


def _d2_np(z: np.ndarray) -> float:
    return float(np.mean(np.abs(z[:-2] - 2.0 * z[1:-1] + z[2:]))) if z.size >= 3 else 0.0


def _d3_np(z: np.ndarray) -> float:
    return float(np.mean(np.abs(z[:-3] - 3.0 * z[1:-2] + 3.0 * z[2:-1] - z[3:]))) if z.size >= 4 else 0.0


def _project(zeta: torch.Tensor, q_grid: tuple[float, ...]) -> dict[str, float | np.ndarray]:
    mrw = MRWProjection(q_grid=q_grid)(zeta)
    mono = MonofractalProjection(q_grid=q_grid)(zeta)
    q = torch.tensor(q_grid, dtype=zeta.dtype, device=zeta.device).view(1, -1)
    diag = compute_curvature_diagnostics(
        q_grid=q,
        zeta_emp=zeta,
        zeta_mono=mono.zeta_mono,
        zeta_mrw=mrw.zeta_mrw_proj,
        residual_norm=mrw.residual_norm,
        mono_residual_norm=mono.mono_residual_norm,
        lambda2_proj=mrw.lambda2_proj,
    )
    return {
        "H_proj": float(mrw.H_proj.item()),
        "lambda2_proj": float(mrw.lambda2_proj.item()),
        "H_mono": float(mono.H_mono.item()),
        "mrw_residual_norm": float(mrw.residual_norm.item()),
        "mono_residual_norm": float(mono.mono_residual_norm.item()),
        "mrw_vs_mono_gain": float(diag.mrw_vs_mono_gain.item()),
        "curvature_score": float(diag.curvature_score.item()),
        "scaling_fit_quality": float(mrw.fit_quality.item()),
    }


def estimate_curvature_identifiability(
    x: np.ndarray,
    q_grid: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    scales: tuple[int, ...] = (2, 4, 8, 16, 32, 64),
    estimators: tuple[str, ...] = ESTIMATORS,
) -> list[CurvatureEstimate]:
    xt = torch.tensor(x[None, :], dtype=torch.float32)
    need_robust = any(name in ROBUST_STRUCTURE_ESTIMATORS for name in estimators)
    base = estimate_robust_zeta(xt, q_grid=q_grid, scales=scales, n_bootstrap=12, smooth=False) if need_robust else None
    zetas = {}
    if base is not None:
        zetas = {
            "structure_ols": base.zeta_ols,
            "structure_trimmed": base.zeta_robust,
            "structure_bootstrap": base.zeta_robust,
            "structure_smoothed": _smooth_zeta(base.zeta_robust),
        }
    out: list[CurvatureEstimate] = []
    for name in estimators:
        if name in CLASSICAL_ESTIMATORS:
            classical = estimate_classical_zeta(x, q_grid=q_grid, scales=scales, estimator=name)
            z_np = np.nan_to_num(classical.zeta.astype(np.float32))
            z = torch.tensor(z_np[None, :], dtype=torch.float32)
            proj = _project(z, q_grid=q_grid)
            out.append(
                CurvatureEstimate(
                    estimator=name,
                    zeta_est=z_np,
                    zeta_std=np.nan_to_num(classical.zeta_std.astype(np.float32)),
                    H_proj=float(proj["H_proj"]),
                    lambda2_proj=float(proj["lambda2_proj"]),
                    H_mono=float(proj["H_mono"]),
                    mrw_residual_norm=float(proj["mrw_residual_norm"]),
                    mono_residual_norm=float(proj["mono_residual_norm"]),
                    mrw_vs_mono_gain=float(proj["mrw_vs_mono_gain"]),
                    curvature_score=float(proj["curvature_score"]),
                    second_diff_norm=_d2_np(z_np),
                    third_diff_norm=_d3_np(z_np),
                    scaling_fit_quality=float(classical.fit_quality),
                    high_q_instability=float(classical.high_q_instability),
                    warning_flags=float(classical.warning_flags),
                )
            )
            continue
        if name not in zetas or base is None:
            continue
        z = torch.nan_to_num(zetas[name])
        proj = _project(z, q_grid=q_grid)
        zn = z.detach().cpu().numpy().reshape(-1)
        std = base.zeta_bootstrap_std.detach().cpu().numpy().reshape(-1)
        out.append(
            CurvatureEstimate(
                estimator=name,
                zeta_est=zn,
                zeta_std=std if name in {"structure_bootstrap", "structure_smoothed"} else np.zeros_like(zn),
                H_proj=float(proj["H_proj"]),
                lambda2_proj=float(proj["lambda2_proj"]),
                H_mono=float(proj["H_mono"]),
                mrw_residual_norm=float(proj["mrw_residual_norm"]),
                mono_residual_norm=float(proj["mono_residual_norm"]),
                mrw_vs_mono_gain=float(proj["mrw_vs_mono_gain"]),
                curvature_score=float(proj["curvature_score"]),
                second_diff_norm=_d2_np(zn),
                third_diff_norm=_d3_np(zn),
                scaling_fit_quality=float(base.scaling_fit_r2.item()),
                high_q_instability=float(base.high_q_instability.item()),
                warning_flags=float(base.warning_flags.item()),
            )
        )
    return out


def estimate_many(samples: list[np.ndarray], **kwargs) -> list[list[CurvatureEstimate]]:
    return [estimate_curvature_identifiability(x, **kwargs) for x in samples]
