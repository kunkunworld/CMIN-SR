from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from mrw_dl.baselines import (
    legendre_spectrum_from_zeta,
    structure_functions_from_increments,
    true_mrw_zeta,
)

from .models import CMINRegressor


DEFAULT_Q_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
DEFAULT_SCALES = (2, 4, 8, 16, 32, 64)
DEFAULT_LAGS = (1, 2, 4, 8, 16, 32, 64)


@dataclass
class InverseEstimate:
    mode: str
    model_name: str
    checkpoint_path: str
    pred_H: float
    pred_lambda2: float
    p_MRW: float
    residual_norm: float
    logvol_cov_slope: float
    empirical_zeta_curvature: float
    f_alpha_width: float
    lambda2_boundary_hit: bool
    zeta_fit_r2: float
    notes: str


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    coef = np.polyfit(x, y, deg=1)
    return float(coef[0]), float(coef[1])


def _fit_parabolic_mrw(q: np.ndarray, zeta_emp: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    design = np.column_stack([q, -0.5 * q * (q - 2.0)])
    coef, _, _, _ = np.linalg.lstsq(design, zeta_emp, rcond=None)
    h_hat = float(coef[0])
    lambda2_hat = float(max(coef[1], 0.0))
    zeta_fit = true_mrw_zeta(q, h=h_hat, lambda2=lambda2_hat)
    ss_res = float(np.sum((zeta_emp - zeta_fit) ** 2))
    ss_tot = float(np.sum((zeta_emp - zeta_emp.mean()) ** 2))
    r2 = 1.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    return zeta_fit, h_hat, lambda2_hat, r2


def _logvol_covariance_slope(dx: np.ndarray, lags: Iterable[int]) -> tuple[np.ndarray, float]:
    x = np.asarray(dx, dtype=np.float64)
    log_abs = np.log(np.abs(x) + 1e-6)
    centered = log_abs - np.mean(log_abs)
    covs = []
    valid_lags = []
    for lag in lags:
        if lag >= len(x) // 2:
            continue
        cov = float(np.mean(centered[:-lag] * centered[lag:]))
        covs.append(cov)
        valid_lags.append(lag)
    covs_arr = np.asarray(covs, dtype=np.float64)
    if len(valid_lags) < 2:
        return covs_arr, 0.0
    slope, _ = _fit_line(np.log(np.asarray(valid_lags, dtype=np.float64)), covs_arr)
    return covs_arr, float(-slope)


def _empirical_zeta(dx: np.ndarray, q_grid: Iterable[float], scales: Iterable[int]) -> np.ndarray:
    q = np.asarray(tuple(q_grid), dtype=np.float64)
    scales_arr = np.asarray(tuple(scales), dtype=np.int64)
    sq = structure_functions_from_increments(dx, q, scales_arr)
    xs = np.log(scales_arr.astype(np.float64))
    zeta = np.zeros(len(q), dtype=np.float64)
    for i in range(len(q)):
        ys = np.log(sq[i] + 1e-30)
        zeta[i], _ = _fit_line(xs, ys)
    return zeta


def _curvature_stat(q: np.ndarray, zeta_emp: np.ndarray) -> float:
    if len(q) < 3:
        return 0.0
    second = np.gradient(np.gradient(zeta_emp, q), q)
    return float(np.mean(-second))


def proxy_inverse_estimate(
    dx: np.ndarray,
    q_grid: Iterable[float] = DEFAULT_Q_GRID,
    scales: Iterable[int] = DEFAULT_SCALES,
    lags: Iterable[int] = DEFAULT_LAGS,
) -> InverseEstimate:
    q = np.asarray(tuple(q_grid), dtype=np.float64)
    zeta_emp = _empirical_zeta(dx, q, scales)
    zeta_fit, h_parabolic, lambda_from_curvature, r2 = _fit_parabolic_mrw(q, zeta_emp)
    _, lambda_from_logvol = _logvol_covariance_slope(dx, lags)

    # Combine curvature and log-vol covariance evidence. This is a statistical
    # proxy estimator, not the final neural inverse estimator.
    lambda2_hat = float(np.clip(0.5 * lambda_from_curvature + 0.5 * max(lambda_from_logvol, 0.0), 0.0, 0.2))
    h_hat = float(np.clip(0.5 * h_parabolic + 0.5 * (zeta_emp[np.argmin(np.abs(q - 2.0))] / 2.0), 1e-4, 1.0 - 1e-4))
    zeta_final = true_mrw_zeta(q, h=h_hat, lambda2=lambda2_hat)
    alpha, f_alpha = legendre_spectrum_from_zeta(q, zeta_final)
    residual_norm = float(np.mean(np.abs(zeta_emp - zeta_final)))
    curvature = _curvature_stat(q, zeta_emp)
    f_width = float(np.max(alpha) - np.min(alpha))
    logvol_covs, logvol_slope = _logvol_covariance_slope(dx, lags)
    boundary_hit = bool(lambda2_hat <= 0.005 or lambda2_hat >= 0.195)

    # Heuristic MRW-validity score for no-checkpoint fallback mode.
    score_raw = 3.5 * (r2 - 0.75) + 6.0 * lambda2_hat + 2.0 * max(logvol_slope, 0.0) - 4.0 * residual_norm
    p_mrw = float(1.0 / (1.0 + np.exp(-score_raw)))
    notes = "statistical_proxy"
    if boundary_hit:
        notes += ";lambda2_near_boundary"
    if len(logvol_covs) == 0:
        notes += ";weak_logvol_curve"

    return InverseEstimate(
        mode="proxy",
        model_name="statistical_proxy",
        checkpoint_path="",
        pred_H=h_hat,
        pred_lambda2=lambda2_hat,
        p_MRW=p_mrw,
        residual_norm=residual_norm,
        logvol_cov_slope=float(logvol_slope),
        empirical_zeta_curvature=curvature,
        f_alpha_width=f_width,
        lambda2_boundary_hit=boundary_hit,
        zeta_fit_r2=float(r2),
        notes=notes,
    )


def load_cmin_model(checkpoint_path: Path | str | None) -> tuple[CMINRegressor | None, dict[str, object]]:
    meta: dict[str, object] = {"checkpoint_path": "", "model_name": "statistical_proxy", "config": None}
    if checkpoint_path is None:
        return None, meta
    path = Path(checkpoint_path)
    if not path.exists():
        return None, meta
    model = CMINRegressor()
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
        meta["config"] = state.get("config")
        meta["model_name"] = str(state.get("model_name", "cmin_tiny_synthetic"))
    else:
        model.load_state_dict(state, strict=False)
        meta["model_name"] = "cmin_legacy_state_dict"
    meta["checkpoint_path"] = str(path)
    model.eval()
    return model, meta


def model_inverse_estimate(dx: np.ndarray, model: CMINRegressor, checkpoint_path: str = "", model_name: str = "cmin_model") -> InverseEstimate:
    x = torch.tensor(np.asarray(dx, dtype=np.float32)[None, :], dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
    residual_norm = float(torch.mean(torch.abs(out.residual_zeta)).cpu())
    q = torch.tensor(DEFAULT_Q_GRID, dtype=out.zeta_hat.dtype, device=out.zeta_hat.device)
    zeta_np = out.zeta_hat.squeeze(0).cpu().numpy()
    q_np = q.cpu().numpy()
    curvature = _curvature_stat(q_np, zeta_np)
    return InverseEstimate(
        mode="model",
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        pred_H=float(out.h_hat.squeeze().cpu()),
        pred_lambda2=float(out.lambda2_hat.squeeze().cpu()),
        p_MRW=float(out.p_mrw.squeeze().cpu()),
        residual_norm=residual_norm,
        logvol_cov_slope=float(out.logvol_covariance_slope.squeeze().cpu()),
        empirical_zeta_curvature=float(curvature),
        f_alpha_width=float((out.alpha_hat.max(dim=1).values - out.alpha_hat.min(dim=1).values).squeeze().cpu()),
        lambda2_boundary_hit=bool(float(out.lambda2_hat.squeeze().cpu()) <= 0.005 or float(out.lambda2_hat.squeeze().cpu()) >= 0.195),
        zeta_fit_r2=float("nan"),
        notes="trained_cmin_model",
    )


def estimate_window(
    dx: np.ndarray,
    checkpoint_path: Path | str | None = None,
    mode: str = "auto",
) -> InverseEstimate:
    resolved_mode = mode.lower()
    if resolved_mode not in {"auto", "proxy", "model"}:
        raise ValueError(f"Unknown estimation mode: {mode}")
    if resolved_mode == "proxy":
        return proxy_inverse_estimate(dx)

    model, meta = load_cmin_model(checkpoint_path)
    if model is None:
        if resolved_mode == "model":
            raise FileNotFoundError(f"CMIN checkpoint not found: {checkpoint_path}")
        return proxy_inverse_estimate(dx)
    return model_inverse_estimate(
        dx,
        model=model,
        checkpoint_path=str(meta.get("checkpoint_path", "")),
        model_name=str(meta.get("model_name", "cmin_model")),
    )


def estimate_to_dict(est: InverseEstimate) -> dict[str, object]:
    return asdict(est)
