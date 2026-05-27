from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClassicalZetaEstimate:
    estimator: str
    zeta: np.ndarray
    zeta_std: np.ndarray
    fit_quality: float
    high_q_instability: float
    warning_flags: float


def _safe_polyfit_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0, 0.0
    xs = x[mask].astype(np.float64)
    ys = y[mask].astype(np.float64)
    coef = np.polyfit(xs, ys, deg=1)
    pred = np.polyval(coef, xs)
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1.0 if ss_tot <= 1e-30 else 1.0 - ss_res / ss_tot
    return float(coef[0]), float(np.clip(r2, 0.0, 1.0))


def _aggregate_increments(x: np.ndarray, scale: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if scale <= 1:
        return x.copy()
    csum = np.concatenate([[0.0], np.cumsum(x)])
    return csum[scale:] - csum[:-scale]


def _fit_moments_to_zeta(
    moments: np.ndarray,
    q_grid: tuple[float, ...],
    scales: tuple[int, ...],
    estimator: str,
    zeta_transform: str = "direct",
) -> ClassicalZetaEstimate:
    q_arr = np.asarray(q_grid, dtype=np.float64)
    log_s = np.log(np.asarray(scales, dtype=np.float64))
    zeta = []
    r2s = []
    for qi in range(moments.shape[0]):
        y = np.log(np.asarray(moments[qi], dtype=np.float64) + 1e-30)
        slope, r2 = _safe_polyfit_slope(log_s, y)
        zeta.append(slope)
        r2s.append(r2)
    z = np.nan_to_num(np.asarray(zeta, dtype=np.float64))
    if zeta_transform == "mfdfa_hq":
        z = z * q_arr
    zeta_std = np.zeros_like(z)
    high_q_instability = float(np.nanmean(np.abs(np.diff(z[-3:])))) if z.size >= 3 else 0.0
    fit_quality = float(np.nanmean(r2s)) if r2s else 0.0
    warning_flags = float((fit_quality < 0.5) or (not np.isfinite(z).all()))
    return ClassicalZetaEstimate(
        estimator=estimator,
        zeta=z,
        zeta_std=zeta_std,
        fit_quality=fit_quality,
        high_q_instability=float(np.clip(high_q_instability, 0.0, 1.0)),
        warning_flags=warning_flags,
    )


def structure_aggregated_zeta(
    x: np.ndarray,
    q_grid: tuple[float, ...],
    scales: tuple[int, ...],
    estimator: str = "structure_aggregated_ols",
) -> ClassicalZetaEstimate:
    moments = []
    for q in q_grid:
        vals = []
        for s in scales:
            inc = np.abs(_aggregate_increments(x, int(s))) + 1e-30
            vals.append(float(np.mean(inc ** float(q))))
        moments.append(vals)
    return _fit_moments_to_zeta(np.asarray(moments), q_grid, scales, estimator=estimator)


def mfdfa_zeta(
    x: np.ndarray,
    q_grid: tuple[float, ...],
    scales: tuple[int, ...],
    order: int = 1,
    estimator: str = "mfdfa",
) -> ClassicalZetaEstimate:
    x = np.asarray(x, dtype=np.float64)
    profile = np.cumsum(x - np.mean(x))
    n = profile.size
    fq = np.full((len(q_grid), len(scales)), np.nan, dtype=np.float64)
    for sj, scale in enumerate(scales):
        s = int(scale)
        if s <= order + 2:
            continue
        segments = n // s
        if segments < 4:
            continue
        t = np.arange(s, dtype=np.float64)
        variances: list[float] = []
        for v in range(segments):
            seg = profile[v * s : (v + 1) * s]
            coef = np.polyfit(t, seg, deg=order)
            trend = np.polyval(coef, t)
            variances.append(float(np.mean((seg - trend) ** 2)))
        for v in range(segments):
            seg = profile[n - (v + 1) * s : n - v * s]
            coef = np.polyfit(t, seg, deg=order)
            trend = np.polyval(coef, t)
            variances.append(float(np.mean((seg - trend) ** 2)))
        f2 = np.asarray(variances, dtype=np.float64) + 1e-30
        for qi, q in enumerate(q_grid):
            qq = float(q)
            if abs(qq) < 1e-12:
                fq[qi, sj] = np.exp(0.5 * np.mean(np.log(f2)))
            else:
                fq[qi, sj] = float((np.mean(f2 ** (qq / 2.0))) ** (1.0 / qq))
    return _fit_moments_to_zeta(fq, q_grid, scales, estimator=estimator, zeta_transform="mfdfa_hq")


def _haar_block_contrasts(x: np.ndarray, scale: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    s = int(scale)
    if s < 2:
        return np.abs(x)
    half = max(1, s // 2)
    width = half * 2
    if x.size < width:
        return np.empty(0, dtype=np.float64)
    csum = np.concatenate([[0.0], np.cumsum(x)])
    starts = np.arange(0, x.size - width + 1, dtype=int)
    left = csum[starts + half] - csum[starts]
    right = csum[starts + width] - csum[starts + half]
    # Unnormalized Haar block contrast. For return/increment inputs this keeps
    # the scaling convention close to aggregate-increment zeta(q).
    return np.abs(left - right) + 1e-30


def _local_maxima(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return values
    mid = values[1:-1]
    mask = (mid >= values[:-2]) & (mid >= values[2:])
    maxima = mid[mask]
    return maxima if maxima.size >= 4 else values


def wavelet_leader_zeta(
    x: np.ndarray,
    q_grid: tuple[float, ...],
    scales: tuple[int, ...],
    estimator: str = "wavelet_leader_haar",
) -> ClassicalZetaEstimate:
    coeff_by_scale = [np.abs(_haar_block_contrasts(x, int(s))) + 1e-30 for s in scales]
    moments = []
    for q in q_grid:
        vals = []
        for j, coeff in enumerate(coeff_by_scale):
            if coeff.size == 0:
                vals.append(np.nan)
                continue
            leader = coeff.copy()
            # Local sup across neighboring positions and one finer scale when
            # available. This is a compact Haar leader approximation, not a full
            # production wavelet-leader package.
            if coeff.size >= 3:
                leader[1:-1] = np.maximum.reduce([coeff[:-2], coeff[1:-1], coeff[2:]])
            if j > 0 and coeff_by_scale[j - 1].size:
                finer = coeff_by_scale[j - 1]
                idx = np.minimum((np.arange(leader.size) * 2).astype(int), finer.size - 1)
                leader = np.maximum(leader, finer[idx])
            vals.append(float(np.mean((leader + 1e-30) ** float(q))))
        moments.append(vals)
    return _fit_moments_to_zeta(np.asarray(moments), q_grid, scales, estimator=estimator)


def wtmm_haar_zeta(
    x: np.ndarray,
    q_grid: tuple[float, ...],
    scales: tuple[int, ...],
    estimator: str = "wtmm_haar",
) -> ClassicalZetaEstimate:
    moments = []
    for q in q_grid:
        vals = []
        for s in scales:
            coeff = np.abs(_haar_block_contrasts(x, int(s))) + 1e-30
            maxima = _local_maxima(coeff)
            vals.append(float(np.mean((maxima + 1e-30) ** float(q))) if maxima.size else np.nan)
        moments.append(vals)
    return _fit_moments_to_zeta(np.asarray(moments), q_grid, scales, estimator=estimator)


def estimate_classical_zeta(
    x: np.ndarray,
    q_grid: tuple[float, ...],
    scales: tuple[int, ...],
    estimator: str,
) -> ClassicalZetaEstimate:
    if estimator == "structure_aggregated_ols":
        return structure_aggregated_zeta(x, q_grid, scales, estimator=estimator)
    if estimator == "mfdfa":
        return mfdfa_zeta(x, q_grid, scales, estimator=estimator)
    if estimator == "mfdfa_quadratic":
        return mfdfa_zeta(x, q_grid, scales, order=2, estimator=estimator)
    if estimator == "wavelet_leader_haar":
        return wavelet_leader_zeta(x, q_grid, scales, estimator=estimator)
    if estimator == "wtmm_haar":
        return wtmm_haar_zeta(x, q_grid, scales, estimator=estimator)
    raise ValueError(f"Unknown classical estimator: {estimator}")
