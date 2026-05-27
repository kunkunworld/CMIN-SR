from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def true_mrw_zeta(q: np.ndarray, h: float, lambda2: float) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q * h - 0.5 * lambda2 * q * (q - 2.0)


def legendre_spectrum_from_zeta(q: np.ndarray, zeta_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=np.float64)
    zeta_q = np.asarray(zeta_q, dtype=np.float64)
    alpha = np.gradient(zeta_q, q)
    f_alpha = q * alpha - zeta_q + 1.0
    return alpha, f_alpha


def structure_functions_from_increments(dx: np.ndarray, q_vals: np.ndarray, scales: np.ndarray) -> np.ndarray:
    dx = np.asarray(dx, dtype=np.float64)
    q_vals = np.asarray(q_vals, dtype=np.float64)
    scales = np.asarray(scales, dtype=int)

    csum = np.concatenate([[0.0], np.cumsum(dx)])
    sq = np.zeros((len(q_vals), len(scales)), dtype=np.float64)

    for j, scale in enumerate(scales):
        inc = csum[scale:] - csum[:-scale]
        abs_inc = np.abs(inc) + 1e-30
        for i, q in enumerate(q_vals):
            sq[i, j] = np.mean(abs_inc ** q)

    return sq


def fit_zeta_from_structure_functions(
    sq: np.ndarray,
    q_vals: np.ndarray,
    scales: np.ndarray,
    fit_slice: slice | None = None,
) -> np.ndarray:
    del q_vals
    if fit_slice is None:
        fit_slice = slice(None)

    xs = np.log(scales[fit_slice].astype(np.float64))
    zeta_est = np.zeros(sq.shape[0], dtype=np.float64)

    for i in range(sq.shape[0]):
        ys = np.log(sq[i, fit_slice] + 1e-30)
        coef = np.polyfit(xs, ys, deg=1)
        zeta_est[i] = coef[0]

    return zeta_est


def mfdfa(signal: np.ndarray, q_vals: np.ndarray, scales: np.ndarray, order: int = 1) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    q_vals = np.asarray(q_vals, dtype=np.float64)
    scales = np.asarray(scales, dtype=int)

    profile = np.cumsum(x - np.mean(x))
    n = len(profile)
    fq = np.full((len(q_vals), len(scales)), np.nan, dtype=np.float64)

    for j, scale in enumerate(scales):
        if scale <= order + 2:
            continue

        segments = n // scale
        if segments < 4:
            continue

        f2_all: List[float] = []
        t = np.arange(scale, dtype=np.float64)

        for v in range(segments):
            seg = profile[v * scale:(v + 1) * scale]
            coeff = np.polyfit(t, seg, deg=order)
            trend = np.polyval(coeff, t)
            f2_all.append(float(np.mean((seg - trend) ** 2)))

        for v in range(segments):
            seg = profile[n - (v + 1) * scale:n - v * scale]
            coeff = np.polyfit(t, seg, deg=order)
            trend = np.polyval(coeff, t)
            f2_all.append(float(np.mean((seg - trend) ** 2)))

        f2_arr = np.asarray(f2_all, dtype=np.float64) + 1e-30
        for i, q in enumerate(q_vals):
            if np.isclose(q, 0.0):
                fq[i, j] = np.exp(0.5 * np.mean(np.log(f2_arr)))
            else:
                fq[i, j] = (np.mean(f2_arr ** (q / 2.0))) ** (1.0 / q)

    return fq


def fit_hq_from_mfdfa(
    fq: np.ndarray,
    q_vals: np.ndarray,
    scales: np.ndarray,
    fit_slice: slice | None = None,
) -> np.ndarray:
    del q_vals
    if fit_slice is None:
        fit_slice = slice(None)

    xs = np.log(scales[fit_slice].astype(np.float64))
    hq = np.zeros(fq.shape[0], dtype=np.float64)

    for i in range(fq.shape[0]):
        ys = np.log(fq[i, fit_slice] + 1e-30)
        coef = np.polyfit(xs, ys, deg=1)
        hq[i] = coef[0]

    return hq


def zeta_from_hq(q_vals: np.ndarray, hq: np.ndarray) -> np.ndarray:
    return np.asarray(q_vals, dtype=np.float64) * np.asarray(hq, dtype=np.float64)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


@dataclass
class BaselineResult:
    sample_index: int
    lambda2: float
    l_value: float
    sigma: float
    h: float
    q_vals: np.ndarray
    zeta_true: np.ndarray
    zeta_sf: np.ndarray
    zeta_mfdfa: np.ndarray
    alpha_true: np.ndarray
    f_true: np.ndarray
    alpha_sf: np.ndarray
    f_sf: np.ndarray
    alpha_mfdfa: np.ndarray
    f_mfdfa: np.ndarray
    zeta_mae_sf: float
    zeta_mae_mfdfa: float
    spectrum_mae_sf: float
    spectrum_mae_mfdfa: float


def run_baselines_on_sample(
    dx: np.ndarray,
    params: np.ndarray,
    sample_index: int,
    q_vals: np.ndarray,
    sf_scales: np.ndarray,
    mfdfa_scales: np.ndarray,
    fit_slice_sf: slice | None = None,
    fit_slice_mfdfa: slice | None = None,
    mfdfa_order: int = 1,
) -> BaselineResult:
    lambda2, l_value, sigma, h = [float(v) for v in params]

    zeta_true = true_mrw_zeta(q_vals, h=h, lambda2=lambda2)
    alpha_true, f_true = legendre_spectrum_from_zeta(q_vals, zeta_true)

    sq = structure_functions_from_increments(dx, q_vals, sf_scales)
    zeta_sf = fit_zeta_from_structure_functions(sq, q_vals, sf_scales, fit_slice=fit_slice_sf)
    alpha_sf, f_sf = legendre_spectrum_from_zeta(q_vals, zeta_sf)

    fq = mfdfa(dx, q_vals, mfdfa_scales, order=mfdfa_order)
    hq = fit_hq_from_mfdfa(fq, q_vals, mfdfa_scales, fit_slice=fit_slice_mfdfa)
    zeta_mfdfa = zeta_from_hq(q_vals, hq)
    alpha_mfdfa, f_mfdfa = legendre_spectrum_from_zeta(q_vals, zeta_mfdfa)

    return BaselineResult(
        sample_index=sample_index,
        lambda2=lambda2,
        l_value=l_value,
        sigma=sigma,
        h=h,
        q_vals=q_vals,
        zeta_true=zeta_true,
        zeta_sf=zeta_sf,
        zeta_mfdfa=zeta_mfdfa,
        alpha_true=alpha_true,
        f_true=f_true,
        alpha_sf=alpha_sf,
        f_sf=f_sf,
        alpha_mfdfa=alpha_mfdfa,
        f_mfdfa=f_mfdfa,
        zeta_mae_sf=mae(zeta_true, zeta_sf),
        zeta_mae_mfdfa=mae(zeta_true, zeta_mfdfa),
        spectrum_mae_sf=mae(f_true, f_sf),
        spectrum_mae_mfdfa=mae(f_true, f_mfdfa),
    )


def result_to_dict(result: BaselineResult) -> Dict[str, object]:
    return {
        "sample_index": result.sample_index,
        "params": {
            "lambda2": result.lambda2,
            "L": result.l_value,
            "sigma": result.sigma,
            "H": result.h,
        },
        "q_vals": np.round(result.q_vals, 6).tolist(),
        "zeta_true": np.round(result.zeta_true, 6).tolist(),
        "zeta_sf": np.round(result.zeta_sf, 6).tolist(),
        "zeta_mfdfa": np.round(result.zeta_mfdfa, 6).tolist(),
        "alpha_true": np.round(result.alpha_true, 6).tolist(),
        "f_true": np.round(result.f_true, 6).tolist(),
        "alpha_sf": np.round(result.alpha_sf, 6).tolist(),
        "f_sf": np.round(result.f_sf, 6).tolist(),
        "alpha_mfdfa": np.round(result.alpha_mfdfa, 6).tolist(),
        "f_mfdfa": np.round(result.f_mfdfa, 6).tolist(),
        "metrics": {
            "zeta_mae_sf": result.zeta_mae_sf,
            "zeta_mae_mfdfa": result.zeta_mae_mfdfa,
            "spectrum_mae_sf": result.spectrum_mae_sf,
            "spectrum_mae_mfdfa": result.spectrum_mae_mfdfa,
        },
    }


def structure_functions_nonoverlap(dx: np.ndarray, q_vals: np.ndarray, scales: np.ndarray) -> np.ndarray:
    dx = np.asarray(dx, dtype=np.float64)
    q_vals = np.asarray(q_vals, dtype=np.float64)
    scales = np.asarray(scales, dtype=int)

    csum = np.concatenate([[0.0], np.cumsum(dx)])
    sq = np.zeros((len(q_vals), len(scales)), dtype=np.float64)

    for j, scale in enumerate(scales):
        starts = np.arange(0, len(dx) - scale + 1, scale, dtype=int)
        if len(starts) < 8:
            sq[:, j] = np.nan
            continue
        inc = csum[starts + scale] - csum[starts]
        abs_inc = np.abs(inc) + 1e-30
        for i, q in enumerate(q_vals):
            sq[i, j] = np.mean(abs_inc ** q)
    return sq


def _regression_r2(xs: np.ndarray, ys: np.ndarray, coef: np.ndarray) -> float:
    pred = np.polyval(coef, xs)
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    if ss_tot <= 1e-30:
        return 1.0
    return 1.0 - ss_res / ss_tot


def select_best_fit_slice(log_scales: np.ndarray, log_values: np.ndarray, min_points: int = 5) -> slice:
    n_scales = len(log_scales)
    best_score = -np.inf
    best_window = slice(0, n_scales)

    for start in range(0, n_scales - min_points + 1):
        for stop in range(start + min_points, n_scales + 1):
            xs = log_scales[start:stop]
            r2_scores = []
            for row in log_values:
                ys = row[start:stop]
                if np.any(~np.isfinite(ys)):
                    continue
                coef = np.polyfit(xs, ys, deg=1)
                r2_scores.append(_regression_r2(xs, ys, coef))
            if not r2_scores:
                continue
            score = float(np.mean(r2_scores))
            if score > best_score:
                best_score = score
                best_window = slice(start, stop)
    return best_window


def fit_zeta_with_adaptive_window(
    sq: np.ndarray,
    scales: np.ndarray,
    min_points: int = 5,
) -> Tuple[np.ndarray, slice]:
    log_scales = np.log(np.asarray(scales, dtype=np.float64))
    log_values = np.log(np.asarray(sq, dtype=np.float64) + 1e-30)
    fit_slice = select_best_fit_slice(log_scales, log_values, min_points=min_points)

    zeta_est = np.zeros(log_values.shape[0], dtype=np.float64)
    xs = log_scales[fit_slice]
    for i in range(log_values.shape[0]):
        ys = log_values[i, fit_slice]
        coef = np.polyfit(xs, ys, deg=1)
        zeta_est[i] = coef[0]
    return zeta_est, fit_slice


def fit_hq_with_adaptive_window(
    fq: np.ndarray,
    scales: np.ndarray,
    min_points: int = 5,
) -> Tuple[np.ndarray, slice]:
    log_scales = np.log(np.asarray(scales, dtype=np.float64))
    log_values = np.log(np.asarray(fq, dtype=np.float64) + 1e-30)
    fit_slice = select_best_fit_slice(log_scales, log_values, min_points=min_points)

    hq = np.zeros(log_values.shape[0], dtype=np.float64)
    xs = log_scales[fit_slice]
    for i in range(log_values.shape[0]):
        ys = log_values[i, fit_slice]
        coef = np.polyfit(xs, ys, deg=1)
        hq[i] = coef[0]
    return hq, fit_slice


def smooth_zeta_quadratic(q_vals: np.ndarray, zeta_est: np.ndarray) -> np.ndarray:
    q_vals = np.asarray(q_vals, dtype=np.float64)
    zeta_est = np.asarray(zeta_est, dtype=np.float64)
    coef = np.polyfit(q_vals, zeta_est, deg=2)
    return np.polyval(coef, q_vals)


def fit_mrw_parabolic_zeta(q_vals: np.ndarray, zeta_est: np.ndarray) -> Tuple[np.ndarray, float, float]:
    q_vals = np.asarray(q_vals, dtype=np.float64)
    zeta_est = np.asarray(zeta_est, dtype=np.float64)
    design = np.column_stack([q_vals, -0.5 * q_vals * (q_vals - 2.0)])
    coef, _, _, _ = np.linalg.lstsq(design, zeta_est, rcond=None)
    h_hat = float(coef[0])
    lambda2_hat = float(coef[1])
    zeta_fit = true_mrw_zeta(q_vals, h=h_hat, lambda2=lambda2_hat)
    return zeta_fit, h_hat, lambda2_hat


@dataclass
class ImprovedBaselineResult:
    sample_index: int
    lambda2: float
    l_value: float
    sigma: float
    h: float
    q_vals: np.ndarray
    zeta_true: np.ndarray
    zeta_sf: np.ndarray
    zeta_mfdfa: np.ndarray
    alpha_true: np.ndarray
    f_true: np.ndarray
    alpha_sf: np.ndarray
    f_sf: np.ndarray
    alpha_mfdfa: np.ndarray
    f_mfdfa: np.ndarray
    sf_fit_slice: Tuple[int, int]
    mfdfa_fit_slice: Tuple[int, int]
    sf_h_hat: float
    sf_lambda2_hat: float
    mfdfa_h_hat: float
    mfdfa_lambda2_hat: float
    zeta_mae_sf: float
    zeta_mae_mfdfa: float
    spectrum_mae_sf: float
    spectrum_mae_mfdfa: float


def run_improved_baselines_on_sample(
    dx: np.ndarray,
    params: np.ndarray,
    sample_index: int,
    q_vals: np.ndarray,
    sf_scales: np.ndarray,
    mfdfa_scales: np.ndarray,
    mfdfa_order: int = 1,
    min_fit_points: int = 5,
) -> ImprovedBaselineResult:
    lambda2, l_value, sigma, h = [float(v) for v in params]

    zeta_true = true_mrw_zeta(q_vals, h=h, lambda2=lambda2)
    alpha_true, f_true = legendre_spectrum_from_zeta(q_vals, zeta_true)

    sq = structure_functions_from_increments(dx, q_vals, sf_scales)
    zeta_sf_raw, sf_fit_slice = fit_zeta_with_adaptive_window(sq, sf_scales, min_points=min_fit_points)
    zeta_sf, sf_h_hat, sf_lambda2_hat = fit_mrw_parabolic_zeta(q_vals, zeta_sf_raw)
    alpha_sf, f_sf = legendre_spectrum_from_zeta(q_vals, zeta_sf)

    fq = mfdfa(dx, q_vals, mfdfa_scales, order=mfdfa_order)
    hq_raw, mfdfa_fit_slice = fit_hq_with_adaptive_window(fq, mfdfa_scales, min_points=min_fit_points)
    zeta_mfdfa, mfdfa_h_hat, mfdfa_lambda2_hat = fit_mrw_parabolic_zeta(q_vals, zeta_from_hq(q_vals, hq_raw))
    alpha_mfdfa, f_mfdfa = legendre_spectrum_from_zeta(q_vals, zeta_mfdfa)

    return ImprovedBaselineResult(
        sample_index=sample_index,
        lambda2=lambda2,
        l_value=l_value,
        sigma=sigma,
        h=h,
        q_vals=q_vals,
        zeta_true=zeta_true,
        zeta_sf=zeta_sf,
        zeta_mfdfa=zeta_mfdfa,
        alpha_true=alpha_true,
        f_true=f_true,
        alpha_sf=alpha_sf,
        f_sf=f_sf,
        alpha_mfdfa=alpha_mfdfa,
        f_mfdfa=f_mfdfa,
        sf_fit_slice=(sf_fit_slice.start, sf_fit_slice.stop),
        mfdfa_fit_slice=(mfdfa_fit_slice.start, mfdfa_fit_slice.stop),
        sf_h_hat=sf_h_hat,
        sf_lambda2_hat=sf_lambda2_hat,
        mfdfa_h_hat=mfdfa_h_hat,
        mfdfa_lambda2_hat=mfdfa_lambda2_hat,
        zeta_mae_sf=mae(zeta_true, zeta_sf),
        zeta_mae_mfdfa=mae(zeta_true, zeta_mfdfa),
        spectrum_mae_sf=mae(f_true, f_sf),
        spectrum_mae_mfdfa=mae(f_true, f_mfdfa),
    )


def improved_result_to_dict(result: ImprovedBaselineResult) -> Dict[str, object]:
    return {
        "sample_index": result.sample_index,
        "params": {
            "lambda2": result.lambda2,
            "L": result.l_value,
            "sigma": result.sigma,
            "H": result.h,
        },
        "q_vals": np.round(result.q_vals, 6).tolist(),
        "zeta_true": np.round(result.zeta_true, 6).tolist(),
        "zeta_sf": np.round(result.zeta_sf, 6).tolist(),
        "zeta_mfdfa": np.round(result.zeta_mfdfa, 6).tolist(),
        "alpha_true": np.round(result.alpha_true, 6).tolist(),
        "f_true": np.round(result.f_true, 6).tolist(),
        "alpha_sf": np.round(result.alpha_sf, 6).tolist(),
        "f_sf": np.round(result.f_sf, 6).tolist(),
        "alpha_mfdfa": np.round(result.alpha_mfdfa, 6).tolist(),
        "f_mfdfa": np.round(result.f_mfdfa, 6).tolist(),
        "sf_fit_slice": list(result.sf_fit_slice),
        "mfdfa_fit_slice": list(result.mfdfa_fit_slice),
        "sf_estimated_params": {
            "H_hat": result.sf_h_hat,
            "lambda2_hat": result.sf_lambda2_hat,
        },
        "mfdfa_estimated_params": {
            "H_hat": result.mfdfa_h_hat,
            "lambda2_hat": result.mfdfa_lambda2_hat,
        },
        "metrics": {
            "zeta_mae_sf": result.zeta_mae_sf,
            "zeta_mae_mfdfa": result.zeta_mae_mfdfa,
            "spectrum_mae_sf": result.spectrum_mae_sf,
            "spectrum_mae_mfdfa": result.spectrum_mae_mfdfa,
        },
    }
