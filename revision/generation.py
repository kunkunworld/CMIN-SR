"""Auditable Gaussian normalization correction; historical source stays frozen."""
from __future__ import annotations

import numpy as np
from mrw_dl.generation import build_fgn_covariance, build_log_covariance

PROTOCOL = "fft_covariance_normalization_corrected_v1"


def embedding(cov: np.ndarray) -> tuple[np.ndarray, dict]:
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim != 1 or len(cov) < 2 or not np.isfinite(cov).all() or cov[0] < 0:
        raise ValueError("Expected a finite covariance vector")
    n = len(cov)
    c = np.zeros(2 * n)
    c[:n] = cov
    c[n + 1:] = cov[1:][::-1]
    eig = np.fft.fft(c).real
    tol = 1e-10 * max(1., float(np.max(np.abs(eig))))
    if eig.min() < -tol:
        raise ValueError(f"Non-PSD embedding: min eigenvalue={eig.min()}; no silent clipping")
    effective = np.fft.ifft(np.maximum(eig, 0)).real[:n]
    return np.maximum(eig, 0), {
        "min_eigenvalue": float(eig.min()),
        "roundoff_negative_count": int((eig < 0).sum()),
        "max_covariance_change": float(np.max(np.abs(effective - cov))),
        "fft_length": 2 * n,
    }


def sample_gaussian(cov: np.ndarray, rng: np.random.Generator, *, corrected=True):
    eig, meta = embedding(cov)
    m = len(eig)
    z = rng.normal(size=m) + 1j * rng.normal(size=m)
    y = np.fft.ifft(np.sqrt(eig) * z).real
    if corrected:
        y *= np.sqrt(m)
    return y[:len(cov)], meta


def generate_pair(length: int, h: float, lambda2: float, seed: int,
                  integral_scale: int | None = None) -> dict:
    """Same random innovations; isolate scaling, retaining legacy mean convention.

    For h=.5, canonical omega mean=-Var vs legacy -Var/2 differs only
    by a constant amplitude at fixed discretization. Standardized increments
    are identical. For other h no analytic target is asserted here.
    """
    if not (0 < h < 1) or not (0 <= lambda2 <= .2):
        raise ValueError("Parameters outside the audited range")
    L = min(512, max(64, length // 4)) if integral_scale is None else integral_scale
    if not (2 <= L <= length):
        raise ValueError("Require 2 <= integral scale <= length")
    rng = np.random.default_rng(seed)
    cov = build_log_covariance(length, lambda2, L)
    omega0, ometa = sample_gaussian(cov, rng)
    eps, emeta = sample_gaussian(build_fgn_covariance(length, h), rng)
    m = 2 * length
    legacy_dx = np.exp(omega0 / np.sqrt(m) - .5 * cov[0]) * eps / np.sqrt(m)
    corrected_dx = np.exp(omega0 - .5 * cov[0]) * eps
    canonical_dx = np.exp(omega0 - cov[0]) * eps
    return dict(legacy=legacy_dx, corrected=corrected_dx, canonical=canonical_dx,
                omega_centered=omega0, epsilon=eps, integral_scale=L,
                omega_covariance=cov, omega_embedding=ometa, fgn_embedding=emeta)


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    if not np.isfinite(x).all() or s <= 0:
        raise ValueError("Nonfinite or degenerate series")
    return (x - x.mean()) / s
