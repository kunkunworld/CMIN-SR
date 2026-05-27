from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class MRWParams:
    length: int = 4096
    dt: float = 1.0
    lambda2: float = 0.03
    L: int = 512
    sigma: float = 1.0
    H: float = 0.5
    seed: Optional[int] = None


def build_log_covariance(n: int, lambda2: float, l_value: int, eps: float = 1.0) -> np.ndarray:
    cov = np.zeros(n, dtype=np.float64)
    for k in range(n):
        if k < l_value:
            val = lambda2 * np.log(l_value / (k + eps))
            cov[k] = max(val, 0.0)
        else:
            cov[k] = 0.0
    return cov


def sample_stationary_gaussian_from_toeplitz(cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(cov)
    m = 2 * n

    circ_col = np.zeros(m, dtype=np.float64)
    circ_col[:n] = cov
    circ_col[n + 1:] = cov[1:][::-1]

    eigvals = np.real(np.fft.fft(circ_col))
    eigvals[eigvals < 0] = 0.0

    z = rng.normal(size=m) + 1j * rng.normal(size=m)
    y = np.fft.ifft(np.sqrt(eigvals) * z).real
    return y[:n]


def build_fgn_covariance(n: int, h: float) -> np.ndarray:
    if not (0.0 < h < 1.0):
        raise ValueError("H must be in (0, 1) for fractional Gaussian noise.")

    k = np.arange(n, dtype=np.float64)
    cov = 0.5 * (
        np.abs(k - 1.0) ** (2.0 * h)
        - 2.0 * np.abs(k) ** (2.0 * h)
        + np.abs(k + 1.0) ** (2.0 * h)
    )
    cov[0] = 1.0
    return cov


def sample_fractional_gaussian_noise(n: int, h: float, rng: np.random.Generator) -> np.ndarray:
    cov = build_fgn_covariance(n, h)
    return sample_stationary_gaussian_from_toeplitz(cov, rng)


def generate_mrw(params: MRWParams) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(params.seed)
    n = params.length

    cov = build_log_covariance(n, params.lambda2, params.L)
    omega = sample_stationary_gaussian_from_toeplitz(cov, rng)
    omega = omega - 0.5 * cov[0]

    eps = rng.normal(size=n)
    dx = params.sigma * (params.dt ** params.H) * np.exp(omega) * eps
    x = np.cumsum(dx)
    t = np.arange(n) * params.dt

    return {
        "t": t.astype(np.float32),
        "x": x.astype(np.float32),
        "dx": dx.astype(np.float32),
        "omega": omega.astype(np.float32),
    }


def generate_mrw_fgn(params: MRWParams) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(params.seed)
    n = params.length

    cov = build_log_covariance(n, params.lambda2, params.L)
    omega = sample_stationary_gaussian_from_toeplitz(cov, rng)
    omega = omega - 0.5 * cov[0]

    eps_h = sample_fractional_gaussian_noise(n, params.H, rng)
    dx = params.sigma * np.exp(omega) * eps_h
    x = np.cumsum(dx)
    t = np.arange(n) * params.dt

    return {
        "t": t.astype(np.float32),
        "x": x.astype(np.float32),
        "dx": dx.astype(np.float32),
        "omega": omega.astype(np.float32),
    }


def generate_mrw_dataset(
    num_samples: int,
    base_params: MRWParams,
    lambda2_range: Optional[Tuple[float, float]] = None,
    l_range: Optional[Tuple[int, int]] = None,
    sigma_range: Optional[Tuple[float, float]] = None,
    h_range: Optional[Tuple[float, float]] = None,
    seed: int = 2026,
    use_fgn_base: bool = False,
) -> Dict[str, np.ndarray]:
    master_rng = np.random.default_rng(seed)
    xs = []
    dxs = []
    omegas = []
    ts = []
    labels = []

    for _ in range(num_samples):
        params = MRWParams(**asdict(base_params))
        if lambda2_range is not None:
            params.lambda2 = float(master_rng.uniform(*lambda2_range))
        if l_range is not None:
            params.L = int(master_rng.integers(l_range[0], l_range[1] + 1))
        if sigma_range is not None:
            params.sigma = float(master_rng.uniform(*sigma_range))
        if h_range is not None:
            params.H = float(master_rng.uniform(*h_range))

        params.seed = int(master_rng.integers(0, 2**31 - 1))
        sample = generate_mrw_fgn(params) if use_fgn_base else generate_mrw(params)

        ts.append(sample["t"])
        xs.append(sample["x"])
        dxs.append(sample["dx"])
        omegas.append(sample["omega"])
        labels.append([params.lambda2, params.L, params.sigma, params.H])

    return {
        "t": np.stack(ts, axis=0),
        "x": np.stack(xs, axis=0),
        "dx": np.stack(dxs, axis=0),
        "omega": np.stack(omegas, axis=0),
        "params": np.array(labels, dtype=np.float32),
        "param_names": np.array(["lambda2", "L", "sigma", "H"]),
    }


def save_dataset_npz(path: Path | str, dataset: Dict[str, np.ndarray], meta: Optional[Dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)

    if meta is not None:
        meta_path = path.with_name(f"{path.stem}_meta.json")
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
