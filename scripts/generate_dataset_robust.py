from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_dl.generation import MRWParams, generate_mrw_fgn, save_dataset_npz


def theoretical_alpha_summary(h: float, lambda2: float) -> list[float]:
    alpha_min_q3 = h + lambda2 - 3.0 * lambda2
    alpha_max_qm3 = h + lambda2 + 3.0 * lambda2
    alpha_peak = h + lambda2
    alpha_width = alpha_max_qm3 - alpha_min_q3
    return [alpha_min_q3, alpha_peak, alpha_max_qm3, alpha_width]


def sample_perturbation(rng: np.random.Generator) -> dict[str, float]:
    regime = rng.choice(["clean", "low", "medium", "high"], p=[0.15, 0.35, 0.35, 0.15])
    if regime == "clean":
        noise_std = 0.0
        outlier_prob = 0.0
        drift_strength = 0.0
    elif regime == "low":
        noise_std = float(rng.uniform(0.01, 0.03))
        outlier_prob = float(rng.uniform(0.0, 0.002))
        drift_strength = float(rng.uniform(0.0, 0.01))
    elif regime == "medium":
        noise_std = float(rng.uniform(0.03, 0.08))
        outlier_prob = float(rng.uniform(0.002, 0.006))
        drift_strength = float(rng.uniform(0.01, 0.03))
    else:
        noise_std = float(rng.uniform(0.08, 0.15))
        outlier_prob = float(rng.uniform(0.006, 0.012))
        drift_strength = float(rng.uniform(0.03, 0.06))

    return {
        "noise_std": noise_std,
        "outlier_prob": outlier_prob,
        "outlier_scale": float(rng.uniform(3.0, 8.0)),
        "drift_strength": drift_strength,
        "gain": float(rng.uniform(0.7, 1.3)),
        "ar_noise_phi": float(rng.uniform(0.0, 0.6)),
    }


def apply_perturbations(dx: np.ndarray, rng: np.random.Generator, cfg: dict[str, float]) -> np.ndarray:
    dx = np.asarray(dx, dtype=np.float64)
    scale = float(np.std(dx) + 1e-8)
    n = len(dx)
    y = dx.copy() * cfg["gain"]

    if cfg["noise_std"] > 0:
        white = rng.normal(0.0, cfg["noise_std"] * scale, size=n)
        ar = np.zeros(n, dtype=np.float64)
        eps = rng.normal(0.0, cfg["noise_std"] * scale, size=n)
        phi = cfg["ar_noise_phi"]
        for i in range(1, n):
            ar[i] = phi * ar[i - 1] + eps[i]
        y = y + 0.7 * white + 0.3 * ar

    if cfg["outlier_prob"] > 0:
        mask = rng.random(n) < cfg["outlier_prob"]
        signs = rng.choice([-1.0, 1.0], size=n)
        y = y + mask * signs * cfg["outlier_scale"] * scale

    if cfg["drift_strength"] > 0:
        t = np.linspace(-1.0, 1.0, n, dtype=np.float64)
        linear = rng.normal() * t
        sinusoid = np.sin(2.0 * np.pi * rng.uniform(0.5, 3.0) * (t + 1.0) / 2.0 + rng.uniform(0, 2 * np.pi))
        drift = cfg["drift_strength"] * scale * (0.5 * linear + 0.5 * sinusoid)
        y = y + drift

    return y.astype(np.float32)


def main() -> None:
    output_path = ROOT / "data" / "raw" / "mrw_dataset_robust_fgn.npz"
    rng = np.random.default_rng(2027)

    length = 4096
    samples_per_cell = 100
    h_edges = np.linspace(0.24, 0.86, 9)
    lambda2_edges = np.linspace(0.02, 0.18, 7)
    l_range = (64, 1024)
    sigma_range = (0.7, 1.3)

    xs = []
    dxs = []
    omegas = []
    ts = []
    params_list = []
    alpha_summaries = []
    perturbations = []

    for h_bin in range(len(h_edges) - 1):
        for lambda_bin in range(len(lambda2_edges) - 1):
            h_lo, h_hi = float(h_edges[h_bin]), float(h_edges[h_bin + 1])
            lam_lo, lam_hi = float(lambda2_edges[lambda_bin]), float(lambda2_edges[lambda_bin + 1])

            for _ in range(samples_per_cell):
                lambda2 = float(rng.uniform(lam_lo, lam_hi))
                min_h = max(h_lo, 2.0 * lambda2 + 0.04)
                if min_h >= h_hi:
                    min_h = h_lo
                h = float(rng.uniform(min_h, h_hi))
                params = MRWParams(
                    length=length,
                    dt=1.0,
                    lambda2=lambda2,
                    L=int(rng.integers(l_range[0], l_range[1] + 1)),
                    sigma=float(rng.uniform(*sigma_range)),
                    H=h,
                    seed=int(rng.integers(0, 2**31 - 1)),
                )

                clean = generate_mrw_fgn(params)
                perturb_cfg = sample_perturbation(rng)
                dx_obs = apply_perturbations(clean["dx"], rng, perturb_cfg)
                x_obs = np.cumsum(dx_obs).astype(np.float32)

                ts.append(clean["t"])
                xs.append(x_obs)
                dxs.append(dx_obs)
                omegas.append(clean["omega"])
                params_list.append([params.lambda2, params.L, params.sigma, params.H])
                alpha_summaries.append(theoretical_alpha_summary(params.H, params.lambda2))
                perturbations.append([
                    perturb_cfg["noise_std"],
                    perturb_cfg["outlier_prob"],
                    perturb_cfg["outlier_scale"],
                    perturb_cfg["drift_strength"],
                    perturb_cfg["gain"],
                    perturb_cfg["ar_noise_phi"],
                    float(h_bin),
                    float(lambda_bin),
                ])

    dataset = {
        "t": np.stack(ts, axis=0),
        "x": np.stack(xs, axis=0),
        "dx": np.stack(dxs, axis=0),
        "omega": np.stack(omegas, axis=0),
        "params": np.asarray(params_list, dtype=np.float32),
        "param_names": np.array(["lambda2", "L", "sigma", "H"]),
        "alpha_summary": np.asarray(alpha_summaries, dtype=np.float32),
        "alpha_summary_names": np.array(["alpha_min_q3", "alpha_peak", "alpha_max_qm3", "alpha_width_q-3_to_3"]),
        "perturbations": np.asarray(perturbations, dtype=np.float32),
        "perturbation_names": np.array([
            "noise_std",
            "outlier_prob",
            "outlier_scale",
            "drift_strength",
            "gain",
            "ar_noise_phi",
            "H_bin",
            "lambda2_bin",
        ]),
    }

    meta = {
        "description": "Robust FGN-based MRW dataset with stratified alpha coverage and observation perturbations",
        "num_samples": int(dataset["dx"].shape[0]),
        "base_params": asdict(MRWParams(length=length, dt=1.0, lambda2=0.1, L=512, sigma=1.0, H=0.5)),
        "sampling": {
            "samples_per_cell": samples_per_cell,
            "H_edges": h_edges.round(6).tolist(),
            "lambda2_edges": lambda2_edges.round(6).tolist(),
            "L_range": list(l_range),
            "sigma_range": list(sigma_range),
            "alpha_summary": "alpha(q)=H+lambda2-lambda2*q over q in [-3, 3]",
        },
        "perturbations": {
            "regimes": ["clean", "low", "medium", "high"],
            "includes": [
                "random gain",
                "additive white measurement noise",
                "AR(1)-colored measurement noise",
                "sparse impulse outliers",
                "low-frequency drift",
            ],
        },
        "notes": [
            "Labels remain the clean MRW generation parameters.",
            "dx and x are perturbed observations used as harder model inputs.",
            "alpha_summary stores theoretical clean-spectrum coverage for stratification checks.",
        ],
    }

    save_dataset_npz(output_path, dataset, meta=meta)

    alpha = dataset["alpha_summary"]
    params = dataset["params"]
    perturb = dataset["perturbations"]
    summary = {
        "saved_to": str(output_path),
        "num_samples": int(dataset["dx"].shape[0]),
        "sequence_length": int(dataset["dx"].shape[1]),
        "lambda2_range": [float(params[:, 0].min()), float(params[:, 0].max())],
        "H_range": [float(params[:, 3].min()), float(params[:, 3].max())],
        "alpha_min_range": [float(alpha[:, 0].min()), float(alpha[:, 0].max())],
        "alpha_peak_range": [float(alpha[:, 1].min()), float(alpha[:, 1].max())],
        "alpha_max_range": [float(alpha[:, 2].min()), float(alpha[:, 2].max())],
        "alpha_width_range": [float(alpha[:, 3].min()), float(alpha[:, 3].max())],
        "noise_std_mean": float(perturb[:, 0].mean()),
        "outlier_prob_mean": float(perturb[:, 1].mean()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
