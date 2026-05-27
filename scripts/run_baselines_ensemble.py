from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_dl.baselines import (
    fit_hq_from_mfdfa,
    fit_zeta_from_structure_functions,
    legendre_spectrum_from_zeta,
    mfdfa,
    structure_functions_from_increments,
    true_mrw_zeta,
)
from mrw_dl.generation import MRWParams, generate_mrw


def main() -> None:
    params = MRWParams(length=4096, dt=1.0, lambda2=0.14, L=512, sigma=1.0, H=0.68, seed=None)
    num_realizations = 128

    q_vals = np.linspace(0.5, 3.0, 11)
    sf_scales = np.unique(np.logspace(np.log10(8), np.log10(512), 16).astype(int))
    mfdfa_scales = np.unique(np.logspace(np.log10(16), np.log10(1024), 16).astype(int))

    sq_list = []
    fq_list = []
    for seed in range(num_realizations):
        p = MRWParams(**params.__dict__)
        p.seed = 1000 + seed
        sample = generate_mrw(p)
        sq_list.append(structure_functions_from_increments(sample["dx"], q_vals, sf_scales))
        fq_list.append(mfdfa(sample["dx"], q_vals, mfdfa_scales, order=1))

    sq_mean = np.mean(np.stack(sq_list, axis=0), axis=0)
    fq_mean = np.mean(np.stack(fq_list, axis=0), axis=0)

    zeta_true = true_mrw_zeta(q_vals, h=params.H, lambda2=params.lambda2)
    zeta_sf = fit_zeta_from_structure_functions(sq_mean, q_vals, sf_scales, fit_slice=slice(2, -2))
    hq = fit_hq_from_mfdfa(fq_mean, q_vals, mfdfa_scales, fit_slice=slice(2, -2))
    zeta_mfdfa = q_vals * hq

    alpha_true, f_true = legendre_spectrum_from_zeta(q_vals, zeta_true)
    alpha_sf, f_sf = legendre_spectrum_from_zeta(q_vals, zeta_sf)
    alpha_mfdfa, f_mfdfa = legendre_spectrum_from_zeta(q_vals, zeta_mfdfa)

    def _mae(a, b):
        return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

    result = {
        "params": {
            "lambda2": params.lambda2,
            "L": params.L,
            "sigma": params.sigma,
            "H": params.H,
        },
        "num_realizations": num_realizations,
        "q_vals": q_vals.round(6).tolist(),
        "zeta_true": zeta_true.round(6).tolist(),
        "zeta_sf": zeta_sf.round(6).tolist(),
        "zeta_mfdfa": zeta_mfdfa.round(6).tolist(),
        "alpha_true": alpha_true.round(6).tolist(),
        "f_true": f_true.round(6).tolist(),
        "alpha_sf": alpha_sf.round(6).tolist(),
        "f_sf": f_sf.round(6).tolist(),
        "alpha_mfdfa": alpha_mfdfa.round(6).tolist(),
        "f_mfdfa": f_mfdfa.round(6).tolist(),
        "metrics": {
            "zeta_mae_sf": _mae(zeta_true, zeta_sf),
            "zeta_mae_mfdfa": _mae(zeta_true, zeta_mfdfa),
            "spectrum_mae_sf": _mae(f_true, f_sf),
            "spectrum_mae_mfdfa": _mae(f_true, f_mfdfa),
        },
    }

    output_dir = ROOT / "outputs" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_ensemble_result.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
