from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_dl.generation import MRWParams, generate_mrw_dataset, save_dataset_npz


def main() -> None:
    output_path = ROOT / "data" / "raw" / "mrw_dataset_wide_fgn.npz"

    base = MRWParams(
        length=4096,
        dt=1.0,
        lambda2=0.08,
        L=512,
        sigma=1.0,
        H=0.5,
        seed=None,
    )

    ranges = {
        "lambda2_range": [0.02, 0.18],
        "L_range": [64, 1024],
        "sigma_range": [0.6, 1.4],
        "H_range": [0.25, 0.75],
    }

    dataset = generate_mrw_dataset(
        num_samples=3000,
        base_params=base,
        lambda2_range=tuple(ranges["lambda2_range"]),
        l_range=tuple(ranges["L_range"]),
        sigma_range=tuple(ranges["sigma_range"]),
        h_range=tuple(ranges["H_range"]),
        seed=2026,
        use_fgn_base=True,
    )

    meta = {
        "description": "Synthetic FGN-based MRW dataset with wider parameter ranges for stronger multifractality",
        "base_params": asdict(base),
        "num_samples": 3000,
        "ranges": ranges,
        "design_notes": [
            "Expanded lambda2 to increase intermittency and spectral width.",
            "Expanded H to diversify global roughness.",
            "Expanded L to vary correlation scale more strongly.",
            "Sequence length increased to 4096 for more stable baseline estimation.",
            "Uses fractional Gaussian noise as the base process so H affects scaling behavior.",
        ],
    }

    save_dataset_npz(output_path, dataset, meta=meta)

    summary = {
        "saved_to": str(output_path),
        "num_samples": int(dataset["dx"].shape[0]),
        "sequence_length": int(dataset["dx"].shape[1]),
        "param_means": dataset["params"].mean(axis=0).round(6).tolist(),
        "param_stds": dataset["params"].std(axis=0).round(6).tolist(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
