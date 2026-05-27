from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_dl.baselines import improved_result_to_dict, run_improved_baselines_on_sample
from mrw_dl.data import load_dataset


def main() -> None:
    dataset_path = ROOT / "data" / "raw" / "mrw_dataset_wide.npz"
    output_stem = "baseline_results_wide_improved_10"
    if len(sys.argv) >= 2:
        dataset_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        output_stem = sys.argv[2]

    bundle = load_dataset(dataset_path)

    q_vals = np.linspace(0.5, 3.0, 11)
    sf_scales = np.unique(np.logspace(np.log10(8), np.log10(512), 16).astype(int))
    mfdfa_scales = np.unique(np.logspace(np.log10(16), np.log10(1024), 16).astype(int))

    results = []
    for sample_index in range(10):
        result = run_improved_baselines_on_sample(
            dx=bundle.dx[sample_index],
            params=bundle.params[sample_index],
            sample_index=sample_index,
            q_vals=q_vals,
            sf_scales=sf_scales,
            mfdfa_scales=mfdfa_scales,
            mfdfa_order=1,
            min_fit_points=5,
        )
        results.append(improved_result_to_dict(result))

    output_dir = ROOT / "outputs" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_stem}.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary = {
        "dataset_path": str(dataset_path),
        "q_vals": q_vals.round(4).tolist(),
        "avg_metrics": {
            "zeta_mae_sf": float(np.mean([r["metrics"]["zeta_mae_sf"] for r in results])),
            "zeta_mae_mfdfa": float(np.mean([r["metrics"]["zeta_mae_mfdfa"] for r in results])),
            "spectrum_mae_sf": float(np.mean([r["metrics"]["spectrum_mae_sf"] for r in results])),
            "spectrum_mae_mfdfa": float(np.mean([r["metrics"]["spectrum_mae_mfdfa"] for r in results])),
        },
        "saved_to": str(output_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
