from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_dl.baselines import result_to_dict, run_baselines_on_sample
from mrw_dl.data import load_dataset


def main() -> None:
    dataset_path = ROOT / "data" / "raw" / "mrw_dataset.npz"
    output_stem = "baseline_results_10"
    if len(sys.argv) >= 2:
        dataset_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        output_stem = sys.argv[2]

    bundle = load_dataset(dataset_path)

    q_vals = np.linspace(-2.0, 2.0, 9)
    q_vals = q_vals[np.abs(q_vals) > 1e-12]

    sf_scales = np.unique(np.logspace(np.log10(4), np.log10(256), 18).astype(int))
    mfdfa_scales = np.unique(np.logspace(np.log10(16), np.log10(512), 18).astype(int))

    results = []
    for sample_index in range(10):
        result = run_baselines_on_sample(
            dx=bundle.dx[sample_index],
            params=bundle.params[sample_index],
            sample_index=sample_index,
            q_vals=q_vals,
            sf_scales=sf_scales,
            mfdfa_scales=mfdfa_scales,
            fit_slice_sf=slice(2, -2),
            fit_slice_mfdfa=slice(2, -2),
            mfdfa_order=1,
        )
        results.append(result_to_dict(result))

    output_dir = ROOT / "outputs" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_stem}.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report_path = output_dir / f"{output_stem}.txt"

    report_lines = []
    report_lines.append(f"dataset_path: {dataset_path}")
    report_lines.append("")
    for item in results:
        report_lines.append(f"Sample {item['sample_index']}")
        report_lines.append(f"params: {item['params']}")
        report_lines.append(f"q_vals: {item['q_vals']}")
        report_lines.append(f"f_true: {item['f_true']}")
        report_lines.append(f"f_sf: {item['f_sf']}")
        report_lines.append(f"f_mfdfa: {item['f_mfdfa']}")
        report_lines.append(f"metrics: {item['metrics']}")
        report_lines.append("")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"\nSaved to: {output_path}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
