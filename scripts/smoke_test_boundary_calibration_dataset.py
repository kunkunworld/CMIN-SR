from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_inverse.data import BoundaryCalibrationDatasetConfig, generate_boundary_calibration_dataset


def main() -> None:
    ds = generate_boundary_calibration_dataset(BoundaryCalibrationDatasetConfig(length=256, num_groups=3, seed=7))
    result = {
        "status": "ok",
        "x_shape": list(ds["x"].shape),
        "num_groups": int(len(set(ds["group_id"].tolist()))),
        "lambda2_levels": sorted(set(float(x) for x in ds["lambda2_true"].reshape(-1).tolist())),
        "classes": sorted(set(ds["target_curvature_class"].tolist())),
    }
    path = ROOT / "outputs" / "reports" / "boundary_calibration_dataset_smoke_test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
