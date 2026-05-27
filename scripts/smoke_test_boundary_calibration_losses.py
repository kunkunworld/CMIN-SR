from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_inverse.data import BoundaryCalibrationDatasetConfig, generate_boundary_calibration_dataset
from mrw_inverse.losses import boundary_calibration_loss
from mrw_inverse.models import CMINSRv3Model


def main() -> None:
    ds = generate_boundary_calibration_dataset(BoundaryCalibrationDatasetConfig(length=256, num_groups=2, seed=9))
    model = CMINSRv3Model()
    x = torch.tensor(ds["x"], dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
    loss = boundary_calibration_loss(
        out,
        torch.tensor(ds["process_code"], dtype=torch.long),
        torch.tensor(ds["group_id"], dtype=torch.long),
        torch.tensor(ds["lambda2_true"], dtype=torch.float32),
        torch.tensor(ds["rank_curvature_target"], dtype=torch.float32),
        torch.tensor(ds["target_p_scaling"], dtype=torch.float32),
        torch.tensor(ds["target_p_curved"], dtype=torch.float32),
        torch.tensor(ds["target_p_mrw"], dtype=torch.float32),
        torch.tensor(ds["target_p_mono"], dtype=torch.float32),
        torch.tensor(ds["target_boundary_mrw"], dtype=torch.float32),
    )
    result = {"status": "ok", "loss": float(loss.total.item()), "isfinite": bool(torch.isfinite(loss.total).item())}
    path = ROOT / "outputs" / "reports" / "boundary_calibration_losses_smoke_test.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
