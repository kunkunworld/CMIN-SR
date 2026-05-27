from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_inverse.models import CMINSRv3Model


def main() -> None:
    ckpt = ROOT / "checkpoints" / "cmin" / "cmin_sr_calibrated_synthetic.pt"
    if not ckpt.exists():
        result = {"status": "missing_checkpoint", "checkpoint": str(ckpt.relative_to(ROOT))}
    else:
        state = torch.load(ckpt, map_location="cpu")
        model = CMINSRv3Model()
        model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state, strict=False)
        with torch.no_grad():
            out = model(torch.randn(3, 512))
        result = {
            "status": "ok",
            "p_curved_range": [float(out["p_curved"].min()), float(out["p_curved"].max())],
            "p_mrw_range": [float(out["p_mrw"].min()), float(out["p_mrw"].max())],
            "boundary_range": [float(out["boundary_mrw_score"].min()), float(out["boundary_mrw_score"].max())],
        }
    path = ROOT / "outputs" / "reports" / "cmin_sr_boundary_calibrated_checkpoint_smoke_test.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
