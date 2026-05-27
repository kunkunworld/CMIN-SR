from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_inverse.models import CMINSRv2Model, CMINSRv3Model


def main() -> None:
    ckpt = ROOT / "checkpoints" / "cmin" / "cmin_sr_v3_synthetic.pt"
    if not ckpt.exists():
        result = {"status": "missing_checkpoint", "checkpoint": str(ckpt.relative_to(ROOT))}
    else:
        state = torch.load(ckpt, map_location="cpu")
        model = CMINSRv3Model()
        model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state, strict=False)
        x = torch.randn(2, 512)
        with torch.no_grad():
            out = model(x)
        v2_load_ok = True
        v2_ckpt = ROOT / "checkpoints" / "cmin" / "cmin_sr_v2_synthetic.pt"
        if v2_ckpt.exists():
            try:
                v2_state = torch.load(v2_ckpt, map_location="cpu")
                _ = model.load_state_dict(v2_state["model_state_dict"] if isinstance(v2_state, dict) and "model_state_dict" in v2_state else v2_state, strict=False)
            except Exception:
                v2_load_ok = False
        _ = CMINSRv2Model()
        result = {
            "status": "ok",
            "p_curved_range": [float(out["p_curved"].min().item()), float(out["p_curved"].max().item())],
            "p_mrw_range": [float(out["p_mrw"].min().item()), float(out["p_mrw"].max().item())],
            "boundary_range": [float(out["boundary_mrw_score"].min().item()), float(out["boundary_mrw_score"].max().item())],
            "v2_checkpoint_loadable_strict_false": v2_load_ok,
        }
    out_path = ROOT / "outputs" / "reports" / "cmin_sr_v3_checkpoint_smoke_test.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
