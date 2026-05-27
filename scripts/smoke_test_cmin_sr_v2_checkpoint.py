from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_inverse.models import CMINSRv2Model


def main() -> None:
    ckpt = ROOT / "checkpoints" / "cmin" / "cmin_sr_v2_synthetic.pt"
    if not ckpt.exists():
        result = {"status": "missing_checkpoint", "checkpoint": str(ckpt.relative_to(ROOT))}
    else:
        state = torch.load(ckpt, map_location="cpu")
        model = CMINSRv2Model()
        model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state, strict=False)
        x = torch.randn(2, 512)
        with torch.no_grad():
            out = model(x)
        result = {
            "status": "ok",
            "p_mrw_range": [float(out["p_mrw"].min().item()), float(out["p_mrw"].max().item())],
            "p_mono_range": [float(out["p_mono"].min().item()), float(out["p_mono"].max().item())],
            "gain_range": [float(out["mrw_vs_mono_gain"].min().item()), float(out["mrw_vs_mono_gain"].max().item())],
        }
    out_path = ROOT / "outputs" / "reports" / "cmin_sr_v2_checkpoint_smoke_test.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

