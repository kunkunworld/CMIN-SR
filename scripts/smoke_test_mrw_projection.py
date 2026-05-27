from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_inverse.models import MRWProjection


def main() -> None:
    zeta = torch.tensor([[0.28, 0.55, 0.81, 1.05, 1.28, 1.50]], dtype=torch.float32)
    proj = MRWProjection()
    out = proj(zeta)
    result = {
        "status": "ok",
        "H_proj": float(out.H_proj.squeeze().item()),
        "lambda2_proj": float(out.lambda2_proj.squeeze().item()),
        "residual_norm": float(out.residual_norm.squeeze().item()),
        "projection_r2": float(out.projection_r2.squeeze().item()),
        "no_nan": bool(torch.isfinite(out.zeta_mrw_proj).all().item()),
    }
    out_path = ROOT / "outputs" / "reports" / "mrw_projection_smoke_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

