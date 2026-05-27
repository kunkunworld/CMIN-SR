from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_inverse.models import MonofractalProjection


def main() -> None:
    q = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=torch.float32)
    zeta = (0.65 * q).unsqueeze(0)
    proj = MonofractalProjection()
    out = proj(zeta)
    result = {
        "status": "ok",
        "H_mono": float(out.H_mono.squeeze().item()),
        "mono_residual_norm": float(out.mono_residual_norm.squeeze().item()),
        "mono_fit_quality": float(out.mono_fit_quality.squeeze().item()),
    }
    path = ROOT / "outputs" / "reports" / "monofractal_projection_smoke_test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

