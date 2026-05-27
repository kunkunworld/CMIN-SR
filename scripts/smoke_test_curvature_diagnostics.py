from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_inverse.models import compute_curvature_diagnostics


def main() -> None:
    q = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=torch.float32)
    zeta_emp = torch.tensor([[0.30, 0.62, 0.95, 1.27, 1.57, 1.86]], dtype=torch.float32)
    zeta_mono = torch.tensor([[0.30, 0.60, 0.90, 1.20, 1.50, 1.80]], dtype=torch.float32)
    zeta_mrw = torch.tensor([[0.31, 0.62, 0.93, 1.23, 1.52, 1.80]], dtype=torch.float32)
    out = compute_curvature_diagnostics(
        q_grid=q,
        zeta_emp=zeta_emp,
        zeta_mono=zeta_mono,
        zeta_mrw=zeta_mrw,
        residual_norm=torch.tensor([[0.03]], dtype=torch.float32),
        mono_residual_norm=torch.tensor([[0.08]], dtype=torch.float32),
        lambda2_proj=torch.tensor([[0.06]], dtype=torch.float32),
    )
    result = {
        "status": "ok",
        "curvature_score": float(out.curvature_score.item()),
        "linearity_score": float(out.linearity_score.item()),
        "curvature_confidence": float(out.curvature_confidence.item()),
        "boundary_mrw_score": float(out.boundary_mrw_score.item()),
    }
    path = ROOT / "outputs" / "reports" / "curvature_diagnostics_smoke_test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
