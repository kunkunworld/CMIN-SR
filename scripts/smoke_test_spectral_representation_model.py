from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_inverse.models import SpectralRepresentationModel


def main() -> None:
    model = SpectralRepresentationModel()
    x = torch.randn(3, 512)
    with torch.no_grad():
        out = model(x)
    result = {
        "status": "ok",
        "mode": out["mode"],
        "zeta_shape": list(out["zeta_emp"].shape),
        "lambda2_shape": list(out["lambda2_proj"].shape),
        "p_scaling_range": [float(out["p_scaling"].min().item()), float(out["p_scaling"].max().item())],
        "p_mrw_range": [float(out["p_mrw"].min().item()), float(out["p_mrw"].max().item())],
        "no_nan": bool(torch.isfinite(out["zeta_emp"]).all().item() and torch.isfinite(out["zeta_mrw"]).all().item()),
    }
    out_path = ROOT / "outputs" / "reports" / "spectral_representation_model_smoke_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

