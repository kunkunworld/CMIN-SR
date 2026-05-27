from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_inverse.models import estimate_empirical_spectrum


def main() -> None:
    x = torch.randn(4, 512)
    out = estimate_empirical_spectrum(x)
    result = {
        "status": "ok",
        "zeta_shape": list(out.zeta_emp.shape),
        "no_nan_zeta": bool(torch.isfinite(out.zeta_emp).all().item()),
        "p_scaling_min": float(out.p_scaling.min().item()),
        "p_scaling_max": float(out.p_scaling.max().item()),
        "stability_min": float(out.spectrum_stability.min().item()),
        "stability_max": float(out.spectrum_stability.max().item()),
    }
    out_path = ROOT / "outputs" / "reports" / "empirical_spectrum_smoke_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

