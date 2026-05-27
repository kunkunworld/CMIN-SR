from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_dl.baselines import legendre_spectrum_from_zeta, true_mrw_zeta  # noqa: E402
from mrw_inverse.losses import mrw_total_loss  # noqa: E402
from mrw_inverse.models import CMINRegressor  # noqa: E402


def main() -> None:
    torch.manual_seed(2026)
    batch, length = 4, 512
    x = torch.randn(batch, length)
    h_true = torch.rand(batch, 1) * 0.6 + 0.2
    lambda2_true = torch.rand(batch, 1) * 0.08
    q_grid = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32)
    zeta_true_np = np.stack([true_mrw_zeta(q_grid, float(h_true[i]), float(lambda2_true[i])) for i in range(batch)], axis=0)
    f_true_np = np.stack([legendre_spectrum_from_zeta(q_grid, row)[1] for row in zeta_true_np], axis=0)
    zeta_true = torch.tensor(zeta_true_np, dtype=torch.float32)
    f_true = torch.tensor(f_true_np, dtype=torch.float32)

    model = CMINRegressor()
    output = model(x)
    loss = mrw_total_loss(
        output=output,
        h_true=h_true,
        lambda2_true=lambda2_true,
        zeta_true=zeta_true,
        f_true=f_true,
        log_scales=model.structure_branch.frontend.log_scales,
        log_lags=model.logvol_branch.log_lags,
    )
    report = {
        "import_test": "ok",
        "h_shape": list(output.h_hat.shape),
        "lambda2_shape": list(output.lambda2_hat.shape),
        "zeta_shape": list(output.zeta_hat.shape),
        "alpha_shape": list(output.alpha_hat.shape),
        "f_alpha_shape": list(output.f_alpha_hat.shape),
        "p_mrw_shape": list(output.p_mrw.shape),
        "nan_check": {
            "h_nan": bool(torch.isnan(output.h_hat).any().item()),
            "lambda2_nan": bool(torch.isnan(output.lambda2_hat).any().item()),
            "zeta_nan": bool(torch.isnan(output.zeta_hat).any().item()),
            "alpha_nan": bool(torch.isnan(output.alpha_hat).any().item()),
            "f_alpha_nan": bool(torch.isnan(output.f_alpha_hat).any().item()),
            "loss_nan": bool(torch.isnan(loss.total).any().item()),
        },
        "loss_components": {
            "total": float(loss.total.detach()),
            "param": float(loss.l_param.detach()),
            "spectrum": float(loss.l_spectrum.detach()),
            "scaling": float(loss.l_scaling.detach()),
            "logvol": float(loss.l_logvol.detach()),
            "constraint": float(loss.l_constraint.detach()),
            "contrast": float(loss.l_contrast.detach()),
        },
    }
    out_path = ROOT / "outputs" / "reports" / "inverse_framework_smoke_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
