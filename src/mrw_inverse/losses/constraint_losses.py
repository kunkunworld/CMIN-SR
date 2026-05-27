from __future__ import annotations

import torch
import torch.nn.functional as F


def concavity_penalty(zeta: torch.Tensor) -> torch.Tensor:
    if zeta.shape[1] < 3:
        return zeta.new_zeros(())
    second_diff = zeta[:, 2:] - 2.0 * zeta[:, 1:-1] + zeta[:, :-2]
    return F.relu(second_diff).square().mean()


def residual_smallness_penalty(residual: torch.Tensor) -> torch.Tensor:
    return residual.square().mean()


def mrw_parameter_box_penalty(h_hat: torch.Tensor, lambda2_hat: torch.Tensor) -> torch.Tensor:
    # H and lambda2 are already transformed into valid ranges by the decoder, so
    # this acts mainly as a soft barrier away from the extremes.
    h_pen = F.relu(0.02 - h_hat).mean() + F.relu(h_hat - 0.98).mean()
    lambda_pen = F.relu(-lambda2_hat).mean()
    return h_pen + lambda_pen
