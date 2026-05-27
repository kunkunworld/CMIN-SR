from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MRWDecodedSpectrum:
    h: torch.Tensor
    lambda2: torch.Tensor
    q_grid: torch.Tensor
    zeta: torch.Tensor
    alpha: torch.Tensor
    f_alpha: torch.Tensor


class MRWAnalyticDecoder(nn.Module):
    """Hard physics layer for MRW parameter-to-spectrum decoding.

    This enforces the MRW parametric form instead of allowing free-form spectrum
    regression:

        zeta(q) = q H - 0.5 lambda2 q (q - 2)
        alpha(q) = d zeta / d q = H - lambda2 (q - 1)
        f(alpha) = 1 + q alpha - zeta(q)
    """

    def __init__(self, q_grid: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)) -> None:
        super().__init__()
        self.register_buffer("q_grid", torch.tensor(q_grid, dtype=torch.float32))

    def forward(self, h_raw: torch.Tensor, lambda2_raw: torch.Tensor) -> MRWDecodedSpectrum:
        h = torch.sigmoid(h_raw).clamp(1e-4, 1.0 - 1e-4)
        lambda2 = torch.nn.functional.softplus(lambda2_raw)
        q = self.q_grid.view(1, -1)
        zeta = q * h - 0.5 * lambda2 * q * (q - 2.0)
        alpha = h - lambda2 * (q - 1.0)
        f_alpha = 1.0 + q * alpha - zeta
        return MRWDecodedSpectrum(h=h, lambda2=lambda2, q_grid=q.expand_as(zeta), zeta=zeta, alpha=alpha, f_alpha=f_alpha)
