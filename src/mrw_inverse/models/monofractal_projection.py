from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MonofractalProjectionOutput:
    H_mono: torch.Tensor
    zeta_mono: torch.Tensor
    mono_residual: torch.Tensor
    mono_residual_norm: torch.Tensor
    mono_fit_quality: torch.Tensor


class MonofractalProjection(torch.nn.Module):
    """Fits the best linear monofractal spectrum zeta(q)=qH."""

    def __init__(self, q_grid: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)) -> None:
        super().__init__()
        self.register_buffer("q_grid", torch.tensor(q_grid, dtype=torch.float32))

    def forward(self, zeta_emp: torch.Tensor) -> MonofractalProjectionOutput:
        if zeta_emp.ndim == 1:
            zeta_emp = zeta_emp.unsqueeze(0)
        q = self.q_grid.to(device=zeta_emp.device, dtype=zeta_emp.dtype).view(1, -1)
        h = ((zeta_emp * q).sum(dim=1, keepdim=True) / q.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-8)).clamp(0.0, 1.0)
        zeta_mono = h * q
        residual = zeta_emp - zeta_mono
        residual_norm = residual.abs().mean(dim=1, keepdim=True)
        ss_res = (residual ** 2).sum(dim=1, keepdim=True)
        centered = zeta_emp - zeta_emp.mean(dim=1, keepdim=True)
        ss_tot = (centered ** 2).sum(dim=1, keepdim=True).clamp_min(1e-8)
        r2 = (1.0 - ss_res / ss_tot).clamp(-5.0, 1.0)
        fit_quality = (0.7 * r2.clamp(0.0, 1.0) + 0.3 * torch.exp(-6.0 * residual_norm)).clamp(0.0, 1.0)
        return MonofractalProjectionOutput(
            H_mono=h,
            zeta_mono=zeta_mono,
            mono_residual=residual,
            mono_residual_norm=residual_norm,
            mono_fit_quality=fit_quality,
        )

