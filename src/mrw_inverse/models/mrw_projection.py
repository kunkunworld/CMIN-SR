from __future__ import annotations

from dataclasses import dataclass

import torch

from .mrw_decoder import MRWAnalyticDecoder


@dataclass
class MRWProjectionOutput:
    H_proj: torch.Tensor
    lambda2_proj: torch.Tensor
    zeta_mrw_proj: torch.Tensor
    alpha_mrw_proj: torch.Tensor
    f_mrw_proj: torch.Tensor
    residual: torch.Tensor
    residual_norm: torch.Tensor
    projection_r2: torch.Tensor
    fit_quality: torch.Tensor
    linear_residual_norm: torch.Tensor
    projection_gain: torch.Tensor


class MRWProjection(torch.nn.Module):
    def __init__(self, q_grid: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)) -> None:
        super().__init__()
        self.decoder = MRWAnalyticDecoder(q_grid=q_grid)
        q = self.decoder.q_grid
        design = torch.stack([q, -0.5 * q * (q - 2.0)], dim=1)
        self.register_buffer("design", design)

    def forward(self, zeta_emp: torch.Tensor) -> MRWProjectionOutput:
        if zeta_emp.ndim == 1:
            zeta_emp = zeta_emp.unsqueeze(0)
        design = self.design.to(dtype=zeta_emp.dtype, device=zeta_emp.device)
        coef = torch.linalg.pinv(design).matmul(zeta_emp.transpose(0, 1)).transpose(0, 1)
        h_raw = coef[:, :1]
        lambda2_raw = coef[:, 1:2].clamp_min(0.0)
        decoded = self.decoder(torch.logit(h_raw.clamp(1e-4, 1 - 1e-4)), torch.log(torch.expm1(lambda2_raw.clamp_min(1e-6))))
        residual = zeta_emp - decoded.zeta
        residual_norm = residual.abs().mean(dim=1, keepdim=True)

        q = self.decoder.q_grid.to(dtype=zeta_emp.dtype, device=zeta_emp.device).view(1, -1)
        h_linear = ((zeta_emp * q).sum(dim=1, keepdim=True) / q.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-8)).clamp(0.0, 1.0)
        zeta_linear = h_linear * q
        linear_residual_norm = (zeta_emp - zeta_linear).abs().mean(dim=1, keepdim=True)
        projection_gain = ((linear_residual_norm - residual_norm) / linear_residual_norm.clamp_min(1e-6)).clamp(0.0, 1.0)

        ss_res = (residual ** 2).sum(dim=1, keepdim=True)
        centered = zeta_emp - zeta_emp.mean(dim=1, keepdim=True)
        ss_tot = (centered ** 2).sum(dim=1, keepdim=True).clamp_min(1e-8)
        r2 = (1.0 - ss_res / ss_tot).clamp(-5.0, 1.0)
        fit_quality = (0.7 * r2.clamp(0.0, 1.0) + 0.3 * torch.exp(-6.0 * residual_norm)).clamp(0.0, 1.0)
        return MRWProjectionOutput(
            H_proj=decoded.h,
            lambda2_proj=decoded.lambda2,
            zeta_mrw_proj=decoded.zeta,
            alpha_mrw_proj=decoded.alpha,
            f_mrw_proj=decoded.f_alpha,
            residual=residual,
            residual_norm=residual_norm,
            projection_r2=r2,
            fit_quality=fit_quality,
            linear_residual_norm=linear_residual_norm,
            projection_gain=projection_gain,
        )
