from __future__ import annotations

import torch
from torch import nn


class LogVolCovarianceBranch(nn.Module):
    """MRW log-volatility covariance features for lambda2 inference.

    In MRW, a key intermittency diagnostic is approximately

        Cov(log |ΔX_t|, log |ΔX_{t+tau}|) ≈ C - lambda2 log tau

    This branch computes finite-sample covariance curves and their log-lag slope
    so lambda2 is not inferred solely from opaque neural features.
    """

    def __init__(self, lags: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64), eps: float = 1e-6) -> None:
        super().__init__()
        self.lags = lags
        self.eps = eps
        log_lags = torch.log(torch.tensor(lags, dtype=torch.float32))
        centered = log_lags - log_lags.mean()
        self.register_buffer("log_lags", log_lags)
        self.register_buffer("centered_log_lags", centered)
        self.register_buffer("lag_slope_denom", torch.sum(centered.square()).clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        log_abs = torch.log(x.abs().clamp_min(self.eps))
        centered = log_abs - log_abs.mean(dim=1, keepdim=True)
        covs = []
        for lag in self.lags:
            if lag >= x.shape[1]:
                covs.append(centered.new_zeros((x.shape[0],)))
                continue
            cov = (centered[:, :-lag] * centered[:, lag:]).mean(dim=1)
            covs.append(cov)
        cov_curve = torch.stack(covs, dim=1)
        slope = -torch.sum(
            (cov_curve - cov_curve.mean(dim=1, keepdim=True)) * self.centered_log_lags.view(1, -1),
            dim=1,
            keepdim=True,
        ) / self.lag_slope_denom
        return {
            "log_abs_increment": log_abs,
            "logvol_covariance_curve": cov_curve,
            "logvol_covariance_slope": slope,
            "log_lags": self.log_lags,
        }
