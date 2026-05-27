from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mrw_dl.models import TCNBlock

from .logvol_covariance import LogVolCovarianceBranch
from .mrw_decoder import MRWAnalyticDecoder
from .multiscale_features import ScaleGraphBranch, StructureFunctionBranch


@dataclass
class CMINOutput:
    h_hat: torch.Tensor
    lambda2_hat: torch.Tensor
    zeta_hat: torch.Tensor
    alpha_hat: torch.Tensor
    f_alpha_hat: torch.Tensor
    p_mrw: torch.Tensor
    residual_zeta: torch.Tensor
    log_structure_matrix: torch.Tensor
    empirical_zeta: torch.Tensor
    scale_graph_embedding: torch.Tensor
    logvol_covariance_curve: torch.Tensor
    logvol_covariance_slope: torch.Tensor


class CMINRegressor(nn.Module):
    """Constrained Multifractal Inference Network.

    The model treats multifractal spectrum estimation as a physics-constrained
    inverse problem:

        finite noisy path
            -> multiscale statistics
            -> finite-sample neural correction
            -> (H, lambda2)
            -> analytic MRW decoder

    Optional MRW-validity and residual heads are kept small so real-data mismatch
    can be analyzed without turning the model into unconstrained free-form
    spectrum regression.
    """

    def __init__(
        self,
        scales: tuple[int, ...] = (2, 4, 8, 16, 32, 64),
        q_grid: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
        lags: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
        hidden_dim: int = 128,
        dropout: float = 0.1,
        residual_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.q_grid = q_grid
        self.q_len = len(q_grid)
        self.residual_scale = residual_scale

        self.structure_branch = StructureFunctionBranch(scales=scales, q_grid=q_grid)
        self.logvol_branch = LogVolCovarianceBranch(lags=lags)
        self.scale_graph = ScaleGraphBranch(num_scales=len(scales), hidden_dim=hidden_dim, dropout=dropout)
        self.decoder = MRWAnalyticDecoder(q_grid=q_grid)

        raw_channels = 32
        self.raw_stem = nn.Sequential(
            nn.Conv1d(1, raw_channels, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(raw_channels),
            nn.GELU(),
            TCNBlock(raw_channels, dilation=1, dropout=dropout),
            TCNBlock(raw_channels, dilation=2, dropout=dropout),
            TCNBlock(raw_channels, dilation=4, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(raw_channels, hidden_dim),
            nn.GELU(),
        )

        structure_dim = self.q_len * len(scales) + self.q_len + self.q_len
        self.structure_head = nn.Sequential(
            nn.LayerNorm(structure_dim + hidden_dim),
            nn.Linear(structure_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        logvol_dim = len(lags) + 1
        self.logvol_head = nn.Sequential(
            nn.LayerNorm(logvol_dim + hidden_dim),
            nn.Linear(logvol_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.h_head = nn.Linear(hidden_dim * 2, 1)
        self.lambda2_head = nn.Linear(hidden_dim * 2, 1)
        self.validity_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.q_len),
        )

    def _residual_constraint_mask(self, device: torch.device) -> torch.Tensor:
        q = torch.tensor(self.q_grid, dtype=torch.float32, device=device)
        anchor_zero = q
        anchor_two = q - 2.0
        return anchor_zero * anchor_two

    def forward(self, x: torch.Tensor) -> CMINOutput:
        structure = self.structure_branch(x)
        logvol = self.logvol_branch(x)
        graph = self.scale_graph(
            structure["cross_scale_corr"],
            structure["wavelet_cumulants"],
            structure["log_scales"],
        )
        raw = self.raw_stem(x.unsqueeze(1))

        structure_vec = torch.cat(
            [
                structure["log_structure_matrix"].reshape(x.shape[0], -1),
                structure["empirical_zeta_estimate"],
                structure["q_curvature"],
            ],
            dim=1,
        )
        structure_embed = self.structure_head(torch.cat([structure_vec, raw], dim=1))
        logvol_vec = torch.cat([logvol["logvol_covariance_curve"], logvol["logvol_covariance_slope"]], dim=1)
        logvol_embed = self.logvol_head(torch.cat([logvol_vec, raw], dim=1))
        fused = torch.cat([structure_embed, 0.5 * graph + logvol_embed], dim=1)

        decoded = self.decoder(self.h_head(fused), self.lambda2_head(fused))
        p_mrw = torch.sigmoid(self.validity_head(fused))
        residual_raw = self.residual_head(fused) * self._residual_constraint_mask(x.device).view(1, -1)
        residual = self.residual_scale * p_mrw * residual_raw
        zeta_hat = decoded.zeta + residual
        alpha_hat = torch.gradient(zeta_hat, spacing=(decoded.q_grid[0],), dim=1)[0]
        f_alpha_hat = 1.0 + decoded.q_grid * alpha_hat - zeta_hat

        return CMINOutput(
            h_hat=decoded.h,
            lambda2_hat=decoded.lambda2,
            zeta_hat=zeta_hat,
            alpha_hat=alpha_hat,
            f_alpha_hat=f_alpha_hat,
            p_mrw=p_mrw,
            residual_zeta=residual,
            log_structure_matrix=structure["log_structure_matrix"],
            empirical_zeta=structure["empirical_zeta_estimate"],
            scale_graph_embedding=graph,
            logvol_covariance_curve=logvol["logvol_covariance_curve"],
            logvol_covariance_slope=logvol["logvol_covariance_slope"],
        )
