from __future__ import annotations

import torch
from torch import nn

from mrw_dl.models import CrossScaleGraphAttention, MRWStatisticalFrontend


class StructureFunctionBranch(nn.Module):
    """Extracts log-structure functions and scale slopes.

    The branch is intentionally close to the mathematical object:

        S_q(a) = E |X(t+a) - X(t)|^q

    The learned network is only asked to correct finite-sample distortions after
    these multiscale statistics have been computed.
    """

    def __init__(
        self,
        scales: tuple[int, ...] = (2, 4, 8, 16, 32, 64),
        q_grid: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    ) -> None:
        super().__init__()
        self.frontend = MRWStatisticalFrontend(scales=scales, moment_qs=q_grid)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        stats = self.frontend(x)
        return {
            "log_structure_matrix": stats["moment_surface"].transpose(1, 2),
            "empirical_zeta_estimate": stats["zeta_proxy"],
            "scale_slopes": stats["slope_surface"],
            "q_curvature": stats["q_curvature"],
            "wavelet_cumulants": stats["wavelet_cumulants"],
            "cross_scale_corr": stats["cross_scale_corr"],
            "log_scales": self.frontend.log_scales,
            "centered_log_scales": self.frontend.centered_log_scales,
            "scale_slope_denom": self.frontend.scale_slope_denom,
        }


class ScaleGraphBranch(nn.Module):
    """Scale graph correction layer.

    Each scale is treated as a node carrying multiscale statistics. Message
    passing is used to learn finite-sample consistency corrections across scales,
    rather than using attention as an unstructured black-box trick.
    """

    def __init__(self, num_scales: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = CrossScaleGraphAttention(num_scales=num_scales, hidden_dim=hidden_dim, dropout=dropout)

    def forward(
        self,
        cross_scale_corr: torch.Tensor,
        wavelet_cumulants: torch.Tensor,
        log_scales: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(cross_scale_corr, wavelet_cumulants, log_scales)
