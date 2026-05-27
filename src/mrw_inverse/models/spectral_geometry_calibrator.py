from __future__ import annotations

import torch
from torch import nn


class SpectralGeometryCalibrator(nn.Module):
    """Small spectrum-space interpretation head for zeta geometry."""

    def __init__(self, q_len: int = 6, hidden_dim: int = 128, dropout: float = 0.05) -> None:
        super().__init__()
        self.q_len = q_len
        input_dim = q_len * 3 + 8
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 7),
        )

    def forward(
        self,
        zeta_input: torch.Tensor,
        zeta_mono: torch.Tensor,
        zeta_mrw: torch.Tensor,
        mono_residual_norm: torch.Tensor,
        mrw_residual_norm: torch.Tensor,
        mrw_vs_mono_gain: torch.Tensor,
        normalized_mrw_gain: torch.Tensor,
        curvature_score: torch.Tensor,
        linearity_score: torch.Tensor,
        boundary_score: torch.Tensor,
        tail_instability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if tail_instability is None:
            tail_instability = zeta_input.new_zeros((zeta_input.shape[0], 1))
        parts = [
            torch.nan_to_num(zeta_input),
            torch.nan_to_num(zeta_mono),
            torch.nan_to_num(zeta_mrw),
            torch.nan_to_num(mono_residual_norm),
            torch.nan_to_num(mrw_residual_norm),
            torch.nan_to_num(mrw_vs_mono_gain),
            torch.nan_to_num(normalized_mrw_gain),
            torch.nan_to_num(curvature_score),
            torch.nan_to_num(linearity_score),
            torch.nan_to_num(boundary_score),
            torch.nan_to_num(tail_instability),
        ]
        logits = self.net(torch.cat(parts, dim=1))
        scores = torch.sigmoid(logits).clamp(0.0, 1.0)
        return {
            "p_scaling": scores[:, 0:1],
            "p_curved": scores[:, 1:2],
            "p_mono": scores[:, 2:3],
            "p_mrw": scores[:, 3:4],
            "p_boundary": scores[:, 4:5],
            "calibrated_mrw_score": scores[:, 5:6],
            "calibrated_curvature_score": scores[:, 6:7],
        }
