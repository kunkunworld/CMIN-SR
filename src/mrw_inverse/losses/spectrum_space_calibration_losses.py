from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mrw_inverse.data import SPECTRUM_TYPES


@dataclass
class SpectrumSpaceCalibrationLossOutput:
    total: torch.Tensor
    l_scores: torch.Tensor
    l_rank: torch.Tensor
    l_mono: torch.Tensor
    l_curved: torch.Tensor
    l_caution: torch.Tensor
    l_boundary: torch.Tensor


def _rank_loss(score: torch.Tensor, order: torch.Tensor, margin: float = 0.03, increasing: bool = True) -> torch.Tensor:
    s = score.view(-1)
    o = order.view(-1)
    diff_o = o.view(-1, 1) - o.view(1, -1)
    mask = diff_o > 1e-6
    if not mask.any():
        return score.new_zeros(())
    diff_s = s.view(-1, 1) - s.view(1, -1)
    if not increasing:
        diff_s = -diff_s
    dyn_margin = (margin + 0.4 * diff_o.abs()).clamp(0.02, 0.18)
    return F.relu(dyn_margin - diff_s)[mask].mean()


def spectrum_space_calibration_loss(
    outputs: dict[str, torch.Tensor],
    spectrum_code: torch.Tensor,
    lambda2_true: torch.Tensor,
    target_p_scaling: torch.Tensor,
    target_p_curved: torch.Tensor,
    target_p_mono: torch.Tensor,
    target_p_mrw: torch.Tensor,
    target_p_boundary: torch.Tensor,
    w_scores: float = 1.0,
    w_rank: float = 1.0,
    w_mono: float = 1.0,
    w_curved: float = 1.0,
    w_caution: float = 1.0,
    w_boundary: float = 0.5,
) -> SpectrumSpaceCalibrationLossOutput:
    p_scaling = outputs["p_scaling"]
    p_curved = outputs["p_curved"]
    p_mono = outputs["p_mono"]
    p_mrw = outputs["p_mrw"]
    p_boundary = outputs["p_boundary"]
    code = spectrum_code.view(-1)

    l_scores = (
        F.mse_loss(p_scaling, target_p_scaling)
        + F.mse_loss(p_curved, target_p_curved)
        + F.mse_loss(p_mono, target_p_mono)
        + F.mse_loss(p_mrw, target_p_mrw)
        + F.mse_loss(p_boundary, target_p_boundary)
    )

    mrw_codes = [SPECTRUM_TYPES.index(x) for x in ("boundary_mrw", "curved_mrw", "noisy_mrw", "ambiguous_mild_curvature")]
    mrw_mask = torch.zeros_like(code, dtype=torch.bool)
    for c in mrw_codes:
        mrw_mask |= code == c
    l_rank = p_curved.new_zeros(())
    if mrw_mask.any():
        l_rank = (
            _rank_loss(p_curved[mrw_mask], lambda2_true[mrw_mask], increasing=True)
            + _rank_loss(p_mrw[mrw_mask], lambda2_true[mrw_mask], increasing=True)
            + _rank_loss(p_mono[mrw_mask], lambda2_true[mrw_mask], increasing=False)
        )

    mono_mask = (code == SPECTRUM_TYPES.index("linear_mono")) | (code == SPECTRUM_TYPES.index("noisy_mono"))
    curved_mask = code == SPECTRUM_TYPES.index("curved_mrw")
    caution_mask = (code == SPECTRUM_TYPES.index("heavy_tail_distorted")) | (code == SPECTRUM_TYPES.index("regime_apparent"))
    boundary_mask = code == SPECTRUM_TYPES.index("boundary_mrw")

    l_mono = p_curved.new_zeros(())
    if mono_mask.any():
        m = mono_mask.view(-1, 1)
        l_mono = F.relu(p_curved[m] - 0.25).mean() + F.relu(p_mrw[m] - 0.30).mean() + F.relu(0.75 - p_mono[m]).mean()

    l_curved = p_curved.new_zeros(())
    if curved_mask.any():
        m = curved_mask.view(-1, 1)
        l_curved = F.relu(0.75 - p_curved[m]).mean() + F.relu(0.80 - p_mrw[m]).mean() + F.relu(p_mono[m] - 0.35).mean()

    l_caution = p_curved.new_zeros(())
    if caution_mask.any():
        m = caution_mask.view(-1, 1)
        l_caution = F.relu(p_mrw[m] - 0.25).mean()

    l_boundary = F.mse_loss(p_boundary, target_p_boundary)
    if boundary_mask.any():
        m = boundary_mask.view(-1, 1)
        l_boundary = l_boundary + F.relu(0.65 - p_boundary[m]).mean() + F.relu(torch.abs(p_mrw[m] - 0.50) - 0.25).mean()

    total = (
        w_scores * l_scores
        + w_rank * l_rank
        + w_mono * l_mono
        + w_curved * l_curved
        + w_caution * l_caution
        + w_boundary * l_boundary
    )
    return SpectrumSpaceCalibrationLossOutput(
        total=torch.nan_to_num(total),
        l_scores=torch.nan_to_num(l_scores),
        l_rank=torch.nan_to_num(l_rank),
        l_mono=torch.nan_to_num(l_mono),
        l_curved=torch.nan_to_num(l_curved),
        l_caution=torch.nan_to_num(l_caution),
        l_boundary=torch.nan_to_num(l_boundary),
    )
