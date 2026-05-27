from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mrw_inverse.data import PROCESS_NAME_TO_CODE


@dataclass
class BoundaryCalibrationLossOutput:
    total: torch.Tensor
    l_rank_curved: torch.Tensor
    l_rank_mrw: torch.Tensor
    l_rank_mono: torch.Tensor
    l_residual_margin: torch.Tensor
    l_boundary_smooth: torch.Tensor
    l_decouple: torch.Tensor
    l_curv_reg: torch.Tensor
    l_targets: torch.Tensor


def _zero_like(x: torch.Tensor) -> torch.Tensor:
    return x.new_zeros(())


def _pairwise_rank_loss(score: torch.Tensor, order_value: torch.Tensor, group_id: torch.Tensor, margin_scale: float = 0.05) -> torch.Tensor:
    losses = []
    flat_score = score.view(-1)
    flat_order = order_value.view(-1)
    flat_group = group_id.view(-1)
    for gid in torch.unique(flat_group).tolist():
        idx = torch.nonzero(flat_group == gid, as_tuple=False).view(-1)
        if idx.numel() < 2:
            continue
        s = flat_score[idx]
        o = flat_order[idx]
        diff_o = o.view(-1, 1) - o.view(1, -1)
        mask = diff_o > 1e-6
        if not mask.any():
            continue
        diff_s = s.view(-1, 1) - s.view(1, -1)
        margin = (margin_scale + 0.5 * diff_o.abs()).clamp(0.03, 0.20)
        losses.append(F.relu(margin - diff_s)[mask].mean())
    return torch.stack(losses).mean() if losses else _zero_like(score)


def _opposite_rank_loss(score: torch.Tensor, order_value: torch.Tensor, group_id: torch.Tensor, margin_scale: float = 0.05) -> torch.Tensor:
    return _pairwise_rank_loss(-score, order_value, group_id, margin_scale=margin_scale)


def boundary_calibration_loss(
    outputs: dict[str, torch.Tensor | str],
    process_code: torch.Tensor,
    group_id: torch.Tensor,
    lambda2_true: torch.Tensor,
    rank_curvature_target: torch.Tensor,
    target_p_scaling: torch.Tensor,
    target_p_curved: torch.Tensor,
    target_p_mrw: torch.Tensor,
    target_p_mono: torch.Tensor,
    target_boundary_mrw: torch.Tensor,
    w_rank_curved: float = 1.0,
    w_rank_mrw: float = 1.0,
    w_rank_mono: float = 0.5,
    w_residual_margin: float = 1.0,
    w_boundary_smooth: float = 0.5,
    w_decouple: float = 1.0,
    w_curv_reg: float = 0.5,
) -> BoundaryCalibrationLossOutput:
    p_scaling = torch.nan_to_num(outputs["p_scaling"], nan=0.0).clamp(0.0, 1.0)
    p_curved = torch.nan_to_num(outputs["p_curved"], nan=0.0).clamp(0.0, 1.0)
    p_mrw = torch.nan_to_num(outputs["p_mrw"], nan=0.0).clamp(0.0, 1.0)
    p_mono = torch.nan_to_num(outputs["p_mono"], nan=0.0).clamp(0.0, 1.0)
    boundary_score = torch.nan_to_num(outputs["boundary_mrw_score"], nan=0.0).clamp(0.0, 1.0)
    residual_norm = torch.nan_to_num(outputs["residual_norm"], nan=1.0).clamp_min(0.0)
    mono_residual_norm = torch.nan_to_num(outputs["mono_residual_norm"], nan=1.0).clamp_min(0.0)
    curvature_score = torch.nan_to_num(outputs["curvature_score"], nan=0.0).clamp(0.0, 1.0)

    codes = process_code.view(-1)
    lambda_flat = torch.nan_to_num(lambda2_true.view(-1), nan=0.0)
    strong_mrw = (codes == PROCESS_NAME_TO_CODE["MRW"]) & (lambda_flat >= 0.06)
    boundary_mrw = ((codes == PROCESS_NAME_TO_CODE["MRW"]) | (codes == PROCESS_NAME_TO_CODE["Low-lambda2 MRW"])) & (lambda_flat < 0.04)
    mono = (codes == PROCESS_NAME_TO_CODE["fGn"]) | (codes == PROCESS_NAME_TO_CODE["iid Gaussian"])

    l_rank_curved = _pairwise_rank_loss(p_curved, rank_curvature_target, group_id, margin_scale=0.05)
    l_rank_mrw = _pairwise_rank_loss(p_mrw, rank_curvature_target, group_id, margin_scale=0.05)
    l_rank_mono = _opposite_rank_loss(p_mono, rank_curvature_target, group_id, margin_scale=0.04)

    l_residual_margin = _zero_like(p_curved)
    if mono.any():
        mask = mono.view(-1, 1)
        l_residual_margin = l_residual_margin + F.relu(mono_residual_norm - residual_norm - 0.01)[mask].mean()
    if strong_mrw.any():
        mask = strong_mrw.view(-1, 1)
        l_residual_margin = l_residual_margin + F.relu(0.02 - (mono_residual_norm - residual_norm))[mask].mean()

    l_boundary_smooth = F.mse_loss(boundary_score, target_boundary_mrw)
    if boundary_mrw.any():
        mask = boundary_mrw.view(-1, 1)
        l_boundary_smooth = l_boundary_smooth + 0.5 * F.relu(0.45 - boundary_score[mask]).mean()
    if strong_mrw.any():
        mask = strong_mrw.view(-1, 1)
        l_boundary_smooth = l_boundary_smooth + 0.5 * F.relu(boundary_score[mask] - 0.35).mean()

    l_decouple = F.mse_loss(p_scaling, target_p_scaling)
    if mono.any():
        mask = mono.view(-1, 1)
        l_decouple = l_decouple + F.relu(p_curved[mask] - 0.30).mean() + F.relu(p_mrw[mask] - 0.45).mean()

    lambda_curv_target = (lambda2_true / 0.10).clamp(0.0, 1.0)
    l_curv_reg = F.mse_loss(p_curved, target_p_curved) + 0.25 * F.mse_loss(curvature_score, lambda_curv_target)

    l_targets = (
        F.mse_loss(p_curved, target_p_curved)
        + F.mse_loss(p_mrw, target_p_mrw)
        + F.mse_loss(p_mono, target_p_mono)
        + F.mse_loss(boundary_score, target_boundary_mrw)
    )

    total = (
        w_rank_curved * l_rank_curved
        + w_rank_mrw * l_rank_mrw
        + w_rank_mono * l_rank_mono
        + w_residual_margin * l_residual_margin
        + w_boundary_smooth * l_boundary_smooth
        + w_decouple * l_decouple
        + w_curv_reg * l_curv_reg
        + l_targets
    )
    total = torch.nan_to_num(total, nan=0.0, posinf=1e3, neginf=0.0)
    return BoundaryCalibrationLossOutput(
        total=total,
        l_rank_curved=torch.nan_to_num(l_rank_curved),
        l_rank_mrw=torch.nan_to_num(l_rank_mrw),
        l_rank_mono=torch.nan_to_num(l_rank_mono),
        l_residual_margin=torch.nan_to_num(l_residual_margin),
        l_boundary_smooth=torch.nan_to_num(l_boundary_smooth),
        l_decouple=torch.nan_to_num(l_decouple),
        l_curv_reg=torch.nan_to_num(l_curv_reg),
        l_targets=torch.nan_to_num(l_targets),
    )
