from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mrw_inverse.data import PROCESS_NAME_TO_CODE

from .constraint_losses import concavity_penalty, mrw_parameter_box_penalty


@dataclass
class MonofractalCalibrationLossOutput:
    total: torch.Tensor
    l_zeta: torch.Tensor
    l_mrw: torch.Tensor
    l_mono: torch.Tensor
    l_validity: torch.Tensor
    l_residual: torch.Tensor
    l_surrogate: torch.Tensor
    l_curvature: torch.Tensor
    l_constraint: torch.Tensor


def _masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return pred.new_zeros(())
    return torch.abs(pred[mask] - target[mask]).mean()


def _pairwise_boundary_losses(
    process_code: torch.Tensor,
    pair_id: torch.Tensor,
    p_mrw: torch.Tensor,
    mrw_residual: torch.Tensor,
    mono_residual: torch.Tensor,
    mrw_gain: torch.Tensor,
    lambda2_proj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p_losses = []
    residual_losses = []
    gain_losses = []
    codes = process_code.view(-1)
    pairs = pair_id.view(-1)
    for pid in torch.unique(pairs[pairs >= 0]).tolist():
        idx = torch.nonzero(pairs == pid, as_tuple=False).view(-1)
        if idx.numel() < 2:
            continue
        mrw_idx = idx[codes[idx] == PROCESS_NAME_TO_CODE["MRW"]]
        shuf_idx = idx[codes[idx] == PROCESS_NAME_TO_CODE["Shuffled MRW"]]
        if mrw_idx.numel() == 0 or shuf_idx.numel() == 0:
            continue
        i = mrw_idx[0]
        j = shuf_idx[0]
        p_losses.append(F.relu(0.15 - (p_mrw[i] - p_mrw[j])))
        residual_losses.append(F.relu(0.01 - (mrw_residual[j] - mrw_residual[i])))
        gain_losses.append(F.relu(0.02 - (mrw_gain[i] - mrw_gain[j])))
    zero = p_mrw.new_zeros(())
    return (
        torch.stack(p_losses).mean() if p_losses else zero,
        torch.stack(residual_losses).mean() if residual_losses else zero,
        torch.stack(gain_losses).mean() if gain_losses else zero,
    )


def cmin_sr_v2_loss(
    outputs: dict[str, torch.Tensor | str],
    process_code: torch.Tensor,
    zeta_target: torch.Tensor,
    h_true: torch.Tensor,
    lambda2_true: torch.Tensor,
    target_p_scaling: torch.Tensor,
    target_p_mrw: torch.Tensor,
    target_p_mono: torch.Tensor,
    target_residual_level: torch.Tensor,
    target_stability: torch.Tensor,
    target_curvature_significance: torch.Tensor,
    target_mrw_vs_mono_gain: torch.Tensor,
    pair_id: torch.Tensor,
    w_zeta: float = 1.0,
    w_mrw_proj: float = 1.0,
    w_mono: float = 0.7,
    w_validity: float = 1.0,
    w_residual: float = 0.5,
    w_surrogate: float = 1.0,
    w_stability: float = 0.2,
    w_constraint: float = 0.1,
) -> MonofractalCalibrationLossOutput:
    zeta_emp = outputs["zeta_emp"]
    H_proj = outputs["H_proj"]
    lambda2_proj = outputs["lambda2_proj"]
    residual_norm = outputs["residual_norm"]
    p_scaling = outputs["p_scaling"]
    p_mrw = outputs["p_mrw"]
    p_mono = outputs["p_mono"]
    spectrum_stability = outputs["spectrum_stability"]
    H_mono = outputs["H_mono"]
    zeta_mono = outputs["zeta_mono"]
    mono_residual_norm = outputs["mono_residual_norm"]
    mono_fit_quality = outputs["mono_fit_quality"]
    mrw_vs_mono_gain = outputs["mrw_vs_mono_gain"]
    curvature_significance = outputs["curvature_significance"]

    codes = process_code.view(-1)
    zero = zeta_emp.new_zeros(())
    valid_zeta = ~torch.isnan(zeta_target).any(dim=1)
    mrw_mask = (codes == PROCESS_NAME_TO_CODE["MRW"]) | (codes == PROCESS_NAME_TO_CODE["Low-lambda2 MRW"])
    mono_mask = (codes == PROCESS_NAME_TO_CODE["fGn"]) | (codes == PROCESS_NAME_TO_CODE["iid Gaussian"])

    l_zeta = _masked_mae(zeta_emp, zeta_target, valid_zeta.view(-1, 1).expand_as(zeta_emp))

    l_mrw = zero
    if mrw_mask.any():
        l_mrw = (
            _masked_mae(H_proj, h_true, mrw_mask.view(-1, 1))
            + _masked_mae(lambda2_proj, lambda2_true, mrw_mask.view(-1, 1))
            + _masked_mae(residual_norm, target_residual_level, mrw_mask.view(-1, 1))
            + _masked_mae(p_mrw, target_p_mrw, mrw_mask.view(-1, 1))
            + _masked_mae(mrw_vs_mono_gain, target_mrw_vs_mono_gain, mrw_mask.view(-1, 1))
        )

    l_mono = zero
    if mono_mask.any():
        mono_h_mask = mono_mask & torch.isfinite(h_true.view(-1))
        mono_margin = F.relu(0.01 - (residual_norm - mono_residual_norm))[mono_mask.view(-1, 1)].mean()
        mono_prob_gap = F.relu(0.10 - (p_mono - p_mrw))[mono_mask.view(-1, 1)].mean()
        l_mono = (
            _masked_mae(H_mono, h_true, mono_h_mask.view(-1, 1))
            + _masked_mae(zeta_mono, zeta_target, mono_mask.view(-1, 1).expand_as(zeta_mono))
            + _masked_mae(mono_residual_norm, target_residual_level, mono_mask.view(-1, 1))
            + _masked_mae(p_mono, target_p_mono, mono_mask.view(-1, 1))
            + _masked_mae(curvature_significance, target_curvature_significance, mono_mask.view(-1, 1))
            + mono_margin
            + mono_prob_gap
        )

    mrw_competition = zero
    strong_mrw_mask = mrw_mask & torch.isfinite(lambda2_true.view(-1)) & (lambda2_true.view(-1) >= 0.04)
    boundary_mrw_mask = mrw_mask & torch.isfinite(lambda2_true.view(-1)) & (lambda2_true.view(-1) < 0.04)
    if strong_mrw_mask.any():
        mrw_margin = F.relu(0.02 - (mono_residual_norm - residual_norm))[strong_mrw_mask.view(-1, 1)].mean()
        mrw_prob_gap = F.relu(0.10 - (p_mrw - p_mono))[strong_mrw_mask.view(-1, 1)].mean()
        mrw_competition = mrw_margin + mrw_prob_gap
    if boundary_mrw_mask.any():
        boundary_gap = torch.abs((p_mrw - p_mono)[boundary_mrw_mask.view(-1, 1)]).mean()
        mrw_competition = mrw_competition + 0.5 * boundary_gap

    l_validity = (
        F.mse_loss(p_scaling, target_p_scaling)
        + F.mse_loss(p_mrw, target_p_mrw)
        + F.mse_loss(p_mono, target_p_mono)
    )

    l_residual = (
        F.mse_loss(residual_norm, target_residual_level)
        + 0.5 * F.mse_loss(spectrum_stability, target_stability)
        + 0.5 * F.mse_loss(mrw_vs_mono_gain, target_mrw_vs_mono_gain)
    )

    p_pair, r_pair, g_pair = _pairwise_boundary_losses(
        process_code,
        pair_id,
        p_mrw.view(-1),
        residual_norm.view(-1),
        mono_residual_norm.view(-1),
        mrw_vs_mono_gain.view(-1),
        lambda2_proj.view(-1),
    )
    l_surrogate = p_pair + r_pair + g_pair

    l_curvature = F.mse_loss(curvature_significance, target_curvature_significance)
    if mono_mask.any():
        l_curvature = l_curvature + 0.5 * F.relu(lambda2_proj - 0.015)[mono_mask.view(-1, 1)].mean()

    l_constraint = (
        concavity_penalty(outputs["zeta_mrw"])
        + mrw_parameter_box_penalty(H_proj, lambda2_proj)
        + 0.25 * concavity_penalty(zeta_emp)
        + 0.20 * torch.relu(-mono_fit_quality).mean()
    )

    total = (
        w_zeta * l_zeta
        + w_mrw_proj * l_mrw
        + w_mono * (l_mono + mrw_competition)
        + w_validity * l_validity
        + w_residual * l_residual
        + w_surrogate * l_surrogate
        + w_stability * l_curvature
        + w_constraint * l_constraint
    )
    return MonofractalCalibrationLossOutput(
        total=total,
        l_zeta=l_zeta,
        l_mrw=l_mrw,
        l_mono=l_mono,
        l_validity=l_validity,
        l_residual=l_residual,
        l_surrogate=l_surrogate,
        l_curvature=l_curvature,
        l_constraint=l_constraint,
    )
