from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mrw_inverse.data import PROCESS_NAME_TO_CODE

from .constraint_losses import concavity_penalty, mrw_parameter_box_penalty


@dataclass
class CurvedLinearCalibrationLossOutput:
    total: torch.Tensor
    l_base: torch.Tensor
    l_curved: torch.Tensor
    l_mono_head: torch.Tensor
    l_boundary: torch.Tensor
    l_gate: torch.Tensor
    l_mrw_preserve: torch.Tensor
    l_mono_reject: torch.Tensor
    l_low_lambda: torch.Tensor
    l_constraint: torch.Tensor


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return pred.new_zeros(())
    return F.mse_loss(pred[mask], target[mask])


def cmin_sr_v3_loss(
    outputs: dict[str, torch.Tensor | str],
    process_code: torch.Tensor,
    zeta_target: torch.Tensor,
    h_true: torch.Tensor,
    lambda2_true: torch.Tensor,
    target_p_scaling: torch.Tensor,
    target_p_mrw: torch.Tensor,
    target_p_curved: torch.Tensor,
    target_p_mono: torch.Tensor,
    target_boundary_mrw: torch.Tensor,
    target_residual_level: torch.Tensor,
    target_stability: torch.Tensor,
    target_curvature_significance: torch.Tensor,
    target_mrw_vs_mono_gain: torch.Tensor,
    pair_id: torch.Tensor,
    w_zeta: float = 1.0,
    w_mrw_proj: float = 1.0,
    w_validity: float = 1.0,
    w_residual: float = 0.5,
    w_curved: float = 1.0,
    w_mono_head: float = 0.5,
    w_boundary: float = 0.5,
    w_gate: float = 1.0,
    w_mrw_preserve: float = 1.0,
    w_mono_reject: float = 1.0,
    w_low_lambda: float = 0.5,
    w_constraint: float = 0.1,
) -> CurvedLinearCalibrationLossOutput:
    codes = process_code.view(-1)
    mrw_mask = codes == PROCESS_NAME_TO_CODE["MRW"]
    low_mrw_mask = codes == PROCESS_NAME_TO_CODE["Low-lambda2 MRW"]
    strong_mrw_mask = mrw_mask & torch.isfinite(lambda2_true.view(-1)) & (lambda2_true.view(-1) >= 0.04)
    boundary_mask = low_mrw_mask | (mrw_mask & torch.isfinite(lambda2_true.view(-1)) & (lambda2_true.view(-1) < 0.04))
    mono_mask = (codes == PROCESS_NAME_TO_CODE["fGn"]) | (codes == PROCESS_NAME_TO_CODE["iid Gaussian"])
    stress_mask = (codes == PROCESS_NAME_TO_CODE["iid Student-t"]) | (codes == PROCESS_NAME_TO_CODE["Regime-switching Gaussian"])
    garch_mask = codes == PROCESS_NAME_TO_CODE["GARCH(1,1)"]

    zeta_emp = outputs["zeta_emp"]
    H_proj = outputs["H_proj"]
    lambda2_proj = outputs["lambda2_proj"]
    p_scaling = outputs["p_scaling"]
    p_curved = outputs["p_curved"]
    p_mono = outputs["p_mono"]
    p_mrw = outputs["p_mrw"]
    residual_norm = outputs["residual_norm"]
    mono_residual_norm = outputs["mono_residual_norm"]
    curvature_significance = outputs["curvature_significance"]
    normalized_gain = outputs["normalized_mrw_gain"]
    boundary_score = outputs["boundary_mrw_score"]
    spectrum_stability = outputs["spectrum_stability"]
    H_mono = outputs["H_mono"]
    zeta_mono = outputs["zeta_mono"]

    valid_zeta = ~torch.isnan(zeta_target).any(dim=1)
    base = zeta_emp.new_zeros(())
    if valid_zeta.any():
        base = base + F.l1_loss(zeta_emp[valid_zeta], zeta_target[valid_zeta])
    if strong_mrw_mask.any():
        mask = strong_mrw_mask.view(-1, 1)
        base = base + F.l1_loss(H_proj[mask], h_true[mask]) + F.l1_loss(lambda2_proj[mask], lambda2_true[mask])
    if boundary_mask.any():
        mask = boundary_mask.view(-1, 1)
        base = base + 0.5 * F.l1_loss(H_proj[mask], h_true[mask]) + 0.5 * F.l1_loss(lambda2_proj[mask], lambda2_true[mask])

    l_curved = F.mse_loss(p_curved, target_p_curved) + 0.5 * F.mse_loss(curvature_significance, target_curvature_significance)
    l_mono_head = F.mse_loss(p_mono, target_p_mono)
    l_boundary = F.mse_loss(boundary_score, target_boundary_mrw)

    boundary_allowance = 0.35 * boundary_score
    gate_cap = F.relu(p_mrw - (p_curved + boundary_allowance))
    scaling_cap = F.relu(p_mrw - p_scaling)
    l_gate = gate_cap.mean() + 0.5 * scaling_cap.mean()

    l_mrw_preserve = zeta_emp.new_zeros(())
    if strong_mrw_mask.any():
        mask = strong_mrw_mask.view(-1, 1)
        l_mrw_preserve = (
            F.mse_loss(p_mrw[mask], target_p_mrw[mask])
            + F.mse_loss(p_curved[mask], target_p_curved[mask])
            + F.relu(0.015 - (mono_residual_norm - residual_norm))[mask].mean()
            + F.relu(0.10 - (p_mrw - p_mono))[mask].mean()
        )

    l_mono_reject = zeta_emp.new_zeros(())
    if mono_mask.any():
        mask = mono_mask.view(-1, 1)
        mono_h_mask = mono_mask & torch.isfinite(h_true.view(-1))
        l_mono_reject = (
            F.mse_loss(p_curved[mask], target_p_curved[mask])
            + F.mse_loss(p_mono[mask], target_p_mono[mask])
            + F.mse_loss(p_mrw[mask], target_p_mrw[mask])
            + F.relu(0.01 - (residual_norm - mono_residual_norm))[mask].mean()
        )
        if mono_h_mask.any():
            mono_h_view = mono_h_mask.view(-1, 1)
            l_mono_reject = l_mono_reject + 0.5 * F.l1_loss(H_mono[mono_h_view], h_true[mono_h_view])
        if valid_zeta.any():
            mono_valid = mono_mask & valid_zeta
            if mono_valid.any():
                mono_valid_view = mono_valid.view(-1, 1).expand_as(zeta_mono)
                l_mono_reject = l_mono_reject + 0.5 * F.l1_loss(zeta_mono[mono_valid_view], zeta_target[mono_valid_view])

    l_low_lambda = zeta_emp.new_zeros(())
    if boundary_mask.any():
        mask = boundary_mask.view(-1, 1)
        l_low_lambda = (
            F.mse_loss(boundary_score[mask], target_boundary_mrw[mask])
            + F.mse_loss(p_mrw[mask], target_p_mrw[mask])
            + F.mse_loss(p_mono[mask], target_p_mono[mask])
            + F.mse_loss(p_curved[mask], target_p_curved[mask])
        )

    residual_target_loss = F.mse_loss(residual_norm, target_residual_level) + 0.3 * F.mse_loss(spectrum_stability, target_stability)
    l_curved = l_curved + 0.3 * F.mse_loss(normalized_gain.clamp(0.0, 1.0), target_mrw_vs_mono_gain.clamp(0.0, 1.0))

    if stress_mask.any():
        mask = stress_mask.view(-1, 1)
        l_gate = l_gate + 0.5 * F.relu(p_curved - 0.45)[mask].mean()
    if garch_mask.any():
        mask = garch_mask.view(-1, 1)
        l_gate = l_gate + 0.2 * F.relu(p_mrw - 0.55)[mask].mean()

    l_constraint = (
        concavity_penalty(outputs["zeta_mrw"])
        + mrw_parameter_box_penalty(H_proj, lambda2_proj)
        + 0.15 * concavity_penalty(zeta_emp)
    )

    total = (
        w_zeta * base
        + w_validity * residual_target_loss
        + w_curved * l_curved
        + w_mono_head * l_mono_head
        + w_boundary * l_boundary
        + w_gate * l_gate
        + w_mrw_preserve * l_mrw_preserve
        + w_mono_reject * l_mono_reject
        + w_low_lambda * l_low_lambda
        + w_constraint * l_constraint
    )
    return CurvedLinearCalibrationLossOutput(
        total=total,
        l_base=base,
        l_curved=l_curved,
        l_mono_head=l_mono_head,
        l_boundary=l_boundary,
        l_gate=l_gate,
        l_mrw_preserve=l_mrw_preserve,
        l_mono_reject=l_mono_reject,
        l_low_lambda=l_low_lambda,
        l_constraint=l_constraint,
    )
