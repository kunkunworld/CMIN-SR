from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mrw_inverse.data import PROCESS_NAME_TO_CODE

from .constraint_losses import concavity_penalty, mrw_parameter_box_penalty


@dataclass
class SpectralRepresentationLossOutput:
    total: torch.Tensor
    l_zeta_emp: torch.Tensor
    l_mrw_projection: torch.Tensor
    l_monofractal: torch.Tensor
    l_validity: torch.Tensor
    l_residual: torch.Tensor
    l_surrogate: torch.Tensor
    l_stability: torch.Tensor
    l_constraint: torch.Tensor


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return x.new_zeros(())
    return x[mask].mean()


def _masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return pred.new_zeros(())
    return torch.abs(pred[mask] - target[mask]).mean()


def _mrw_shuffled_surrogate_loss(
    p_mrw: torch.Tensor,
    residual_norm: torch.Tensor,
    lambda2_proj: torch.Tensor,
    process_code: torch.Tensor,
    pair_id: torch.Tensor,
    margin_p: float = 0.15,
    margin_residual: float = 0.02,
    margin_lambda: float = 0.005,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p_losses = []
    r_losses = []
    l_losses = []
    flat_proc = process_code.view(-1)
    flat_pair = pair_id.view(-1)
    for pid in torch.unique(flat_pair[flat_pair >= 0]).tolist():
        idx = torch.nonzero(flat_pair == pid, as_tuple=False).view(-1)
        if idx.numel() < 2:
            continue
        mrw_idx = idx[flat_proc[idx] == PROCESS_NAME_TO_CODE["MRW"]]
        shuf_idx = idx[flat_proc[idx] == PROCESS_NAME_TO_CODE["Shuffled MRW"]]
        if mrw_idx.numel() == 0 or shuf_idx.numel() == 0:
            continue
        i = mrw_idx[0]
        j = shuf_idx[0]
        p_losses.append(F.relu(margin_p - (p_mrw[i] - p_mrw[j])))
        r_losses.append(F.relu(margin_residual - (residual_norm[j] - residual_norm[i])))
        l_losses.append(F.relu(margin_lambda - (lambda2_proj[i] - lambda2_proj[j])))
    zero = p_mrw.new_zeros(())
    p_loss = torch.stack(p_losses).mean() if p_losses else zero
    r_loss = torch.stack(r_losses).mean() if r_losses else zero
    l_loss = torch.stack(l_losses).mean() if l_losses else zero
    return p_loss, r_loss, l_loss


def spectral_representation_loss(
    outputs: dict[str, torch.Tensor | str],
    process_code: torch.Tensor,
    zeta_target: torch.Tensor,
    h_true: torch.Tensor,
    lambda2_true: torch.Tensor,
    target_p_scaling: torch.Tensor,
    target_p_mrw: torch.Tensor,
    target_residual_level: torch.Tensor,
    target_stability: torch.Tensor,
    pair_id: torch.Tensor,
    w_zeta: float = 1.0,
    w_mrw_proj: float = 1.0,
    w_mono: float = 0.5,
    w_validity: float = 1.0,
    w_residual: float = 0.5,
    w_surrogate: float = 1.0,
    w_stability: float = 0.2,
    w_constraint: float = 0.1,
) -> SpectralRepresentationLossOutput:
    zeta_emp = outputs["zeta_emp"]
    H_proj = outputs["H_proj"]
    lambda2_proj = outputs["lambda2_proj"]
    residual_norm = outputs["residual_norm"]
    p_scaling = outputs["p_scaling"]
    p_mrw = outputs["p_mrw"]
    spectrum_stability = outputs["spectrum_stability"]
    projection_gain = outputs["projection_gain"]

    zero = zeta_emp.new_zeros(())
    process_flat = process_code.view(-1)
    mrw_mask = process_flat == PROCESS_NAME_TO_CODE["MRW"]
    mono_mask = (process_flat == PROCESS_NAME_TO_CODE["fGn"]) | (process_flat == PROCESS_NAME_TO_CODE["iid Gaussian"])
    valid_zeta_mask = ~torch.isnan(zeta_target).any(dim=1)

    l_zeta_emp = _masked_mae(zeta_emp, zeta_target, valid_zeta_mask.view(-1, 1).expand_as(zeta_emp))

    l_mrw_projection = zero
    if mrw_mask.any():
        l_mrw_projection = (
            _masked_mae(H_proj, h_true, mrw_mask.view(-1, 1))
            + _masked_mae(lambda2_proj, lambda2_true, mrw_mask.view(-1, 1))
            + _masked_mean(residual_norm, mrw_mask.view(-1, 1))
            + _masked_mean(1.0 - p_mrw, mrw_mask.view(-1, 1))
        )

    l_monofractal = zero
    if mono_mask.any():
        q_axis = torch.arange(zeta_emp.shape[1], device=zeta_emp.device, dtype=zeta_emp.dtype).view(1, -1)
        second = torch.gradient(torch.gradient(zeta_emp[mono_mask], spacing=(q_axis[0],), dim=1)[0], spacing=(q_axis[0],), dim=1)[0]
        l_monofractal = (
            second.abs().mean()
            + lambda2_proj[mono_mask].mean()
            + torch.abs(p_mrw[mono_mask] - target_p_mrw[mono_mask]).mean()
        )

    l_validity = (
        F.mse_loss(p_scaling, target_p_scaling)
        + F.mse_loss(p_mrw, target_p_mrw)
    )

    l_residual = (
        F.mse_loss(residual_norm, target_residual_level)
        + 0.5 * F.mse_loss(spectrum_stability, target_stability)
    )

    p_loss, r_loss, lambda_loss = _mrw_shuffled_surrogate_loss(
        p_mrw.view(-1),
        residual_norm.view(-1),
        lambda2_proj.view(-1),
        process_code,
        pair_id,
    )
    l_surrogate = p_loss + r_loss + lambda_loss

    l_stability = torch.abs(spectrum_stability - target_stability).mean()

    l_constraint = (
        concavity_penalty(outputs["zeta_mrw"])
        + 0.2 * concavity_penalty(zeta_emp)
        + mrw_parameter_box_penalty(H_proj, lambda2_proj)
        + 0.2 * torch.relu(-projection_gain).mean()
    )

    total = (
        w_zeta * l_zeta_emp
        + w_mrw_proj * l_mrw_projection
        + w_mono * l_monofractal
        + w_validity * l_validity
        + w_residual * l_residual
        + w_surrogate * l_surrogate
        + w_stability * l_stability
        + w_constraint * l_constraint
    )
    return SpectralRepresentationLossOutput(
        total=total,
        l_zeta_emp=l_zeta_emp,
        l_mrw_projection=l_mrw_projection,
        l_monofractal=l_monofractal,
        l_validity=l_validity,
        l_residual=l_residual,
        l_surrogate=l_surrogate,
        l_stability=l_stability,
        l_constraint=l_constraint,
    )

