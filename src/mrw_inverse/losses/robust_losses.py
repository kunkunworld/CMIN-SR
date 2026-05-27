from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mrw_inverse.data import PROCESS_NAME_TO_CODE

from .constraint_losses import concavity_penalty, mrw_parameter_box_penalty, residual_smallness_penalty


@dataclass
class RobustLossOutput:
    total: torch.Tensor
    l_param: torch.Tensor
    l_spectrum: torch.Tensor
    l_validity: torch.Tensor
    l_neg_lambda: torch.Tensor
    l_pair: torch.Tensor
    l_mismatch: torch.Tensor
    l_constraint: torch.Tensor


def _safe_masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return pred.new_zeros(())
    diff = torch.abs(pred - target)
    return diff[mask].mean()


def _pairwise_gap_loss(lambda2_hat: torch.Tensor, process_code: torch.Tensor, pair_id: torch.Tensor, margin: float) -> torch.Tensor:
    losses = []
    flat_lambda = lambda2_hat.view(-1)
    flat_process = process_code.view(-1)
    flat_pair = pair_id.view(-1)
    unique_pairs = torch.unique(flat_pair[flat_pair >= 0])
    for pid in unique_pairs.tolist():
        idx = torch.nonzero(flat_pair == pid, as_tuple=False).view(-1)
        if idx.numel() < 2:
            continue
        mrw_mask = flat_process[idx] == PROCESS_NAME_TO_CODE["MRW"]
        shuf_mask = flat_process[idx] == PROCESS_NAME_TO_CODE["Shuffled MRW"]
        if mrw_mask.sum() == 0 or shuf_mask.sum() == 0:
            continue
        orig = flat_lambda[idx[mrw_mask][0]]
        shuf = flat_lambda[idx[shuf_mask][0]]
        losses.append(F.relu(margin - (orig - shuf)))
    if not losses:
        return lambda2_hat.new_zeros(())
    return torch.stack(losses).mean()


def robust_inverse_loss(
    output,
    process_code: torch.Tensor,
    is_mrw: torch.Tensor,
    h_true: torch.Tensor,
    lambda2_true: torch.Tensor,
    target_lambda2: torch.Tensor,
    target_p_mrw: torch.Tensor,
    target_mismatch: torch.Tensor,
    zeta_true: torch.Tensor,
    f_true: torch.Tensor,
    pair_id: torch.Tensor,
    param_weight: float = 1.0,
    spectrum_weight: float = 0.5,
    validity_weight: float = 1.0,
    neg_lambda_weight: float = 1.0,
    pair_weight: float = 1.0,
    mismatch_weight: float = 0.25,
    constraint_weight: float = 0.1,
    pair_margin: float = 0.01,
    gaussian_margin: float = 0.01,
    base_negative_margin: float = 0.02,
    ambiguous_margin: float = 0.04,
) -> RobustLossOutput:
    mrw_mask = is_mrw.view(-1) > 0.5
    non_mrw_mask = ~mrw_mask

    l_param = (
        _safe_masked_mae(output.h_hat.view(-1, 1), h_true, mrw_mask.view(-1, 1))
        + _safe_masked_mae(output.lambda2_hat.view(-1, 1), lambda2_true, mrw_mask.view(-1, 1))
    )

    valid_f_mask = mrw_mask & torch.isfinite(f_true).all(dim=1)
    l_spectrum = _safe_masked_mae(output.zeta_hat, zeta_true, mrw_mask.view(-1, 1).expand_as(output.zeta_hat))
    l_spectrum = l_spectrum + 0.4 * _safe_masked_mae(output.f_alpha_hat, f_true, valid_f_mask.view(-1, 1).expand_as(output.f_alpha_hat))

    l_validity = F.mse_loss(output.p_mrw, target_p_mrw)

    margins = torch.full_like(output.lambda2_hat, base_negative_margin)
    margins[process_code.view(-1, 1) == PROCESS_NAME_TO_CODE["iid Gaussian"]] = gaussian_margin
    margins[process_code.view(-1, 1) == PROCESS_NAME_TO_CODE["GARCH(1,1)"]] = ambiguous_margin
    margins[process_code.view(-1, 1) == PROCESS_NAME_TO_CODE["Regime-switching Gaussian"]] = ambiguous_margin
    if non_mrw_mask.sum() == 0:
        l_neg_lambda = output.lambda2_hat.new_zeros(())
    else:
        neg_term = F.relu(output.lambda2_hat - margins)
        l_neg_lambda = neg_term[non_mrw_mask.view(-1, 1)].mean()

    l_pair = _pairwise_gap_loss(output.lambda2_hat, process_code, pair_id, margin=pair_margin)

    residual_norm = torch.mean(torch.abs(output.residual_zeta), dim=1, keepdim=True)
    l_mismatch = F.mse_loss(residual_norm, target_mismatch)

    l_constraint = concavity_penalty(output.zeta_hat) + concavity_penalty(output.f_alpha_hat) + mrw_parameter_box_penalty(output.h_hat, output.lambda2_hat)
    if mrw_mask.any():
        l_constraint = l_constraint + 0.5 * residual_smallness_penalty(output.residual_zeta[mrw_mask])

    total = (
        param_weight * l_param
        + spectrum_weight * l_spectrum
        + validity_weight * l_validity
        + neg_lambda_weight * l_neg_lambda
        + pair_weight * l_pair
        + mismatch_weight * l_mismatch
        + constraint_weight * l_constraint
    )
    return RobustLossOutput(
        total=total,
        l_param=l_param,
        l_spectrum=l_spectrum,
        l_validity=l_validity,
        l_neg_lambda=l_neg_lambda,
        l_pair=l_pair,
        l_mismatch=l_mismatch,
        l_constraint=l_constraint,
    )
