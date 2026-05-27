from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .constraint_losses import concavity_penalty, mrw_parameter_box_penalty, residual_smallness_penalty
from .contrastive_losses import paired_identifiability_loss


@dataclass
class MRWLossOutput:
    total: torch.Tensor
    l_param: torch.Tensor
    l_spectrum: torch.Tensor
    l_scaling: torch.Tensor
    l_logvol: torch.Tensor
    l_constraint: torch.Tensor
    l_contrast: torch.Tensor


def scaling_reconstruction_loss(log_structure_matrix: torch.Tensor, zeta_hat: torch.Tensor, log_scales: torch.Tensor) -> torch.Tensor:
    # log S_q(a) ≈ c_q + zeta(q) log a, with c_q estimated by least squares mean.
    target = log_structure_matrix.transpose(1, 2)
    pred = zeta_hat.unsqueeze(1) * log_scales.view(1, -1, 1)
    intercept = (target - pred).mean(dim=1, keepdim=True)
    recon = intercept + pred
    return F.l1_loss(recon, target)


def logvol_covariance_loss(cov_curve: torch.Tensor, lambda2_hat: torch.Tensor, log_lags: torch.Tensor) -> torch.Tensor:
    pred = -lambda2_hat * log_lags.view(1, -1)
    intercept = (cov_curve - pred).mean(dim=1, keepdim=True)
    recon = intercept + pred
    return F.l1_loss(recon, cov_curve)


def mrw_total_loss(
    output,
    h_true: torch.Tensor,
    lambda2_true: torch.Tensor,
    zeta_true: torch.Tensor,
    f_true: torch.Tensor,
    log_scales: torch.Tensor,
    log_lags: torch.Tensor,
    param_weight: float = 1.0,
    zeta_weight: float = 0.5,
    falpha_weight: float = 0.2,
    scaling_weight: float = 0.0,
    logvol_weight: float = 0.0,
    constraint_weight: float = 0.1,
    contrast_weight: float = 0.1,
) -> MRWLossOutput:
    l_param = F.l1_loss(output.h_hat, h_true) + F.l1_loss(output.lambda2_hat, lambda2_true)
    l_spectrum = zeta_weight * F.l1_loss(output.zeta_hat, zeta_true) + falpha_weight * F.l1_loss(output.f_alpha_hat, f_true)
    l_scaling = scaling_reconstruction_loss(output.log_structure_matrix, output.zeta_hat, log_scales)
    l_logvol = logvol_covariance_loss(output.logvol_covariance_curve, output.lambda2_hat, log_lags)
    l_constraint = (
        concavity_penalty(output.zeta_hat)
        + concavity_penalty(output.f_alpha_hat)
        + mrw_parameter_box_penalty(output.h_hat, output.lambda2_hat)
        + 0.5 * residual_smallness_penalty(output.residual_zeta)
    )
    l_contrast = paired_identifiability_loss(
        output.scale_graph_embedding,
        h_true.squeeze(1),
        lambda2_true.squeeze(1),
        mode="same_h_diff_lambda",
    )
    total = (
        param_weight * l_param
        + l_spectrum
        + scaling_weight * l_scaling
        + logvol_weight * l_logvol
        + constraint_weight * l_constraint
        + contrast_weight * l_contrast
    )
    return MRWLossOutput(total, l_param, l_spectrum, l_scaling, l_logvol, l_constraint, l_contrast)
