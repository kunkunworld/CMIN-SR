from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mrw_inverse.data import PROCESS_NAME_TO_CODE


@dataclass
class CurvaturePreservingZetaLossOutput:
    total: torch.Tensor
    l_zeta: torch.Tensor
    l_mono_linear: torch.Tensor
    l_mrw_curv_band: torch.Tensor
    l_lambda_proj: torch.Tensor
    l_residual_margin: torch.Tensor
    l_third_smooth: torch.Tensor
    l_tail: torch.Tensor
    l_stability: torch.Tensor
    l_weak_smooth: torch.Tensor


def _d2(z: torch.Tensor) -> torch.Tensor:
    return z[:, :-2] - 2.0 * z[:, 1:-1] + z[:, 2:]


def _d3(z: torch.Tensor) -> torch.Tensor:
    if z.shape[1] < 4:
        return z.new_zeros((z.shape[0], 1))
    return z[:, :-3] - 3.0 * z[:, 1:-2] + 3.0 * z[:, 2:-1] - z[:, 3:]


def _band_weights(lambda2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lam = torch.nan_to_num(lambda2.view(-1), nan=0.0)
    curv_w = torch.where(lam < 0.03, torch.full_like(lam, 0.5), torch.where(lam < 0.08, torch.full_like(lam, 2.0), torch.full_like(lam, 5.0)))
    lambda_w = torch.where(lam < 0.03, torch.full_like(lam, 0.5), torch.where(lam < 0.08, torch.full_like(lam, 1.5), torch.full_like(lam, 3.0)))
    margin = torch.where(lam < 0.08, torch.full_like(lam, 0.005), torch.full_like(lam, 0.01))
    return curv_w, lambda_w, margin


def curvature_preserving_zeta_loss(
    outputs: dict[str, torch.Tensor | str],
    process_code: torch.Tensor,
    zeta_target: torch.Tensor,
    zeta_target_mask: torch.Tensor,
    zeta_weight_by_q: torch.Tensor,
    h_true: torch.Tensor,
    lambda2_true: torch.Tensor,
    target_tail_instability: torch.Tensor,
    w_zeta: float = 2.0,
    w_mono_linear: float = 1.0,
    w_residual_margin: float = 1.0,
    w_third_smooth: float = 0.2,
    w_tail: float = 0.5,
    w_stability: float = 0.2,
    w_smooth_weak: float = 0.05,
) -> CurvaturePreservingZetaLossOutput:
    zeta = torch.nan_to_num(outputs["zeta_emp"])
    target = torch.nan_to_num(zeta_target)
    mask = torch.nan_to_num(zeta_target_mask).clamp(0.0, 1.0)
    weights = torch.nan_to_num(zeta_weight_by_q).clamp_min(0.0)
    denom = (mask * weights).sum().clamp_min(1.0)
    l_zeta = (torch.abs(zeta - target) * mask * weights).sum() / denom

    codes = process_code.view(-1)
    mono_mask = (codes == PROCESS_NAME_TO_CODE["fGn"]) | (codes == PROCESS_NAME_TO_CODE["iid Gaussian"])
    mrw_mask = (codes == PROCESS_NAME_TO_CODE["MRW"]) | (codes == PROCESS_NAME_TO_CODE["Low-lambda2 MRW"])
    stress_mask = (codes == PROCESS_NAME_TO_CODE["iid Student-t"]) | (codes == PROCESS_NAME_TO_CODE["GARCH(1,1)"]) | (codes == PROCESS_NAME_TO_CODE["Regime-switching Gaussian"])
    zero = zeta.new_zeros(())

    l_mono_linear = zero
    if mono_mask.any():
        m = mono_mask.view(-1, 1)
        l_mono_linear = 2.0 * _d2(zeta[mono_mask]).abs().mean()
        l_mono_linear = l_mono_linear + 4.0 * outputs["lambda2_proj"][m].abs().mean()
        l_mono_linear = l_mono_linear + 2.0 * F.relu(outputs["mono_residual_norm"][m] - outputs["residual_norm"][m] - 0.005).mean()

    l_mrw_curv_band = zero
    l_lambda_proj = zero
    l_residual_margin = zero
    if mrw_mask.any():
        curv_w, lambda_w, margin = _band_weights(lambda2_true[mrw_mask])
        d2_err = (_d2(zeta[mrw_mask]) - _d2(target[mrw_mask])).abs().mean(dim=1)
        l_mrw_curv_band = (curv_w * d2_err).mean()
        lam_err = (outputs["lambda2_proj"][mrw_mask.view(-1, 1)].view(-1) - lambda2_true[mrw_mask].view(-1)).abs()
        l_lambda_proj = (lambda_w * lam_err).mean()
        mono_minus_mrw = (outputs["mono_residual_norm"][mrw_mask.view(-1, 1)].view(-1) - outputs["residual_norm"][mrw_mask.view(-1, 1)].view(-1))
        active = lambda2_true[mrw_mask].view(-1) >= 0.03
        if active.any():
            l_residual_margin = F.relu(margin[active] - mono_minus_mrw[active]).mean()

    l_third_smooth = _d3(zeta).abs().mean()
    l_weak_smooth = zero
    if stress_mask.any():
        l_weak_smooth = 0.25 * _d3(zeta[stress_mask]).abs().mean()

    tail = torch.nan_to_num(outputs["tail_instability"]).clamp(0.0, 1.0)
    l_tail = F.mse_loss(tail, target_tail_instability)
    zeta_unc = torch.nan_to_num(outputs.get("zeta_uncertainty", zeta.new_zeros((zeta.shape[0], 1))))
    clean = mono_mask | mrw_mask
    l_stability = zeta_unc[clean.view(-1, 1)].mean() if clean.any() else zero

    total = (
        w_zeta * l_zeta
        + w_mono_linear * l_mono_linear
        + l_mrw_curv_band
        + l_lambda_proj
        + w_residual_margin * l_residual_margin
        + w_third_smooth * l_third_smooth
        + w_tail * l_tail
        + w_stability * l_stability
        + w_smooth_weak * l_weak_smooth
    )
    return CurvaturePreservingZetaLossOutput(
        total=torch.nan_to_num(total),
        l_zeta=torch.nan_to_num(l_zeta),
        l_mono_linear=torch.nan_to_num(l_mono_linear),
        l_mrw_curv_band=torch.nan_to_num(l_mrw_curv_band),
        l_lambda_proj=torch.nan_to_num(l_lambda_proj),
        l_residual_margin=torch.nan_to_num(l_residual_margin),
        l_third_smooth=torch.nan_to_num(l_third_smooth),
        l_tail=torch.nan_to_num(l_tail),
        l_stability=torch.nan_to_num(l_stability),
        l_weak_smooth=torch.nan_to_num(l_weak_smooth),
    )
