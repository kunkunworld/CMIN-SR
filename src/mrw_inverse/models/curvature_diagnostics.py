from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CurvatureDiagnosticsOutput:
    curvature_score: torch.Tensor
    curvature_significance: torch.Tensor
    linearity_score: torch.Tensor
    mrw_vs_mono_gain: torch.Tensor
    normalized_mrw_gain: torch.Tensor
    curvature_confidence: torch.Tensor
    boundary_mrw_score: torch.Tensor


def compute_curvature_diagnostics(
    q_grid: torch.Tensor,
    zeta_emp: torch.Tensor,
    zeta_mono: torch.Tensor,
    zeta_mrw: torch.Tensor,
    residual_norm: torch.Tensor,
    mono_residual_norm: torch.Tensor,
    lambda2_proj: torch.Tensor,
) -> CurvatureDiagnosticsOutput:
    zeta_emp = torch.nan_to_num(zeta_emp, nan=0.0, posinf=0.0, neginf=0.0)
    zeta_mono = torch.nan_to_num(zeta_mono, nan=0.0, posinf=0.0, neginf=0.0)
    zeta_mrw = torch.nan_to_num(zeta_mrw, nan=0.0, posinf=0.0, neginf=0.0)
    residual_norm = torch.nan_to_num(residual_norm, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(0.0)
    mono_residual_norm = torch.nan_to_num(mono_residual_norm, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(0.0)
    lambda2_proj = torch.nan_to_num(lambda2_proj, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if q_grid.ndim == 1:
        q = q_grid.view(1, -1).to(zeta_emp)
    else:
        q = q_grid.to(zeta_emp)

    grad_1 = torch.gradient(zeta_emp, spacing=(q[0],), dim=1)[0]
    grad_2 = torch.gradient(grad_1, spacing=(q[0],), dim=1)[0]
    second_diff_mag = grad_2.abs().mean(dim=1, keepdim=True)
    curvature_score = (second_diff_mag / (second_diff_mag + 0.10)).clamp(0.0, 1.0)

    curve_gap = (zeta_mrw - zeta_mono).abs().mean(dim=1, keepdim=True)
    linearity_score = (1.0 - mono_residual_norm / (mono_residual_norm + 0.10 + curve_gap)).clamp(0.0, 1.0)
    normalized_gain = ((mono_residual_norm - residual_norm) / (mono_residual_norm + 1e-6)).clamp(-1.0, 1.0)
    positive_gain = normalized_gain.clamp_min(0.0)
    lambda_score = (lambda2_proj / (lambda2_proj + 0.04)).clamp(0.0, 1.0)
    curvature_confidence = (0.50 * positive_gain + 0.30 * curvature_score + 0.20 * lambda_score).clamp(0.0, 1.0)
    curvature_significance = (0.55 * lambda_score + 0.30 * positive_gain + 0.15 * curvature_score).clamp(0.0, 1.0)
    gain_balance = 1.0 - positive_gain
    small_lambda = 1.0 - lambda_score
    good_dual_fit = (1.0 - (residual_norm + mono_residual_norm) / (residual_norm + mono_residual_norm + 0.16)).clamp(0.0, 1.0)
    boundary_mrw_score = (0.45 * small_lambda + 0.35 * gain_balance + 0.20 * good_dual_fit).clamp(0.0, 1.0)

    return CurvatureDiagnosticsOutput(
        curvature_score=torch.nan_to_num(curvature_score).clamp(0.0, 1.0),
        curvature_significance=torch.nan_to_num(curvature_significance).clamp(0.0, 1.0),
        linearity_score=torch.nan_to_num(linearity_score).clamp(0.0, 1.0),
        mrw_vs_mono_gain=torch.nan_to_num(positive_gain).clamp(0.0, 1.0),
        normalized_mrw_gain=torch.nan_to_num(normalized_gain).clamp(-1.0, 1.0),
        curvature_confidence=torch.nan_to_num(curvature_confidence).clamp(0.0, 1.0),
        boundary_mrw_score=torch.nan_to_num(boundary_mrw_score).clamp(0.0, 1.0),
    )
