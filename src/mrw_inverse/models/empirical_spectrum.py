from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EmpiricalSpectrumOutput:
    log_structure_matrix: torch.Tensor
    zeta_emp: torch.Tensor
    zeta_fit_r2: torch.Tensor
    scaling_fit_quality: torch.Tensor
    alpha_emp: torch.Tensor
    f_emp: torch.Tensor
    spectrum_width: torch.Tensor
    spectrum_curvature: torch.Tensor
    p_scaling: torch.Tensor
    spectrum_stability: torch.Tensor
    valid_scale_fraction: torch.Tensor
    instability_flag: torch.Tensor
    tail_instability: torch.Tensor


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim != 2:
        raise ValueError(f"Expected [batch, time] tensor, got shape {tuple(x.shape)}")
    return x


def _q_tensor(q_grid: tuple[float, ...] | list[float] | torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(q_grid, torch.Tensor):
        return q_grid.to(device=device, dtype=dtype)
    return torch.tensor(q_grid, device=device, dtype=dtype)


def _scale_tensor(scales: tuple[int, ...] | list[int] | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(scales, torch.Tensor):
        return scales.to(device=device, dtype=torch.long)
    return torch.tensor(scales, device=device, dtype=torch.long)


def estimate_structure_functions(
    x: torch.Tensor,
    q_grid: tuple[float, ...] | list[float] | torch.Tensor = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    scales: tuple[int, ...] | list[int] | torch.Tensor = (2, 4, 8, 16, 32, 64),
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    x = _ensure_2d(x)
    path = torch.cumsum(x, dim=1)
    q = _q_tensor(q_grid, x.device, x.dtype)
    scale_tensor = _scale_tensor(scales, x.device)

    moment_rows = []
    valid_mask = []
    for scale in scale_tensor.tolist():
        if scale >= x.shape[1]:
            inc = torch.full((x.shape[0], 1), float("nan"), device=x.device, dtype=x.dtype)
            moments = torch.full((x.shape[0], q.numel()), float("nan"), device=x.device, dtype=x.dtype)
            valid = torch.zeros((x.shape[0],), device=x.device, dtype=x.dtype)
        else:
            inc = path[:, scale:] - path[:, :-scale]
            abs_inc = inc.abs().clamp_min(eps)
            moments = torch.stack([(abs_inc.pow(float(qi))).mean(dim=1) for qi in q], dim=1)
            valid = torch.ones((x.shape[0],), device=x.device, dtype=x.dtype)
        moment_rows.append(torch.log(moments.clamp_min(eps)))
        valid_mask.append(valid)

    log_structure = torch.stack(moment_rows, dim=2)
    valid_scale_fraction = torch.stack(valid_mask, dim=1).mean(dim=1, keepdim=True)
    return {
        "log_structure_matrix": log_structure,
        "q_grid": q,
        "scales": scale_tensor,
        "valid_scale_fraction": valid_scale_fraction,
    }


def estimate_empirical_zeta(
    x: torch.Tensor,
    q_grid: tuple[float, ...] | list[float] | torch.Tensor = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    scales: tuple[int, ...] | list[int] | torch.Tensor = (2, 4, 8, 16, 32, 64),
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    stats = estimate_structure_functions(x, q_grid=q_grid, scales=scales, eps=eps)
    log_structure = stats["log_structure_matrix"]
    log_scales = torch.log(stats["scales"].to(dtype=log_structure.dtype)).view(1, 1, -1)
    x_centered = log_scales - log_scales.mean(dim=2, keepdim=True)
    denom = (x_centered.square().sum(dim=2, keepdim=True)).clamp_min(eps)
    y_centered = log_structure - log_structure.mean(dim=2, keepdim=True)
    slope = (x_centered * y_centered).sum(dim=2) / denom.squeeze(2)
    intercept = log_structure.mean(dim=2) - slope * log_scales.mean(dim=2).squeeze(1)
    fitted = intercept.unsqueeze(2) + slope.unsqueeze(2) * log_scales
    ss_res = ((log_structure - fitted) ** 2).sum(dim=2)
    ss_tot = ((log_structure - log_structure.mean(dim=2, keepdim=True)) ** 2).sum(dim=2).clamp_min(eps)
    r2_per_q = 1.0 - ss_res / ss_tot
    scaling_fit_quality = r2_per_q.nanmean(dim=1, keepdim=True).clamp(0.0, 1.0)
    return {
        **stats,
        "zeta_emp": slope,
        "zeta_fit_r2_per_q": r2_per_q,
        "zeta_fit_r2": scaling_fit_quality,
    }


def estimate_empirical_spectrum(
    x: torch.Tensor,
    q_grid: tuple[float, ...] | list[float] | torch.Tensor = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    scales: tuple[int, ...] | list[int] | torch.Tensor = (2, 4, 8, 16, 32, 64),
    eps: float = 1e-8,
) -> EmpiricalSpectrumOutput:
    x2d = _ensure_2d(x)
    out = estimate_empirical_zeta(x2d, q_grid=q_grid, scales=scales, eps=eps)
    q = out["q_grid"].view(1, -1)
    zeta_emp = out["zeta_emp"]
    alpha_emp = torch.gradient(zeta_emp, spacing=(q[0],), dim=1)[0]
    f_emp = 1.0 + q * alpha_emp - zeta_emp
    second = torch.gradient(torch.gradient(zeta_emp, spacing=(q[0],), dim=1)[0], spacing=(q[0],), dim=1)[0]
    spectrum_curvature = (-second.nanmean(dim=1, keepdim=True)).clamp_min(0.0)
    spectrum_width = (alpha_emp.max(dim=1, keepdim=True).values - alpha_emp.min(dim=1, keepdim=True).values).clamp_min(0.0)

    half = max(x2d.shape[1] // 2, 8)
    left = x2d[:, :half]
    right = x2d[:, -half:]
    left_zeta = estimate_empirical_zeta(left, q_grid=q_grid, scales=scales, eps=eps)["zeta_emp"]
    right_zeta = estimate_empirical_zeta(right, q_grid=q_grid, scales=scales, eps=eps)["zeta_emp"]
    split_gap = (left_zeta - right_zeta).abs().mean(dim=1, keepdim=True)
    spectrum_stability = torch.exp(-4.0 * split_gap).clamp(0.0, 1.0)

    centered_x = x2d - x2d.mean(dim=1, keepdim=True)
    standardized_x = centered_x / centered_x.std(dim=1, keepdim=True).clamp_min(eps)
    excess_kurtosis = standardized_x.pow(4).mean(dim=1, keepdim=True) - 3.0
    tail_instability = ((excess_kurtosis - 2.0) / 10.0).clamp(0.0, 1.0)

    high_q_tail = zeta_emp[:, -1:] - zeta_emp[:, -2:-1]
    instability_flag = ((high_q_tail.abs() > 2.0) | (out["zeta_fit_r2"] < 0.5)).to(dtype=x2d.dtype)
    p_scaling = (
        0.55 * out["zeta_fit_r2"]
        + 0.25 * spectrum_stability
        + 0.20 * out["valid_scale_fraction"]
        - 0.20 * instability_flag
        - 0.25 * tail_instability
    ).clamp(0.0, 1.0)

    return EmpiricalSpectrumOutput(
        log_structure_matrix=out["log_structure_matrix"],
        zeta_emp=zeta_emp,
        zeta_fit_r2=out["zeta_fit_r2"],
        scaling_fit_quality=out["zeta_fit_r2"],
        alpha_emp=alpha_emp,
        f_emp=f_emp,
        spectrum_width=spectrum_width,
        spectrum_curvature=spectrum_curvature,
        p_scaling=p_scaling,
        spectrum_stability=spectrum_stability,
        valid_scale_fraction=out["valid_scale_fraction"],
        instability_flag=instability_flag,
        tail_instability=tail_instability,
    )
