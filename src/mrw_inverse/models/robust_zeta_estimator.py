from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RobustZetaOutput:
    zeta_robust: torch.Tensor
    zeta_ols: torch.Tensor
    zeta_bootstrap_std: torch.Tensor
    scaling_fit_r2: torch.Tensor
    high_q_instability: torch.Tensor
    scale_coverage: torch.Tensor
    warning_flags: torch.Tensor


def _increments_at_scale(x: torch.Tensor, scale: int) -> torch.Tensor:
    return x[:, scale:] - x[:, :-scale]


def _fit_slope(log_s: torch.Tensor, log_m: torch.Tensor, trim: bool) -> tuple[torch.Tensor, torch.Tensor]:
    xs = log_s.view(1, -1).expand_as(log_m)
    ys = log_m
    if trim and ys.shape[1] > 4:
        resid = ys - ys.mean(dim=1, keepdim=True)
        keep = resid.abs() <= torch.quantile(resid.abs(), 0.80, dim=1, keepdim=True)
        xs = torch.where(keep, xs, torch.nan)
        ys = torch.where(keep, ys, torch.nan)
    xm = torch.nanmean(xs, dim=1, keepdim=True)
    ym = torch.nanmean(ys, dim=1, keepdim=True)
    cov = torch.nanmean((xs - xm) * (ys - ym), dim=1, keepdim=True)
    var = torch.nanmean((xs - xm) ** 2, dim=1, keepdim=True).clamp_min(1e-8)
    slope = cov / var
    pred = ym + slope * (log_s.view(1, -1) - xm)
    ss_res = torch.nanmean((log_m - pred) ** 2, dim=1, keepdim=True)
    ss_tot = torch.nanmean((log_m - torch.nanmean(log_m, dim=1, keepdim=True)) ** 2, dim=1, keepdim=True).clamp_min(1e-8)
    r2 = (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    return torch.nan_to_num(slope), torch.nan_to_num(r2)


def estimate_robust_zeta(
    x: torch.Tensor,
    q_grid: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    scales: tuple[int, ...] = (2, 4, 8, 16, 32, 64),
    n_bootstrap: int = 8,
    smooth: bool = True,
) -> RobustZetaOutput:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    device, dtype = x.device, x.dtype
    q = torch.tensor(q_grid, device=device, dtype=dtype)
    valid_scales = [s for s in scales if s < x.shape[1] // 2]
    log_s = torch.log(torch.tensor(valid_scales, device=device, dtype=dtype))
    moments = []
    for s in valid_scales:
        inc = _increments_at_scale(x, s).abs().clamp_min(1e-8)
        moments.append(torch.stack([torch.mean(inc ** qq, dim=1) for qq in q], dim=1))
    m = torch.stack(moments, dim=2).clamp_min(1e-12)
    log_m = torch.log(m)
    z_ols, r2s = [], []
    z_trim = []
    for qi in range(len(q_grid)):
        slope_ols, r2 = _fit_slope(log_s, log_m[:, qi, :], trim=False)
        slope_trim, _ = _fit_slope(log_s, log_m[:, qi, :], trim=True)
        z_ols.append(slope_ols)
        z_trim.append(slope_trim)
        r2s.append(r2)
    zeta_ols = torch.cat(z_ols, dim=1)
    zeta_robust = torch.cat(z_trim, dim=1)
    if smooth and zeta_robust.shape[1] >= 3:
        z = zeta_robust
        zeta_robust = torch.cat([z[:, :1], 0.25 * z[:, :-2] + 0.5 * z[:, 1:-1] + 0.25 * z[:, 2:], z[:, -1:]], dim=1)
    boot = []
    if len(valid_scales) >= 4:
        g = torch.Generator(device=device)
        g.manual_seed(1234)
        for _ in range(n_bootstrap):
            idx = torch.randperm(len(valid_scales), generator=g, device=device)[: max(4, len(valid_scales) - 1)]
            idx = idx.sort().values
            slopes = []
            for qi in range(len(q_grid)):
                s, _ = _fit_slope(log_s[idx], log_m[:, qi, idx], trim=True)
                slopes.append(s)
            boot.append(torch.cat(slopes, dim=1))
    zeta_bootstrap_std = torch.stack(boot, dim=0).std(dim=0) if boot else torch.zeros_like(zeta_robust)
    scaling_fit_r2 = torch.cat(r2s, dim=1).mean(dim=1, keepdim=True)
    high_q_instability = zeta_bootstrap_std[:, -2:].mean(dim=1, keepdim=True).clamp(0.0, 1.0)
    scale_coverage = x.new_full((x.shape[0], 1), len(valid_scales) / max(len(scales), 1))
    warning_flags = ((high_q_instability > 0.25) | (scaling_fit_r2 < 0.5)).to(dtype)
    return RobustZetaOutput(
        zeta_robust=torch.nan_to_num(zeta_robust),
        zeta_ols=torch.nan_to_num(zeta_ols),
        zeta_bootstrap_std=torch.nan_to_num(zeta_bootstrap_std),
        scaling_fit_r2=torch.nan_to_num(scaling_fit_r2),
        high_q_instability=torch.nan_to_num(high_q_instability),
        scale_coverage=scale_coverage,
        warning_flags=warning_flags,
    )
