from __future__ import annotations

import torch
from torch import nn

from .cmin import CMINRegressor
from .curvature_diagnostics import compute_curvature_diagnostics
from .empirical_spectrum import estimate_empirical_spectrum
from .logvol_covariance import LogVolCovarianceBranch
from .monofractal_projection import MonofractalProjection
from .mrw_projection import MRWProjection
from .multiscale_features import ScaleGraphBranch, StructureFunctionBranch
from mrw_dl.models import TCNBlock


class SpectralRepresentationModel(nn.Module):
    """High-level spectral representation wrapper.

    This layer separates:
    1. empirical multiscale spectrum estimation
    2. MRW parametric projection
    3. validity / mismatch interpretation
    """

    def __init__(
        self,
        q_grid: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
        scales: tuple[int, ...] = (2, 4, 8, 16, 32, 64),
        cmin_model: CMINRegressor | None = None,
    ) -> None:
        super().__init__()
        self.q_grid = q_grid
        self.scales = scales
        self.projection = MRWProjection(q_grid=q_grid)
        self.cmin_model = cmin_model

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | str]:
        emp = estimate_empirical_spectrum(x, q_grid=self.q_grid, scales=self.scales)
        proj = self.projection(emp.zeta_emp)
        spectral_embedding = torch.cat(
            [
                emp.zeta_emp,
                emp.alpha_emp,
                emp.spectrum_width,
                emp.spectrum_curvature,
                emp.scaling_fit_quality,
                emp.spectrum_stability,
                proj.residual_norm,
            ],
            dim=1,
        )

        heuristic_p_mrw = (
            0.30 * proj.fit_quality
            + 0.25 * proj.projection_gain
            + 0.25 * emp.p_scaling
            + 0.20 * emp.spectrum_stability
            - 0.30 * emp.tail_instability
        ).clamp(0.0, 1.0)
        if self.cmin_model is not None:
            with torch.no_grad():
                cmin_out = self.cmin_model(x if x.ndim == 2 else x.unsqueeze(0))
            p_mrw = (0.5 * heuristic_p_mrw + 0.5 * cmin_out.p_mrw).clamp(0.0, 1.0)
            mode = "hybrid_statistical_plus_cmin"
        else:
            p_mrw = heuristic_p_mrw
            mode = "deterministic_empirical_projection"

        return {
            "zeta_emp": emp.zeta_emp,
            "alpha_emp": emp.alpha_emp,
            "f_emp": emp.f_emp,
            "spectral_embedding": spectral_embedding,
            "H_proj": proj.H_proj,
            "lambda2_proj": proj.lambda2_proj,
            "zeta_mrw": proj.zeta_mrw_proj,
            "alpha_mrw": proj.alpha_mrw_proj,
            "f_mrw": proj.f_mrw_proj,
            "residual": proj.residual,
            "residual_norm": proj.residual_norm,
            "p_scaling": emp.p_scaling,
            "p_mrw": p_mrw,
            "spectrum_stability": emp.spectrum_stability,
            "scaling_fit_quality": emp.scaling_fit_quality,
            "spectrum_width": emp.spectrum_width,
            "spectrum_curvature": emp.spectrum_curvature,
            "tail_instability": emp.tail_instability,
            "projection_gain": proj.projection_gain,
            "mode": mode,
        }


class CMINSRModel(nn.Module):
    """Trainable stable spectral representation learner.

    The model learns a process-agnostic empirical spectrum representation first,
    then projects it onto the MRW family, instead of directly treating every
    signal as a pure MRW-parameter regression problem.
    """

    def __init__(
        self,
        q_grid: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
        scales: tuple[int, ...] = (2, 4, 8, 16, 32, 64),
        lags: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.q_grid = q_grid
        self.q_len = len(q_grid)
        self.structure_branch = StructureFunctionBranch(scales=scales, q_grid=q_grid)
        self.logvol_branch = LogVolCovarianceBranch(lags=lags)
        self.scale_graph = ScaleGraphBranch(num_scales=len(scales), hidden_dim=hidden_dim, dropout=dropout)
        self.projection = MRWProjection(q_grid=q_grid)

        raw_channels = 32
        self.raw_stem = nn.Sequential(
            nn.Conv1d(1, raw_channels, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(raw_channels),
            nn.GELU(),
            TCNBlock(raw_channels, dilation=1, dropout=dropout),
            TCNBlock(raw_channels, dilation=2, dropout=dropout),
            TCNBlock(raw_channels, dilation=4, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(raw_channels, hidden_dim),
            nn.GELU(),
        )

        structure_dim = self.q_len * len(scales) + self.q_len + self.q_len
        self.structure_head = nn.Sequential(
            nn.LayerNorm(structure_dim + hidden_dim),
            nn.Linear(structure_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        logvol_dim = len(lags) + 1
        self.logvol_head = nn.Sequential(
            nn.LayerNorm(logvol_dim + hidden_dim),
            nn.Linear(logvol_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.zeta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.q_len),
        )
        self.p_scaling_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.p_mrw_head = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.stability_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | str]:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        structure = self.structure_branch(x)
        logvol = self.logvol_branch(x)
        graph = self.scale_graph(
            structure["cross_scale_corr"],
            structure["wavelet_cumulants"],
            structure["log_scales"],
        )
        raw = self.raw_stem(x.unsqueeze(1))
        structure_vec = torch.cat(
            [
                structure["log_structure_matrix"].reshape(x.shape[0], -1),
                structure["empirical_zeta_estimate"],
                structure["q_curvature"],
            ],
            dim=1,
        )
        structure_embed = self.structure_head(torch.cat([structure_vec, raw], dim=1))
        logvol_vec = torch.cat([logvol["logvol_covariance_curve"], logvol["logvol_covariance_slope"]], dim=1)
        logvol_embed = self.logvol_head(torch.cat([logvol_vec, raw], dim=1))
        fused = self.fusion(torch.cat([structure_embed, 0.5 * graph + logvol_embed], dim=1))

        zeta_prior = structure["empirical_zeta_estimate"]
        zeta_correction = 0.25 * self.zeta_head(fused)
        zeta_emp = zeta_prior + zeta_correction
        q = torch.tensor(self.q_grid, dtype=zeta_emp.dtype, device=zeta_emp.device).view(1, -1)
        alpha_emp = torch.gradient(zeta_emp, spacing=(q[0],), dim=1)[0]
        f_emp = 1.0 + q * alpha_emp - zeta_emp

        proj = self.projection(zeta_emp)
        tail_instability = ((f_emp.max(dim=1, keepdim=True).values - f_emp.min(dim=1, keepdim=True).values - 0.3) / 1.5).clamp(0.0, 1.0)
        p_scaling = torch.sigmoid(self.p_scaling_head(fused))
        spectrum_stability = torch.sigmoid(self.stability_head(fused))
        mrw_context = torch.cat(
            [
                fused,
                proj.fit_quality,
                proj.projection_gain,
                proj.residual_norm,
                spectrum_stability,
            ],
            dim=1,
        )
        p_mrw_raw = torch.sigmoid(self.p_mrw_head(mrw_context))
        p_mrw = (p_mrw_raw - 0.25 * tail_instability).clamp(0.0, 1.0)

        spectral_embedding = torch.cat(
            [
                zeta_emp,
                alpha_emp,
                proj.residual_norm,
                p_scaling,
                spectrum_stability,
                proj.projection_gain,
            ],
            dim=1,
        )
        return {
            "zeta_emp": zeta_emp,
            "alpha_emp": alpha_emp,
            "f_emp": f_emp,
            "spectral_embedding": spectral_embedding,
            "H_proj": proj.H_proj,
            "lambda2_proj": proj.lambda2_proj,
            "zeta_mrw": proj.zeta_mrw_proj,
            "alpha_mrw": proj.alpha_mrw_proj,
            "f_mrw": proj.f_mrw_proj,
            "residual": proj.residual,
            "residual_norm": proj.residual_norm,
            "p_scaling": p_scaling,
            "p_mrw": p_mrw,
            "spectrum_stability": spectrum_stability,
            "scaling_fit_quality": proj.fit_quality,
            "spectrum_width": (alpha_emp.max(dim=1, keepdim=True).values - alpha_emp.min(dim=1, keepdim=True).values).clamp_min(0.0),
            "spectrum_curvature": (-torch.gradient(torch.gradient(zeta_emp, spacing=(q[0],), dim=1)[0], spacing=(q[0],), dim=1)[0].mean(dim=1, keepdim=True)).clamp_min(0.0),
            "tail_instability": tail_instability,
            "projection_gain": proj.projection_gain,
            "mode": "trainable_cmin_sr",
        }


class CMINSRv2Model(CMINSRModel):
    """CMIN-SR v2 with explicit MRW-vs-monofractal competition."""

    def __init__(
        self,
        q_grid: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
        scales: tuple[int, ...] = (2, 4, 8, 16, 32, 64),
        lags: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(q_grid=q_grid, scales=scales, lags=lags, hidden_dim=hidden_dim, dropout=dropout)
        self.mono_projection = MonofractalProjection(q_grid=q_grid)
        self.p_mono_head = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | str]:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        structure = self.structure_branch(x)
        logvol = self.logvol_branch(x)
        graph = self.scale_graph(
            structure["cross_scale_corr"],
            structure["wavelet_cumulants"],
            structure["log_scales"],
        )
        raw = self.raw_stem(x.unsqueeze(1))
        structure_vec = torch.cat(
            [
                structure["log_structure_matrix"].reshape(x.shape[0], -1),
                structure["empirical_zeta_estimate"],
                structure["q_curvature"],
            ],
            dim=1,
        )
        structure_embed = self.structure_head(torch.cat([structure_vec, raw], dim=1))
        logvol_vec = torch.cat([logvol["logvol_covariance_curve"], logvol["logvol_covariance_slope"]], dim=1)
        logvol_embed = self.logvol_head(torch.cat([logvol_vec, raw], dim=1))
        fused = self.fusion(torch.cat([structure_embed, 0.5 * graph + logvol_embed], dim=1))

        zeta_prior = structure["empirical_zeta_estimate"]
        zeta_correction = 0.25 * self.zeta_head(fused)
        zeta_emp = zeta_prior + zeta_correction
        q = torch.tensor(self.q_grid, dtype=zeta_emp.dtype, device=zeta_emp.device).view(1, -1)
        alpha_emp = torch.gradient(zeta_emp, spacing=(q[0],), dim=1)[0]
        f_emp = 1.0 + q * alpha_emp - zeta_emp
        spectrum_width = (alpha_emp.max(dim=1, keepdim=True).values - alpha_emp.min(dim=1, keepdim=True).values).clamp_min(0.0)
        spectrum_curvature = (-torch.gradient(torch.gradient(zeta_emp, spacing=(q[0],), dim=1)[0], spacing=(q[0],), dim=1)[0].mean(dim=1, keepdim=True)).clamp_min(0.0)

        mrw = self.projection(zeta_emp)
        mono = self.mono_projection(zeta_emp)
        mrw_vs_mono_gain = ((mono.mono_residual_norm - mrw.residual_norm) / mono.mono_residual_norm.clamp_min(1e-6)).clamp(-1.0, 1.0)
        curvature_significance = (mrw_vs_mono_gain * torch.sqrt(mrw.lambda2_proj + 1e-8) * 4.0).clamp(0.0, 1.0)
        tail_instability = ((spectrum_width - 0.3) / 1.5).clamp(0.0, 1.0)

        p_scaling = torch.sigmoid(self.p_scaling_head(fused))
        spectrum_stability = torch.sigmoid(self.stability_head(fused))

        mrw_context = torch.cat(
            [
                fused,
                mrw.fit_quality,
                mrw_vs_mono_gain,
                curvature_significance,
                spectrum_stability,
            ],
            dim=1,
        )
        mono_context = torch.cat(
            [
                fused,
                mono.mono_fit_quality,
                mono.mono_residual_norm,
                1.0 - mrw_vs_mono_gain.clamp(0.0, 1.0),
                spectrum_stability,
            ],
            dim=1,
        )
        p_mrw_raw = torch.sigmoid(self.p_mrw_head(mrw_context))
        p_mono_raw = torch.sigmoid(self.p_mono_head(mono_context))
        competition_gate = torch.sigmoid(10.0 * (mrw_vs_mono_gain - 0.12) + 8.0 * (curvature_significance - 0.10))
        p_mrw = (
            0.35 * p_mrw_raw
            + 0.45 * p_scaling * competition_gate
            + 0.10 * mrw.fit_quality
            - 0.25 * tail_instability
            - 0.25 * p_mono_raw
        ).clamp(0.0, 1.0)
        p_mono = (
            p_mono_raw
            + 0.25 * mono.mono_fit_quality
            - 0.20 * mrw_vs_mono_gain.clamp(0.0, 1.0)
            + 0.10 * p_scaling
        ).clamp(0.0, 1.0)

        spectral_embedding = torch.cat(
            [
                zeta_emp,
                alpha_emp,
                mrw.residual_norm,
                mono.mono_residual_norm,
                p_scaling,
                p_mrw,
                p_mono,
                mrw_vs_mono_gain,
            ],
            dim=1,
        )
        return {
            "zeta_emp": zeta_emp,
            "alpha_emp": alpha_emp,
            "f_emp": f_emp,
            "spectral_embedding": spectral_embedding,
            "H_proj": mrw.H_proj,
            "lambda2_proj": mrw.lambda2_proj,
            "zeta_mrw": mrw.zeta_mrw_proj,
            "alpha_mrw": mrw.alpha_mrw_proj,
            "f_mrw": mrw.f_mrw_proj,
            "residual": mrw.residual,
            "residual_norm": mrw.residual_norm,
            "p_scaling": p_scaling,
            "p_mrw": p_mrw,
            "p_mono": p_mono,
            "spectrum_stability": spectrum_stability,
            "scaling_fit_quality": mrw.fit_quality,
            "spectrum_width": spectrum_width,
            "spectrum_curvature": spectrum_curvature,
            "tail_instability": tail_instability,
            "projection_gain": mrw.projection_gain,
            "H_mono": mono.H_mono,
            "zeta_mono": mono.zeta_mono,
            "mono_residual": mono.mono_residual,
            "mono_residual_norm": mono.mono_residual_norm,
            "mono_fit_quality": mono.mono_fit_quality,
            "mrw_vs_mono_gain": mrw_vs_mono_gain,
            "curvature_significance": curvature_significance,
            "mode": "trainable_cmin_sr_v2",
        }


class CMINSRv3Model(CMINSRv2Model):
    """CMIN-SR v3 with explicit curved-vs-linear calibration."""

    def __init__(
        self,
        q_grid: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
        scales: tuple[int, ...] = (2, 4, 8, 16, 32, 64),
        lags: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(q_grid=q_grid, scales=scales, lags=lags, hidden_dim=hidden_dim, dropout=dropout)
        diag_dim = 6
        self.p_curved_head = nn.Sequential(
            nn.Linear(hidden_dim + diag_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.p_boundary_head = nn.Sequential(
            nn.Linear(hidden_dim + diag_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.p_mrw_v3_head = nn.Sequential(
            nn.Linear(hidden_dim + 9, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | str]:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        structure = self.structure_branch(x)
        logvol = self.logvol_branch(x)
        graph = self.scale_graph(
            structure["cross_scale_corr"],
            structure["wavelet_cumulants"],
            structure["log_scales"],
        )
        raw = self.raw_stem(x.unsqueeze(1))
        structure_vec = torch.cat(
            [
                structure["log_structure_matrix"].reshape(x.shape[0], -1),
                structure["empirical_zeta_estimate"],
                structure["q_curvature"],
            ],
            dim=1,
        )
        structure_embed = self.structure_head(torch.cat([structure_vec, raw], dim=1))
        logvol_vec = torch.cat([logvol["logvol_covariance_curve"], logvol["logvol_covariance_slope"]], dim=1)
        logvol_embed = self.logvol_head(torch.cat([logvol_vec, raw], dim=1))
        fused = self.fusion(torch.cat([structure_embed, 0.5 * graph + logvol_embed], dim=1))

        zeta_prior = structure["empirical_zeta_estimate"]
        zeta_correction = 0.25 * self.zeta_head(fused)
        zeta_emp = zeta_prior + zeta_correction
        q = torch.tensor(self.q_grid, dtype=zeta_emp.dtype, device=zeta_emp.device).view(1, -1)
        alpha_emp = torch.gradient(zeta_emp, spacing=(q[0],), dim=1)[0]
        f_emp = 1.0 + q * alpha_emp - zeta_emp
        spectrum_width = (alpha_emp.max(dim=1, keepdim=True).values - alpha_emp.min(dim=1, keepdim=True).values).clamp_min(0.0)
        spectrum_curvature = (-torch.gradient(torch.gradient(zeta_emp, spacing=(q[0],), dim=1)[0], spacing=(q[0],), dim=1)[0].mean(dim=1, keepdim=True)).clamp_min(0.0)

        mrw = self.projection(zeta_emp)
        mono = self.mono_projection(zeta_emp)
        diags = compute_curvature_diagnostics(
            q_grid=q,
            zeta_emp=zeta_emp,
            zeta_mono=mono.zeta_mono,
            zeta_mrw=mrw.zeta_mrw_proj,
            residual_norm=mrw.residual_norm,
            mono_residual_norm=mono.mono_residual_norm,
            lambda2_proj=mrw.lambda2_proj,
        )
        tail_instability = ((spectrum_width - 0.3) / 1.5).clamp(0.0, 1.0)

        p_scaling = torch.sigmoid(self.p_scaling_head(fused))
        spectrum_stability = torch.sigmoid(self.stability_head(fused))
        diag_features = torch.cat(
            [
                diags.curvature_score,
                diags.linearity_score,
                diags.normalized_mrw_gain.clamp(-1.0, 1.0),
                diags.curvature_confidence,
                diags.boundary_mrw_score,
                1.0 - tail_instability,
            ],
            dim=1,
        )
        p_curved = torch.sigmoid(self.p_curved_head(torch.cat([fused, diag_features], dim=1)))
        boundary_mrw_score = (
            0.65 * torch.sigmoid(self.p_boundary_head(torch.cat([fused, diag_features], dim=1)))
            + 0.35 * diags.boundary_mrw_score
        ).clamp(0.0, 1.0)

        mono_context = torch.cat(
            [
                fused,
                mono.mono_fit_quality,
                mono.mono_residual_norm,
                1.0 - diags.mrw_vs_mono_gain,
                spectrum_stability,
            ],
            dim=1,
        )
        p_mono_raw = torch.sigmoid(self.p_mono_head(mono_context))
        p_mono = (
            0.55 * p_mono_raw
            + 0.20 * mono.mono_fit_quality
            + 0.15 * diags.linearity_score
            + 0.10 * p_scaling
            - 0.20 * p_curved
        ).clamp(0.0, 1.0)

        mrw_context = torch.cat(
            [
                fused,
                mrw.fit_quality,
                diags.mrw_vs_mono_gain,
                diags.curvature_significance,
                p_curved,
                p_mono,
                p_scaling,
                boundary_mrw_score,
                spectrum_stability,
                1.0 - tail_instability,
            ],
            dim=1,
        )
        p_mrw_raw = torch.sigmoid(self.p_mrw_v3_head(mrw_context))
        p_mrw = (
            0.45 * p_mrw_raw
            + 0.25 * p_curved
            + 0.15 * p_scaling
            + 0.10 * diags.mrw_vs_mono_gain
            + 0.10 * diags.curvature_significance
            + 0.10 * boundary_mrw_score
            - 0.30 * p_mono
            - 0.15 * tail_instability
        ).clamp(0.0, 1.0)

        spectral_embedding = torch.cat(
            [
                zeta_emp,
                alpha_emp,
                mrw.residual_norm,
                mono.mono_residual_norm,
                p_scaling,
                p_curved,
                p_mono,
                p_mrw,
                diags.mrw_vs_mono_gain,
                boundary_mrw_score,
            ],
            dim=1,
        )
        return {
            "zeta_emp": zeta_emp,
            "alpha_emp": alpha_emp,
            "f_emp": f_emp,
            "spectral_embedding": spectral_embedding,
            "H_proj": mrw.H_proj,
            "lambda2_proj": mrw.lambda2_proj,
            "zeta_mrw": mrw.zeta_mrw_proj,
            "alpha_mrw": mrw.alpha_mrw_proj,
            "f_mrw": mrw.f_mrw_proj,
            "residual": mrw.residual,
            "residual_norm": mrw.residual_norm,
            "p_scaling": p_scaling,
            "p_curved": p_curved,
            "p_mrw": p_mrw,
            "p_mono": p_mono,
            "spectrum_stability": spectrum_stability,
            "scaling_fit_quality": mrw.fit_quality,
            "spectrum_width": spectrum_width,
            "spectrum_curvature": spectrum_curvature,
            "tail_instability": tail_instability,
            "projection_gain": mrw.projection_gain,
            "H_mono": mono.H_mono,
            "zeta_mono": mono.zeta_mono,
            "mono_residual": mono.mono_residual,
            "mono_residual_norm": mono.mono_residual_norm,
            "mono_fit_quality": mono.mono_fit_quality,
            "mrw_vs_mono_gain": diags.mrw_vs_mono_gain,
            "normalized_mrw_gain": diags.normalized_mrw_gain,
            "curvature_score": diags.curvature_score,
            "curvature_significance": diags.curvature_significance,
            "curvature_confidence": diags.curvature_confidence,
            "linearity_score": diags.linearity_score,
            "boundary_mrw_score": boundary_mrw_score,
            "mode": "trainable_cmin_sr_v3",
        }
