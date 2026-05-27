from __future__ import annotations

import torch

from .curvature_diagnostics import compute_curvature_diagnostics
from .monofractal_projection import MonofractalProjection
from .spectral_geometry_calibrator import SpectralGeometryCalibrator
from .spectral_representation_model import CMINSRv3Model


class CMINSRZetaAlignedModel(CMINSRv3Model):
    """CMIN-SR variant whose training objective focuses on zeta_emp alignment."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mono_projection = MonofractalProjection(q_grid=self.q_grid)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | str]:
        out = super().forward(x)
        zeta_emp = out["zeta_emp"]
        mono = self.mono_projection(zeta_emp)
        mrw = self.projection(zeta_emp)
        q = torch.tensor(self.q_grid, dtype=zeta_emp.dtype, device=zeta_emp.device).view(1, -1)
        diag = compute_curvature_diagnostics(
            q_grid=q,
            zeta_emp=zeta_emp,
            zeta_mono=mono.zeta_mono,
            zeta_mrw=mrw.zeta_mrw_proj,
            residual_norm=mrw.residual_norm,
            mono_residual_norm=mono.mono_residual_norm,
            lambda2_proj=mrw.lambda2_proj,
        )
        zeta_uncertainty = torch.abs(zeta_emp - out["zeta_mrw"]).mean(dim=1, keepdim=True)
        out.update(
            {
                "zeta_target_optional": zeta_emp.new_full(zeta_emp.shape, float("nan")),
                "zeta_uncertainty": zeta_uncertainty,
                "H_proj": mrw.H_proj,
                "lambda2_proj": mrw.lambda2_proj,
                "H_mono": mono.H_mono,
                "zeta_mono": mono.zeta_mono,
                "zeta_mrw": mrw.zeta_mrw_proj,
                "residual_norm": mrw.residual_norm,
                "mono_residual_norm": mono.mono_residual_norm,
                "mrw_vs_mono_gain": diag.mrw_vs_mono_gain,
                "normalized_mrw_gain": diag.normalized_mrw_gain,
                "curvature_score": diag.curvature_score,
                "linearity_score": diag.linearity_score,
                "boundary_score": diag.boundary_mrw_score,
                "boundary_mrw_score": diag.boundary_mrw_score,
                "scaling_fit_quality": mrw.fit_quality,
                "mode": "cmin_sr_zeta_aligned",
            }
        )
        return out


def apply_pretrained_calibrator(outputs: dict[str, torch.Tensor | str], calibrator: SpectralGeometryCalibrator) -> dict[str, torch.Tensor]:
    cal = calibrator(
        outputs["zeta_emp"],
        outputs["zeta_mono"],
        outputs["zeta_mrw"],
        outputs["mono_residual_norm"],
        outputs["residual_norm"],
        outputs["mrw_vs_mono_gain"],
        outputs["normalized_mrw_gain"],
        outputs["curvature_score"],
        outputs["linearity_score"],
        outputs["boundary_score"],
        outputs["tail_instability"],
    )
    return {
        "p_scaling_cal": cal["p_scaling"],
        "p_curved_cal": cal["p_curved"],
        "p_mono_cal": cal["p_mono"],
        "p_mrw_cal": cal["p_mrw"],
        "p_boundary_cal": cal["p_boundary"],
    }
