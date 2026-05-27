# Final Project Manifest

## Final Mainline

CMIN-SR should be presented as a validity-aware spectral diagnostic framework,
not as a guaranteed short-window `lambda2` recovery engine.

The final project story is:

```text
raw stochastic signal
  -> empirical zeta(q)
  -> monofractal and MRW projections
  -> projection residuals and curvature diagnostics
  -> calibrated validity/stability interpretation
  -> finite-sample identifiability limits
```

## Core Claim

Analytic spectral geometry is learnable and interpretable, but short-window raw
MRW curvature recovery is limited by empirical spectrum estimation. CMIN-SR is
therefore most defensible as a diagnostic framework that reports scaling,
curvature, monofractal compatibility, MRW compatibility, residual geometry, and
instability warnings.

## Final Experiments Used in the Paper

1. Spectral geometry calibration in analytic `zeta(q)` space.
2. Controlled stochastic family diagnostics after raw zeta alignment.
3. Monofractal-vs-MRW projection and boundary behavior.
4. Finite-sample curvature identifiability.
5. Optional real-world sanity check, only if data are available.

## Historical Versions Not Used as Main Experiments

- Tiny CMIN: useful as an always-MRW bias baseline.
- CMIN-Robust: useful anti-confounding and negative-control ablation.
- CMIN-SR v1/v2/v3: development progression.
- Boundary Calibration: same-H contrastive fine-tuning ablation.
- Raw Zeta Alignment: successful fake-curvature suppression, but over-linearized MRW.
- Curvature-Preserving Zeta Alignment: confirms the fGn rejection vs MRW recovery trade-off.

## Core Code Files

- `src/mrw_inverse/models/empirical_spectrum.py`
- `src/mrw_inverse/models/robust_zeta_estimator.py`
- `src/mrw_inverse/models/mrw_projection.py`
- `src/mrw_inverse/models/monofractal_projection.py`
- `src/mrw_inverse/models/curvature_diagnostics.py`
- `src/mrw_inverse/models/spectral_geometry_calibrator.py`
- `src/mrw_inverse/models/spectral_representation_model.py`
- `src/mrw_inverse/models/zeta_aligned_encoder.py`
- `src/mrw_inverse/data/analytic_spectrum_dataset.py`
- `src/mrw_inverse/data/raw_zeta_alignment_dataset.py`
- `src/mrw_inverse/data/boundary_calibration_dataset.py`
- `src/mrw_inverse/losses/spectrum_space_calibration_losses.py`
- `src/mrw_inverse/losses/zeta_alignment_losses.py`
- `src/mrw_inverse/losses/curvature_preserving_zeta_losses.py`
- `src/mrw_inverse/analysis/curvature_identifiability.py`

## Core Experiment Scripts

- `experiments/evaluate_spectral_geometry_calibrator.py`
- `experiments/evaluate_curvature_preserving_zeta_alignment.py`
- `experiments/evaluate_cmin_sr_v3.py`
- `experiments/run_finite_sample_curvature_identifiability.py`
- `experiments/run_scale_length_sensitivity.py`
- `experiments/run_qgrid_sensitivity.py`
- `experiments/run_zeta_noise_bridge.py`
- `experiments/run_cmin_sr_failure_attribution.py`
- `experiments/paper/generate_all_paper_assets.py`

## Core Output Tables and Figures

- `outputs/tables/spectral_geometry_calibrator_eval/summary_by_spectrum_type.csv`
- `outputs/tables/spectral_geometry_calibrator_eval/lambda2_sweep.csv`
- `outputs/tables/curvature_preserving_zeta_eval/process_by_T_band.csv`
- `outputs/tables/cmin_sr_v3_eval/process_by_T.csv`
- `outputs/tables/finite_sample_identifiability/lambda2_recovery_by_T.csv`
- `outputs/tables/finite_sample_identifiability/lambda2_recovery_by_qgrid.csv`
- `outputs/tables/zeta_noise_bridge/separation_margin_vs_noise.csv`
- `outputs/tables/cmin_sr_failure_attribution/failure_attribution_summary_table.csv`
- `paper_assets/figures/`
- `paper_assets/tables/`

## Checkpoints

- `checkpoints/cmin/cmin_sr_synthetic.pt`
- `checkpoints/cmin/cmin_sr_v2_synthetic.pt`
- `checkpoints/cmin/cmin_sr_v3_synthetic.pt`
- `checkpoints/cmin/cmin_sr_calibrated_synthetic.pt`
- `checkpoints/cmin/spectral_geometry_calibrator.pt`
- `checkpoints/cmin/cmin_sr_zeta_aligned.pt`
- `checkpoints/cmin/cmin_sr_zeta_curvature_preserved.pt`
- `checkpoints/cmin/cmin_robust_synthetic.pt`
- `checkpoints/cmin/cmin_robust_multilength.pt`
- `checkpoints/cmin/cmin_tiny_synthetic.pt`

## Recommended Main-Paper Results

- Analytic spectral geometry calibration succeeds.
- Boundary MRW is distinct from strong MRW and from linear monofractal spectra.
- Controlled stochastic families require residual and instability diagnostics.
- Deterministic finite-sample estimators do not reliably recover `lambda2` at short T.

## Appendix / Failure Analysis

- v1/v2/v3 progression.
- Boundary calibration negative result.
- Raw zeta alignment trade-off.
- Curvature-preserving zeta alignment trade-off.
- Failure attribution: analytic vs deterministic vs neural zeta.
- Zeta noise bridge.

## Legacy / Not Recommended for Main Text

- Old proxy estimators as primary claims.
- Volatility forecasting as central validation.
- Strong claims that `lambda2_proj` proves a true MRW mechanism.
- Strong claims that short-window raw samples allow guaranteed MRW curvature recovery.
