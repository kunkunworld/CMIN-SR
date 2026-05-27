# Core Code Index

## A. Empirical Spectrum Estimation

### `src/mrw_inverse/models/empirical_spectrum.py`
- Purpose: baseline empirical spectrum estimation from raw signals.
- Status: paper-supporting core.
- Main API: empirical spectrum utilities used by CMIN-SR diagnostics.
- Related scripts: spectral representation and CMIN-SR evaluation scripts.

### `src/mrw_inverse/models/robust_zeta_estimator.py`
- Purpose: deterministic robust structure-function zeta estimator.
- Status: paper-supporting diagnostic core.
- Main API: robust zeta estimates, bootstrap uncertainty, high-q warnings.
- Related scripts: raw zeta alignment and finite-sample identifiability.

## B. Projection Modules

### `src/mrw_inverse/models/mrw_projection.py`
- Purpose: project spectra onto the two-parameter MRW family.
- Status: paper core.
- Main API: `MRWProjection`.
- Related scripts: CMIN-SR eval, zeta alignment eval, identifiability study.

### `src/mrw_inverse/models/monofractal_projection.py`
- Purpose: project spectra onto linear monofractal spectra.
- Status: paper core.
- Main API: `MonofractalProjection`.
- Related scripts: CMIN-SR v2/v3, boundary and identifiability scripts.

## C. Spectral Interpretation

### `src/mrw_inverse/models/curvature_diagnostics.py`
- Purpose: deterministic curvature, linearity, gain, and boundary diagnostics.
- Status: paper core.
- Main API: `compute_curvature_diagnostics`.
- Related scripts: CMIN-SR v3+, zeta alignment, finite-sample identifiability.

### `src/mrw_inverse/models/spectral_geometry_calibrator.py`
- Purpose: small spectrum-space calibrator for `p_scaling`, `p_curved`,
  `p_mono`, `p_MRW`, and `p_boundary`.
- Status: paper core.
- Main classes: `SpectralGeometryCalibrator`.
- Related scripts: train/evaluate/apply spectral geometry calibrator.

## D. CMIN-SR Models

### `src/mrw_inverse/models/spectral_representation_model.py`
- Purpose: CMIN-SR model family and spectral representation outputs.
- Status: core plus historical ablation support.
- Main classes: CMIN-SR variants and wrappers.
- Related scripts: CMIN-SR v1/v2/v3 training and evaluation.

### `src/mrw_inverse/models/zeta_aligned_encoder.py`
- Purpose: raw signal to aligned `zeta_emp(q)` and projection features.
- Status: ablation/failure-analysis core.
- Main classes: zeta-aligned CMIN-SR encoder.
- Related scripts: raw zeta alignment and curvature-preserving alignment.

## E. Datasets

### `src/mrw_inverse/data/analytic_spectrum_dataset.py`
- Purpose: analytic/semi-analytic zeta-space training data.
- Status: paper core.
- Related scripts: spectral geometry calibrator training/eval.

### `src/mrw_inverse/data/raw_zeta_alignment_dataset.py`
- Purpose: raw time series with analytic zeta targets.
- Status: ablation/failure-analysis core.
- Related scripts: raw zeta alignment.

### `src/mrw_inverse/data/boundary_calibration_dataset.py`
- Purpose: same-H fGn/MRW lambda2 sweep calibration data.
- Status: historical ablation.
- Related scripts: boundary calibration training/eval.

## F. Losses

### `src/mrw_inverse/losses/spectral_representation_losses.py`
- Purpose: original CMIN-SR spectral representation losses.
- Status: historical and ablation support.

### `src/mrw_inverse/losses/zeta_alignment_losses.py`
- Purpose: raw `zeta_emp(q)` alignment losses.
- Status: ablation/failure-analysis core.

### `src/mrw_inverse/losses/curvature_preserving_zeta_losses.py`
- Purpose: band-specific curvature-preserving zeta alignment losses.
- Status: ablation/failure-analysis core.

### `src/mrw_inverse/losses/spectrum_space_calibration_losses.py`
- Purpose: supervised and ranking losses for analytic spectrum-space calibration.
- Status: paper core.

## G. Analysis

### `src/mrw_inverse/analysis/curvature_identifiability.py`
- Purpose: deterministic estimator-level MRW curvature recovery analysis.
- Status: paper core.
- Main API: `estimate_curvature_identifiability`, `estimate_many`.
- Related scripts: finite-sample identifiability, scale sensitivity, q-grid sensitivity, failure attribution.
