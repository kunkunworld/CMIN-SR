# Spectral Geometry Calibrator

## Files

- `src/mrw_inverse/data/analytic_spectrum_dataset.py`
- `src/mrw_inverse/models/spectral_geometry_calibrator.py`
- `src/mrw_inverse/losses/spectrum_space_calibration_losses.py`
- `experiments/train_spectral_geometry_calibrator.py`
- `experiments/evaluate_spectral_geometry_calibrator.py`
- `experiments/apply_spectral_calibrator_to_cmin_sr.py`

## Input

The calibrator consumes spectrum-space features:

- `zeta_input`
- `zeta_mono`
- `zeta_mrw`
- `mono_residual_norm`
- `mrw_residual_norm`
- `mrw_vs_mono_gain`
- `normalized_mrw_gain`
- `curvature_score`
- `linearity_score`
- `boundary_score`
- `tail_instability`

## Output

- `p_scaling`
- `p_curved`
- `p_mono`
- `p_mrw`
- `p_boundary`
- `calibrated_mrw_score`
- `calibrated_curvature_score`

The model is intentionally small: a 2-layer MLP with sigmoid heads.

## Checkpoint

```text
checkpoints/cmin/spectral_geometry_calibrator.pt
```

## Current Finding

The calibrator works in analytic spectrum space. When applied to raw CMIN-SR
outputs, it exposes that raw empirical spectra still make `fGn` look like a
boundary/noisy-MRW spectrum. The next useful work is better `zeta_emp`
estimation, not more validity-head complexity.

Raw zeta alignment is the first such attempt. It reduces fGn/Gaussian false
curvature but currently over-flattens MRW, so the next improvement should be
curvature-preserving spectrum estimation.

Curvature-preserving zeta alignment keeps the calibrator unchanged. Its mixed
result reinforces that the calibrator is not the bottleneck; the hard part is
estimating a finite-sample `zeta_emp` that is linear for fGn but curved for
medium/high MRW.
