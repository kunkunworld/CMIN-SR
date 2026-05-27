# CMIN-SR Training

## Purpose

`CMIN-SR` is the first trainable model in the project whose primary task is:

- learn a stable empirical spectral representation,
- project that representation onto the MRW family,
- and output validity / mismatch signals.

It is **not** a pure MRW-only estimator.

## Dataset

Training uses `src/mrw_inverse/data/spectral_representation_dataset.py` with:

- `MRW`
- `Shuffled MRW`
- `fGn`
- `iid Gaussian`
- `iid Student-t`
- `GARCH(1,1)`
- `Regime-switching Gaussian`

The dataset provides:

- `zeta_target` when available
- `H_true`, `lambda2_true` for MRW
- soft targets for:
  - `p_scaling`
  - `p_MRW`
  - residual level
  - spectrum stability

## Training Objective

The loss mixes:

- empirical spectrum fitting
- MRW projection recovery on MRW samples
- monofractal regularization on `fGn` / Gaussian
- validity calibration for `p_scaling` and `p_MRW`
- residual calibration
- surrogate contrast on MRW vs shuffled MRW

This means non-MRW samples are not treated as trivial zero-spectrum nulls.

## Current First-Version Result

The first trained `CMIN-SR` already shows:

- high `p_scaling` on `MRW`, `fGn`, and Gaussian-like monofractal signals,
- clearly lower `p_MRW` on `Student-t` and regime-switching stress cases,
- positive MRW vs shuffled surrogate gaps,
- meaningful residual amplification on stress processes.

But the first version still has a real trade-off:

- `fGn` can still receive relatively high `p_MRW`,
- so boundary-mono vs true-MRW separation is not yet fully calibrated.

## CMIN-SR v2

`CMIN-SR v2` adds:

- explicit monofractal projection
- `p_mono`
- MRW-vs-mono gain supervision
- curvature significance calibration
- softer treatment of low-`lambda2` MRW boundary samples

Current v2 reading:

- `fGn p_MRW` does fall relative to v1,
- Gaussian `p_MRW` also falls,
- `Student-t` and regime-switching remain low-`p_MRW`,
- but MRW `p_MRW` is also pulled downward.

So v2 is a real semantic improvement, but not yet a final boundary-calibrated solution.
