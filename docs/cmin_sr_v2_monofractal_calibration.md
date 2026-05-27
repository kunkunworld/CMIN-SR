# CMIN-SR v2 Monofractal Calibration

## Why v2 Was Needed

`CMIN-SR` v1 already separated:

- empirical spectrum estimation
- MRW projection
- `p_scaling`
- `p_MRW`
- `residual_norm`

But it still had a clear semantic failure:

- `fGn` received very high `p_scaling`, which is correct,
- and also very high `p_MRW`, which is not ideal.

This meant the model was still confusing:

- stable monofractal scaling

with:

- genuinely MRW-like curved multifractal scaling.

## Core Change in v2

`CMIN-SR v2` introduces an explicit monofractal baseline:

- `zeta_mono(q) = q H_mono`

This creates a direct competition:

- MRW projection residual
- monofractal projection residual

The key interpretation becomes:

- `p_scaling`: is there a stable empirical scaling law?
- `p_MRW`: does the MRW family explain that spectrum better than a linear monofractal baseline?
- `p_mono`: is a linear monofractal explanation sufficient?

## New Outputs

Compared with v1, v2 adds:

- `H_mono`
- `zeta_mono`
- `mono_residual_norm`
- `mono_fit_quality`
- `mrw_vs_mono_gain`
- `curvature_significance`
- `p_mono`

## Training Change

The dataset now gives softer boundary labels:

- low-`lambda2` MRW is not forced to be "strong MRW"
- `fGn` and Gaussian are encouraged toward:
  - high `p_scaling`
  - high `p_mono`
  - low `p_MRW`
  - small MRW-vs-mono gain

The new loss emphasizes:

- monofractal projection quality on `fGn` / Gaussian
- MRW-vs-mono competition
- curvature calibration
- soft boundary handling for low-`lambda2` MRW

## Current Empirical Reading

The current v2 run shows a meaningful but incomplete improvement.

What improved:

- `fGn p_MRW` drops relative to v1
- Gaussian `p_MRW` also drops
- `Student-t` and regime-switching cases remain low-`p_MRW`

What remains unsolved:

- `fGn p_MRW` is still too high to claim clean monofractal-vs-MRW separation
- stronger monofractal calibration also lowers MRW `p_MRW`

So the main result of v2 is:

`explicit monofractal competition helps, but the boundary is still only partially calibrated`

## Why v3 Is Needed

v2 still makes `p_MRW` carry too many meanings at once:

- stable scaling
- MRW residual quality
- monofractal-vs-MRW gain
- curvature significance
- process validity

That coupling explains the partial-success pattern: `fGn p_MRW` drops, but
significant MRW `p_MRW` also drops. `CMIN-SR v3` fixes this by adding
`p_curved`, an explicit probability that asks only whether the empirical
spectrum is nonlinearly curved beyond a linear monofractal explanation.

The v3 interpretation becomes:

- `p_scaling`: stable scaling law?
- `p_curved`: significantly curved spectrum?
- `p_mono`: linear monofractal explanation?
- `p_MRW`: stable curved spectrum compatible with MRW?

## Practical Interpretation

For future raw-market analysis:

- large `lambda2_proj` alone is not enough
- high `p_scaling` alone is not enough
- the important new quantity is:
  - `mrw_vs_mono_gain`

If:

- `p_scaling` is high,
- but `mrw_vs_mono_gain` is small,
- and `p_mono` stays high,

then the signal is better interpreted as:

`stable scaling with weak evidence for genuinely MRW-like curvature`
