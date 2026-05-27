# Zeta Alignment Comparison

## Compared Stages

- CMIN-SR v3 + spectrum-space calibrator
- Zeta-aligned CMIN-SR + spectrum-space calibrator

## Main Finding

Raw zeta alignment reduces spurious curvature for fGn and Gaussian, but it also
collapses MRW curvature too much. This means the model learned a conservative
linearizing solution.

At `T=1024`:

- fGn `p_MRW_cal`: about `0.73 -> 0.36`
- Gaussian `p_MRW_cal`: about `0.60 -> 0.21`
- MRW `p_MRW_cal`: about `0.69 -> 0.35`

The result is useful as an ablation, but it is not yet a final main experiment.

Curvature-preserving alignment tests the next trade-off. The final conservative
run keeps fGn/Gaussian controlled but does not recover MRW medium/high
curvature enough for a main-table result.

## Next Direction

The next improvement should target curvature-preserving zeta estimation:

- stronger MRW curvature loss for medium/high lambda2;
- richer q-grid or longer scale range;
- bootstrap / smoothing that suppresses jaggedness without flattening true
  curvature;
- wavelet leader or MFDFA-style zeta estimator;
- report finite-sample weak-intermittency identifiability if separation remains
  impossible.
