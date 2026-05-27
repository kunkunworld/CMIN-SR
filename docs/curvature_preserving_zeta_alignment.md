# Curvature-Preserving Zeta Alignment

## Motivation

Raw zeta alignment suppressed fake curvature for fGn and Gaussian, but it also
flattened true MRW curvature. This stage tries to preserve MRW curvature without
reintroducing monofractal false positives.

## Loss Changes

The curvature-preserving loss adds:

- band-specific MRW curvature matching:
  - low lambda2: weak curvature weight
  - medium lambda2: stronger curvature weight
  - high lambda2: strongest curvature weight
- stronger lambda2 projection consistency for medium/high MRW;
- MRW-vs-mono residual margin for medium/high MRW;
- third-difference smoothness, which penalizes jagged spectra while allowing
  quadratic MRW curvature;
- conditional monofractal linearity preservation for fGn/Gaussian.

## Current Result

The stage is informative but not final. Two training balances were observed:

- emphasizing MRW curvature restores MRW `p_MRW_cal`, but reintroduces high
  fGn `p_MRW_cal`;
- strengthening monofractal preservation keeps fGn/Gaussian controlled, but
  MRW medium/high curvature remains under-recovered.

The final saved run preserves fGn/Gaussian control:

- fGn `p_MRW_cal` around `0.31`
- Gaussian `p_MRW_cal` around `0.13`

but MRW medium/high remains too low:

- medium MRW `p_MRW_cal` around `0.34`
- high MRW `p_MRW_cal` around `0.31`

## Interpretation

This is not enough for the final main experiment. The bottleneck now appears to
be finite-sample curvature identifiability under the current q-grid / scale
range / structure-function estimator, rather than validity-head design.

The follow-up identifiability study confirms this direction: deterministic
estimators have near-zero `lambda2_true` vs `lambda2_proj` correlation in the
quick finite-sample grid, so high-lambda MRW curvature is not reliably available
to either the deterministic or neural pipeline at short T.
