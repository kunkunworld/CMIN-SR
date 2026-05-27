# Final Summary for Paper

## Final Story in 10 Points

1. CMIN-SR studies whether finite stochastic signals support an MRW-like spectral interpretation.
2. The framework estimates empirical `zeta(q)` and compares monofractal and MRW projections.
3. `lambda2_proj` is a projection coordinate, not proof of a true MRW mechanism.
4. Analytic spectral geometry calibration succeeds in clean `zeta(q)` space.
5. Boundary MRW, strong MRW, and linear monofractal spectra are separable analytically.
6. Raw finite-sample `zeta_emp(q)` is the bottleneck.
7. Raw zeta alignment suppresses false fGn/Gaussian curvature but can flatten true MRW curvature.
8. Curvature-preserving zeta alignment exposes a trade-off, not a final fix.
9. Deterministic estimators also struggle to recover `lambda2` at short T.
10. The final contribution is a validity-aware diagnostic framework plus an honest finite-sample limitation analysis.

## Main Claim

CMIN-SR provides a calibrated spectral diagnostic pipeline for deciding whether
finite stochastic signals support linear monofractal, boundary MRW, or curved
MRW-like interpretations, while explicitly reporting finite-sample uncertainty
and instability.

## What Not to Claim

- Do not claim guaranteed short-window `lambda2` recovery.
- Do not claim `lambda2_proj` proves a true MRW generative mechanism.
- Do not claim raw market data has a confirmed MRW mechanism from high `p_MRW` alone.
- Do not present volatility forecasting as the central validation.

## Recommended Paper Title

**Validity-Aware Spectral Diagnostics for Multifractal Random Walk Inference Under Finite Samples**

## Recommended Abstract Points

- MRW spectral inference is under-identified in short finite samples.
- Separating spectrum estimation from spectral interpretation clarifies the bottleneck.
- Analytic spectral geometry is learnable by a small calibrator.
- Empirical zeta estimation limits raw MRW curvature recovery.
- The framework reports calibrated diagnostics rather than overclaiming mechanism recovery.

## Recommended Main Figures

1. CMIN-SR framework diagram.
2. Spectral geometry calibration map.
3. Process-family diagnostic map.
4. MRW vs monofractal projection residual geometry.
5. Finite-sample `lambda2` recovery vs T.
6. Zeta noise bridge or analytic/deterministic/neural failure attribution.

## Recommended Main Tables

1. Process-family diagnostics.
2. MRW/monofractal projection diagnostics.
3. CMIN-SR ablation summary.
4. Finite-sample identifiability.
5. Optional real-world sanity check.

## Recommended Limitations

- Current q-grid and scale range do not reliably recover `lambda2` at short T.
- High-q moments can be unstable.
- Dense q-grids alone may not fix finite-sample curvature recovery.
- More reliable empirical spectrum estimators, such as MFDFA or wavelet leaders,
  may be needed for stronger raw curvature recovery.

## Recommended CSF Framing

CMIN-SR is best framed as a calibrated scientific diagnostic framework:

- interpretable projections;
- calibrated spectral geometry;
- negative controls;
- finite-sample identifiability analysis;
- transparent limitations.
