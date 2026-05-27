# Estimator-Level MRW Curvature Recovery

## Question

Can deterministic, oracle-style estimators recover the MRW intermittency
curvature parameter from raw finite samples?

The test is deliberately model-independent: estimate `zeta(q)` from the raw
series, project it onto monofractal and MRW spectra, then measure whether
`lambda2_proj` tracks `lambda2_true`.

## Module

`src/mrw_inverse/analysis/curvature_identifiability.py` provides:

- `estimate_curvature_identifiability`
- `estimate_many`
- `CurvatureEstimate`

Each estimate returns:

- `zeta_est` and optional `zeta_std`;
- `H_proj`, `lambda2_proj`, and `H_mono`;
- MRW and monofractal residuals;
- MRW-vs-mono gain;
- curvature, second-difference, and third-difference scores;
- scaling quality and high-q instability flags.

The module reuses the existing projection and diagnostic code rather than
introducing a new interpretation head.

## Findings

The current deterministic structure estimators do not recover `lambda2` well in
short samples. In the finite-sample quick run:

- OLS, trimmed, bootstrap, and smoothed estimators all have similar error;
- high-lambda MRW samples are not reliably detected;
- the projected `lambda2` values do not increase monotonically with the true
  `lambda2` grid;
- q-grid densification is not enough by itself in the tested setup.

This means the neural zeta alignment failure is not surprising: the raw data
does not present a clean enough empirical curvature signal to the downstream
spectral geometry calibrator.

## Paper Use

This result should be used as a diagnostic/limitation section:

- analytic geometry is identifiable;
- raw finite-sample empirical geometry is often not;
- model confidence should be interpreted jointly with residuals, instability
  warnings, q-grid choice, and sample length.
