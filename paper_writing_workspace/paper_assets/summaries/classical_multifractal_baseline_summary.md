# Classical Multifractal Baseline Comparison

This experiment adds paper-ready classical multifractal baselines to the finite-sample curvature recovery study.

Implemented estimators:

- `structure_aggregated_ols`: overlapping aggregate-increment structure functions.
- `mfdfa`: first-order MFDFA under the project zeta convention.
- `mfdfa_quadratic`: second-order MFDFA.
- `wavelet_leader_haar`: compact Haar wavelet-leader approximation.
- `wtmm_haar`: compact Haar WTMM-style modulus-maxima approximation.

The two wavelet estimators are dependency-free Haar baselines. They are useful as classical controls, but they should not be described as full production WTMM/wavelet-leader packages.

Run mode: quick

## Summary by T and estimator

| T | estimator | lambda2_mae | lambda2_corr | high_lambda_detection_rate | boundary_accuracy |
| --- | --- | --- | --- | --- | --- |
| 512 | mfdfa | 0.070 | -0.080 | 0.750 | 0.600 |
| 512 | mfdfa_quadratic | 0.071 | -0.079 | 0.150 | 0.300 |
| 512 | structure_aggregated_ols | 0.076 | -0.107 | 0.050 | 0.275 |
| 512 | wavelet_leader_haar | 0.082 | -0.050 | 0.000 | 0.250 |
| 512 | wtmm_haar | 0.082 | -- | 0.000 | 0.250 |
| 1024 | mfdfa | 0.069 | -0.157 | 0.700 | 0.588 |
| 1024 | mfdfa_quadratic | 0.071 | -0.152 | 0.025 | 0.237 |
| 1024 | structure_aggregated_ols | 0.078 | 0.005 | 0.000 | 0.250 |
| 1024 | wavelet_leader_haar | 0.082 | -0.137 | 0.000 | 0.250 |
| 1024 | wtmm_haar | 0.082 | -- | 0.000 | 0.250 |
| 2048 | mfdfa | 0.067 | 0.086 | 0.700 | 0.600 |
| 2048 | mfdfa_quadratic | 0.071 | -0.058 | 0.025 | 0.263 |
| 2048 | structure_aggregated_ols | 0.080 | 0.141 | 0.000 | 0.250 |
| 2048 | wavelet_leader_haar | 0.082 | 0.144 | 0.000 | 0.250 |
| 2048 | wtmm_haar | 0.082 | -- | 0.000 | 0.250 |

## Interpretation

The baseline is intended to test whether stronger classical estimators remove the short-window lambda2 recovery bottleneck. If correlations remain weak or estimator-dependent, the paper should keep the conservative claim that finite-sample empirical spectrum estimation is limiting under the tested settings.