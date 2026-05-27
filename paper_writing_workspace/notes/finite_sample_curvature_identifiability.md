# Finite-Sample Curvature Identifiability Study

## Motivation

The previous CMIN-SR stages showed a consistent bottleneck:

- analytic spectral geometry calibration works in clean `zeta(q)` space;
- raw zeta alignment suppresses fake fGn/Gaussian curvature;
- curvature-preserving alignment still cannot reliably restore medium/high MRW curvature;
- `lambda2_true` vs `lambda2_proj` remains close to uncorrelated under short windows.

This study asks whether the current empirical spectrum estimator can recover
MRW curvature at all under the existing `q_grid`, scale range, and sample
lengths. If deterministic estimators fail too, the CMIN-SR raw failure should
be interpreted as a finite-sample zeta-estimation limitation, not as a missing
validity head.

## Entry Points

```bash
conda run -n for_codex python experiments/run_finite_sample_curvature_identifiability.py --quick
conda run -n for_codex python experiments/run_scale_length_sensitivity.py --quick
conda run -n for_codex python experiments/run_qgrid_sensitivity.py --quick
```

Outputs are written to:

- `outputs/reports/finite_sample_identifiability/`
- `outputs/tables/finite_sample_identifiability/`
- `outputs/figures/finite_sample_identifiability/`

## Deterministic Estimators

The estimator comparison lives in
`src/mrw_inverse/analysis/curvature_identifiability.py`.

Implemented estimators:

- `structure_ols`: ordinary least-squares log-structure regression;
- `structure_trimmed`: robust trimmed scale regression;
- `structure_bootstrap`: bootstrap over scales, with zeta uncertainty;
- `structure_smoothed`: light q-space smoothing after structure estimation.

Optional Huber, MFDFA, and wavelet-leader estimators are intentionally left as
future extensions rather than added dependencies.

## Current Quick-Run Result

In the quick diagnostic grid with `T in {512, 1024, 2048}` and
`lambda2 in {0, 0.03, 0.10, 0.20}`, recovery is poor for all implemented
structure-function estimators:

- `lambda2` MAE is about `0.08`;
- `lambda2` RMSE is about `0.11`;
- `lambda2_true` vs `lambda2_proj` correlation is near zero or negative;
- high-lambda detection rate is `0.0`;
- boundary accuracy is around `0.25`, close to a degenerate decision.

Increasing from `T=512` to `T=2048` in the quick run does not materially fix
the problem. Scale-set and q-grid sweeps also do not reveal a stable recovery
configuration under the current estimator family.

## Interpretation

Under the current structure-function estimator, q-grid, and scale range,
finite-sample MRW curvature is not reliably identifiable at the tested short
sample lengths. This supports the paper-level limitation:

```text
CMIN-SR is a validity-aware spectral diagnostic framework, not a guaranteed
lambda2 recovery engine under short finite samples.
```

The next improvement should target empirical spectrum estimation itself, such
as richer scale support, wavelet leaders, MFDFA, or stronger estimator-level
pretraining, rather than adding more validity heads.
