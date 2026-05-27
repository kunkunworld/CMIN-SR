# CMIN-SR Failure Attribution

## Motivation

This diagnostic compares three levels for the same process families:

1. analytic target zeta plus spectral calibrator;
2. deterministic structure-function zeta plus spectral calibrator;
3. neural raw zeta plus spectral calibrator.

The goal is to locate whether the bottleneck is analytic interpretation,
deterministic empirical estimation, or the neural zeta encoder.

## Entry Point

```bash
conda run -n for_codex python experiments/run_cmin_sr_failure_attribution.py --quick
```

Outputs:

- `outputs/reports/cmin_sr_failure_attribution/`
- `outputs/tables/cmin_sr_failure_attribution/`
- `outputs/figures/cmin_sr_failure_attribution/`

## Current Quick-Run Result

Analytic spectra behave correctly:

- fGn and Gaussian have low `p_MRW`;
- low-lambda MRW has high `p_boundary`;
- high-lambda MRW has high `p_curved` and high `p_MRW`.

Deterministic raw zeta estimates are conservative and nearly linear:

- high-lambda MRW deterministic `p_MRW` is about `0.14`;
- low-lambda MRW deterministic `p_MRW` is about `0.15`;
- fGn deterministic `p_MRW` is about `0.13`.

Neural zeta estimates sit between analytic and deterministic behavior:

- fGn neural `p_MRW` is about `0.29`;
- high-lambda MRW neural `p_MRW` is about `0.24`;
- Gaussian neural `p_MRW` is about `0.17`.

The neural encoder is not cleanly recovering MRW curvature, but the
deterministic estimator is not recovering it either. This points to an
estimator-level finite-sample limitation rather than a pure neural architecture
failure.

## Interpretation

The final CMIN-SR story should be:

- spectrum-space interpretation is learnable;
- finite raw samples make empirical curvature estimation fragile;
- deterministic estimators and neural zeta alignment both struggle to recover
  `lambda2` at short T;
- CMIN-SR should therefore report calibrated spectral diagnostics and
  uncertainty/instability warnings, not overclaim exact MRW mechanism recovery.
