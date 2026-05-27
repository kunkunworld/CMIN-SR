# Surrogate Validation

## Purpose

The surrogate validation experiment asks:

Does the inferred `lambda2` decrease when temporal dependence is destroyed,
while preserving much of the marginal distribution?

This is a more direct real-data test than volatility forecasting.

## Implemented Variants

Current minimal runnable version compares:

- `original`
- `shuffled`
- `block_shuffled`

Phase-randomized surrogates are not yet implemented in this phase.

## Estimation Mode

Current supported modes:

- `proxy`
- `auto`
- `model`

The stage-3 pipeline now supports both proxy and trained-`CMIN` evaluation.
The stage-4 pipeline additionally supports trained `CMIN-Robust`.

## Updated Spectral-Representation Reading

The surrogate tests are no longer interpreted only as:

- does `lambda2` drop after shuffling?

They now ask:

- does empirical spectrum geometry change after temporal dependence is destroyed?
- does `p_scaling` remain high or become unstable?
- does MRW projection residual increase?
- does `p_MRW` drop even if some empirical spectrum remains?

## Current Fama-French Interpretation

On Fama-French factor returns:

- original windows generally have positive `lambda2` gaps over shuffled windows,
- `p_MRW` is often also modestly higher in the original series,
- spectrum width is typically larger in original windows than in shuffled ones.

This is important because it shows:

- even in cleaned, aggregated, economically constructed factor returns,
- the estimator detects some temporal multiscale dependence beyond marginal distribution alone.

At the same time, these gaps are still modest, which supports the interpretation that Fama-French factors are conservative weak-intermittency controls rather than highly multifractal series.

After adding the first tiny trained `CMIN` checkpoint:

- original-vs-shuffled gaps remain positive for several factors,
- but they become much smaller than in proxy mode,
- so the current trained checkpoint is not yet stronger than the proxy baseline for surrogate sensitivity.

After adding `CMIN-Robust`:

- held-out mixed-process validation improves substantially,
- but Fama-French original-vs-shuffled gaps can become even smaller or negative,
- which should be interpreted as a robustness trade-off rather than a final failure.

After adding `CMIN-Robust-Multilength`:

- factor-return gaps become even more conservative on most factors,
- so cleaned factor returns continue to act as a stress test for over-suppression,
- and raw-market validation becomes even more important for deciding whether this conservatism is appropriate.

## Why This Matters

These results refine the earlier factor-return conclusion.

Earlier:
- `lambda2` looked compressed on factor returns.

Now:
- compression does not imply absence of dependence-based signal,
- because `original > shuffled` gaps are still positive for several factors.

So the correct interpretation is:

Fama-French factors may have weak but nonzero multiscale temporal structure, and the model is not simply hallucinating strong intermittency everywhere.

## Outputs

- `outputs/tables/real_surrogate_validation_proxy/real_surrogate_window_metrics.csv`
- `outputs/tables/real_surrogate_validation_proxy/real_surrogate_gap_summary.csv`
- `outputs/tables/real_surrogate_validation_cmin/real_surrogate_gap_summary.csv`
- `outputs/tables/real_surrogate_validation_cmin_robust/real_surrogate_gap_summary.csv`
- `outputs/tables/real_surrogate_validation_cmin_robust_multilength/real_surrogate_gap_summary.csv`
- `outputs/reports/real_surrogate_validation_proxy/real_surrogate_validation_summary.md`
- `outputs/reports/real_surrogate_validation_cmin/real_surrogate_validation_summary.md`
- `outputs/reports/real_surrogate_validation_cmin_robust/real_surrogate_validation_summary.md`
- `outputs/reports/real_surrogate_validation_cmin_robust_multilength/real_surrogate_validation_summary.md`

## What Still Needs To Be Done

- add phase-randomized surrogates,
- phase-randomized surrogate support is now implemented as an optional switch,
- rerun surrogate validation on raw SPY / QQQ / BTC / ETH returns,
- compare whether BTC and other raw-market returns show larger original-vs-shuffled gaps than factor-return data,
- retrain `CMIN` with explicit negative controls so model mode becomes more selective than proxy mode.
