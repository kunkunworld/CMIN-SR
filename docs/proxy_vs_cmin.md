# Proxy vs CMIN

## Purpose

This comparison asks:

Does a trained physics-constrained CMIN reduce false positives relative to the statistical proxy estimator?

The main stress cases are:

- `MRW` vs `Shuffled MRW`
- `iid Student-t`
- `iid Gaussian`
- `GARCH(1,1)`
- `Regime-switching Gaussian`
- real-data `original` vs `shuffled` surrogate gaps

## Current Result

Current answer for the stage-3 tiny checkpoint:

- `CMIN` is trainable and can recover MRW parameters reasonably on synthetic MRW validation.
- But the current tiny checkpoint does **not** outperform the proxy estimator on negative controls.

Observed failure modes:

- `MRW - shuffled MRW` lambda2 gap is not preserved cleanly.
- `Student-t` false-positive lambda2 is higher than in proxy mode.
- `Gaussian` null lambda2 is also higher than in proxy mode.
- `Regime-switching Gaussian` remains a false positive.
- Fama-French original-vs-shuffled lambda2 gaps become much smaller under the current tiny checkpoint.

## Interpretation

This is a useful negative result.

It suggests:

1. physics-constrained architecture alone is not enough,
2. training only on positive MRW samples leads to an "always explain as MRW" bias,
3. anti-confounding training data must be added explicitly,
4. `lambda2` needs stronger supervision tied to temporal dependence rather than only spectrum regression.

## Next Training Signals To Add

Future training should mix:

1. MRW positive samples
2. shuffled MRW negative samples
3. iid Student-t heavy-tail negatives
4. GARCH stress samples
5. regime-switching Gaussian stress samples

And should supervise at least:

- `p_MRW`
- residual mismatch
- original-vs-shuffled lambda2 ordering
- heavy-tail vs temporal-dependence discrimination
- stronger log-volatility covariance consistency

This is now partially implemented in stage 4 through `CMIN-Robust`.

## Outputs

- `outputs/reports/proxy_vs_cmin/proxy_vs_cmin_summary.md`
- `outputs/tables/proxy_vs_cmin/proxy_vs_cmin_negative_controls.csv`
- `outputs/tables/proxy_vs_cmin/proxy_vs_cmin_surrogate_gaps.csv`
