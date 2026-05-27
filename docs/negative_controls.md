# Negative Controls

## Purpose

The negative-controls experiment asks:

Does predicted `lambda2` really come from multiscale temporal dependence,
or can it be spuriously activated by heavy tails, volatility clustering, or nonstationarity?

Under the new framing, this becomes:

- does the signal have a stable empirical spectrum?
- if yes, is that spectrum well explained by MRW?
- if not, is the mismatch captured by `residual_norm`, `p_scaling`, and `p_MRW`?

## Implemented Processes

Current minimal runnable version includes:

1. `MRW`
2. `Shuffled MRW`
3. `fGn`
4. `iid Gaussian`
5. `iid Student-t`
6. `GARCH(1,1)`
7. `Regime-switching Gaussian`

## Estimation Mode

Current supported modes:

- `proxy`
- `auto`
- `model`

Current stage-3 runs include both:

- statistical proxy mode
- trained `CMIN` model mode using `checkpoints/cmin/cmin_tiny_synthetic.pt`
- trained `CMIN-Robust` model mode using `checkpoints/cmin/cmin_robust_synthetic.pt`

Proxy components:

- `H_proxy` from empirical structure-function slope,
- `lambda2_proxy` from a combination of empirical zeta curvature and log-volatility covariance slope,
- `f_alpha_width_proxy` from the analytically decoded spectrum width,
- heuristic `p_MRW` from fit quality, covariance slope, and residual size.

This is a sanity-check baseline, not the final neural estimator.

## Current Results Interpretation

Current findings are mixed but useful:

- `MRW` has higher average `lambda2` than `Shuffled MRW`, which supports the temporal-dependence interpretation.
- `fGn` keeps relatively low `lambda2`, which is encouraging for separating roughness from intermittency.
- `GARCH` remains ambiguous, as expected.
- `iid Student-t` can still generate elevated `lambda2` in proxy mode, which shows the current fallback estimator is still partly sensitive to marginal heavy tails.

After adding a tiny trained `CMIN` checkpoint, the picture becomes clearer:

- the trained model is usable on synthetic MRW validation,
- but it does **not** yet improve selectivity under negative controls,
- in fact, the current tiny checkpoint tends to push many non-MRW processes toward a similar positive `lambda2` regime.

Therefore:

- the current negative controls already support the claim that temporal dependence matters,
- but they do **not yet fully prove** that `lambda2` is cleanly disentangled from heavy tails,
- and the first tiny `CMIN` checkpoint shows that training on MRW positives alone is not enough,
- so anti-confounding training data and stronger validity supervision are still needed.

After adding `CMIN-Robust`, the picture improves again:

- Student-t, Gaussian, and regime-switching false positives are substantially reduced on held-out mixed-process evaluation,
- `p_MRW` becomes much more meaningful,
- but some external negative-control runs still show OOD trade-offs, especially when sequence length differs from the training regime.

After adding `CMIN-Robust-Multilength`:

- internal by-length evaluation improves strongly at both `T=512` and `T=1024`,
- but external `T=1024` negative controls can still produce amplified MRW / shuffled / fGn lambda2,
- so multi-length training helps length interpolation more than full mechanism OOD robustness.

Under the spectral-representation framing, these results are reinterpreted as:

- false-positive reduction is still useful,
- but non-MRW processes are not assumed to have zero spectrum,
- instead, they may have meaningful `zeta_emp` plus higher projection residual or lower validity.

## Outputs

- `outputs/tables/negative_controls_proxy/negative_controls_samples.csv`
- `outputs/tables/negative_controls_proxy/negative_controls_summary.csv`
- `outputs/tables/negative_controls_cmin/negative_controls_summary.csv`
- `outputs/tables/negative_controls_cmin_robust/negative_controls_summary.csv`
- `outputs/tables/negative_controls_multilength/negative_controls_summary.csv`
- `outputs/reports/negative_controls_proxy/negative_controls_summary.md`
- `outputs/reports/negative_controls_cmin/negative_controls_summary.md`
- `outputs/reports/negative_controls_cmin_robust/negative_controls_summary.md`
- `outputs/reports/negative_controls_multilength/negative_controls_summary.md`

## What Should Not Be Overclaimed

- We should not yet claim that the project has fully solved the distinction between MRW intermittency and generic heavy-tail effects.
- In particular, the Student-t control shows that proxy mode is not sufficiently selective to serve as final evidence.
- The tiny trained CMIN checkpoint also does not yet solve this problem, so negative-control selectivity remains an open target.
