# CMIN-Robust Training

## Purpose

`CMIN-Robust` is the first anti-confounded training version of the inverse model.

It keeps the same physics-constrained decoder but changes the training signal:

- no longer only positive MRW regression,
- instead mixed-process supervision teaches the model what should **not** be treated as strong MRW intermittency.

## Mixed Processes

The anti-confounded dataset currently includes:

1. `MRW`
2. `Shuffled MRW`
3. `iid Gaussian`
4. `iid Student-t`
5. `GARCH(1,1)`
6. `Regime-switching Gaussian`

## Main Training Targets

For `MRW`:

- regress `H`
- regress `lambda2`
- high `p_MRW`

For non-MRW / weak-MRW processes:

- suppress `lambda2`
- lower `p_MRW`
- preserve `MRW > shuffled MRW` ordering when paired data are available

## Current Outcome

Current robust training runs stably and gives:

- synthetic MRW regression quality similar to tiny CMIN,
- much lower Student-t / Gaussian / regime-switching false positives on held-out mixed-process evaluation,
- meaningful `p_MRW` separation,
- restored positive `MRW > shuffled MRW` lambda2 gap on held-out mixed-process evaluation.

At the same time, trade-offs remain:

- factor-return surrogate gaps can become smaller or even negative,
- external negative controls at longer sequence length can still produce OOD behavior,
- so robustness is improved but not fully solved.

This is why the next stage adds multi-length robust training rather than a larger backbone.
