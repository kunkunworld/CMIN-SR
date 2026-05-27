# Phase 5 Summary

Phase 5 pursued two goals in parallel:

1. multi-length anti-confounded training
2. raw-market validation readiness

## Multi-length Result

`CMIN-Robust-Multilength` trains successfully and improves held-out by-length stability:

- train lengths `512, 1024` are both stable,
- OOD stress at `256, 2048` remains reasonable in held-out mixed-process evaluation,
- `p_MRW` stays informative,
- `Student-t` and `RegimeSwitch` false positives remain low internally.

But external negative controls still show an important weakness:

- MRW / shuffled / fGn can still be over-amplified at `T=1024`,
- so multi-length training does not fully solve distribution-shift failures.

## Raw-Market Result

The raw-market pipeline is implemented but no SPY / QQQ / BTC / ETH CSVs are currently present.

So Phase 5 establishes:

- runnable preprocessing,
- runnable surrogate validation entry,
- runnable dynamics summary entry,
- and warning-based behavior when data are missing.

## Main Interpretation

At this point the project has moved beyond simple predictive benchmarking.

The main claims are now:

- anti-confounded training materially improves synthetic false-positive selectivity,
- `p_MRW` is now meaningful,
- cleaned factor returns remain conservative controls,
- raw market data are now the critical next empirical test,
- and volatility forecasting is still secondary evidence.
