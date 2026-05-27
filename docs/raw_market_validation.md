# Raw Market Validation

## Purpose

Raw market validation is the decisive real-data test for whether MRW-like intermittency is stronger on actual market returns than on cleaned factor returns.

Primary target assets:

- `SPY`
- `QQQ`
- `BTC`
- `ETH`

## Current State

The raw-market pipeline is now implemented, including:

- market CSV preprocessing,
- raw-market surrogate validation,
- raw-market dynamics reporting,
- warning-only behavior when CSVs are missing.

Current workspace status:

- no SPY / QQQ / BTC / ETH price-history CSVs are present,
- so raw-market validation cannot yet be run empirically,
- but the scripts now fail gracefully and record the missing-data requirement explicitly.

## Expected CSV Format

The preprocessing supports automatic detection of:

- date columns such as `Date`, `date`, `datetime`, `timestamp`, `time`
- price columns such as `Adj Close`, `Close`, `close`, `price`

It writes:

- `data/market_processed/all_market_returns.csv`

with fields including:

- `asset`
- `date`
- `price`
- `log_return`
- `simple_return`
- `source_file`

## Why This Matters

Fama-French factor returns are conservative controls.

Raw market returns are where we expect:

- stronger intermittency,
- richer regime switching,
- clearer original-vs-shuffled gaps,
- especially for BTC / ETH relative to SPY / QQQ.

## Updated Questions

Raw-market validation should now ask:

- do BTC / ETH exhibit wider `zeta_emp` / `f_emp` geometry than SPY / QQQ?
- is `p_scaling` high or unstable?
- is MRW projection residual lower or higher than on cleaned factor returns?
- does `p_MRW` rise in stress windows?
- does original vs shuffled change empirical spectrum geometry more clearly than in Fama-French factors?
