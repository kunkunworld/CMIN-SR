# Real Data Interpretation

## 1. Why Fama-French Factor Returns Matter

Fama-French factor returns are not raw market prices. They are:

- aggregated,
- cleaned,
- economically constructed return series.

This makes them conservative real-world test cases for intermittency.

If a model does **not** hallucinate strong `lambda2` on these series, that can be a scientifically meaningful result rather than a failure.

## 2. Current Factor-Return Interpretation

Observed pattern so far:

- `H` transfers more cleanly than `lambda2`.
- `lambda2` is compressed and often lies near the lower synthetic support boundary.
- forecasting gains over historical volatility are small.
- surrogate validation still shows positive original-vs-shuffled `lambda2` gaps on several factor series.

Interpretation:

- the estimator may be correctly recognizing that these factor-return series do not provide strong evidence for high MRW-style intermittency,
- while still extracting some roughness / scaling information through `H`.
- small but positive surrogate gaps suggest weak temporal multiscale dependence can still be detected even when absolute `lambda2` remains compressed.

So factor returns should be treated as:

`conservative / weak-intermittency real-world controls`

rather than as the decisive place where `lambda2` must succeed.

## 3. Why lambda2 Compression Can Be Meaningful

Compressed `lambda2` on factor returns can indicate at least three possibilities:

1. genuine weak multiscale intermittency in these cleaned series,
2. domain mismatch between synthetic MRW training data and real factor construction,
3. finite-sample bias under data that do not strongly support MRW curvature.

Importantly, this is different from a model falsely producing large `lambda2` everywhere.

## 4. Why Raw Market Returns Are the Next Decisive Test

Raw market returns, especially:

- `SPY`
- `QQQ`
- `BTC`
- `ETH`

are more likely to contain:

- heavier tails,
- stronger volatility clustering,
- clearer regime shifts,
- and richer multiscale temporal dependence.

Therefore they are a more appropriate test for whether `lambda2` is truly informative.

## 5. Surrogate Validation Logic

Real-data validation should not rely only on downstream forecasting.

For each rolling window, compare:

- original returns
- shuffled returns
- phase-randomized returns if implemented
- block-bootstrap surrogates if implemented

Desired pattern:

- `lambda2(original) > lambda2(shuffled)`
- `p_MRW(original) > p_MRW(shuffled)`
- stronger log-vol covariance slope in original
- wider spectrum in original

## 6. Updated Spectral-Representation Interpretation

Real data should now be interpreted through two layers:

1. empirical spectrum:
   - `zeta_emp`
   - `f_emp`
   - `p_scaling`
   - `spectrum_stability`

2. MRW projection:
   - `H_proj`
   - `lambda2_proj`
   - `p_MRW`
   - `residual_norm`

This means:

- low `lambda2_proj` does not imply "no spectrum"
- high spectrum width does not imply "clean MRW"
- a signal can have stable scaling but weak MRW projection

This is the key scientific test that intermittency is being inferred from multiscale dependence rather than from simple heavy tails alone.

## 6. Practical Reporting Guidance

When reporting real-data results:

- do not frame weak factor-return `lambda2` as a failure by default,
- emphasize that Fama-French series are conservative low-intermittency controls,
- reserve the stronger claim for raw SPY / QQQ / BTC / ETH data and surrogate comparisons.

## 7. Current Status

At present:

- factor-return preprocessing and rolling analysis are complete,
- factor-return surrogate validation is now minimally runnable in proxy mode,
- raw-market preprocessing and rolling-analysis scripts are prepared,
- but actual SPY / QQQ / BTC / ETH price CSVs are not yet present in the project folders.

So the next essential project step is data provision, not another synthetic architecture variant.

## 8. Stage 3 Update: Tiny CMIN vs Proxy

After training the first tiny synthetic `CMIN` checkpoint:

- model mode is now runnable on the same factor-return surrogate pipeline,
- but the current checkpoint does not improve on the proxy baseline,
- original-vs-shuffled `lambda2` gaps become smaller rather than larger.

This should be interpreted carefully:

- it does **not** invalidate the inverse-problem framing,
- it does show that positive-only MRW training is not enough,
- and it suggests that anti-confounding supervision must be added before model mode can be trusted more than the statistical proxy.

## 9. Stage 4 Update: CMIN-Robust

After adding anti-confounded mixed-process training:

- held-out mixed-process validation improves sharply,
- Student-t / Gaussian / regime-switching false positives are much lower than under tiny CMIN,
- `p_MRW` becomes meaningfully discriminative.

But real factor-return surrogate gaps can become smaller or negative.

So the current interpretation is:

- robust training improves rejection of synthetic confounders,
- but may become conservative on cleaned real factor returns,
- and raw-market SPY / QQQ / BTC / ETH data are still needed to decide whether this conservatism is appropriate or excessive.

## 10. Phase 5 Update: Multi-length and Raw Market

Multi-length robust training improves held-out length stability:

- both `T=512` and `T=1024` become well-behaved in internal evaluation,
- `p_MRW` remains discriminative,
- false positives stay controlled internally.

But factor-return surrogate gaps become even more conservative on most factors.

Therefore the current interpretation is:

- conservative factor behavior is not enough to judge the model negatively,
- but it is no longer enough to judge the model positively either,
- raw market returns are now required to decide whether the model is appropriately cautious or over-suppressing weak real intermittency.
