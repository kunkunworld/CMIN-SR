# CMIN-Robust Multi-length

## Purpose

This stage extends `CMIN-Robust` from single-length training to multi-length anti-confounded training.

Current train lengths:

- `T = 512`
- `T = 1024`

Main goal:

- reduce length OOD failures,
- preserve anti-confounding behavior,
- stabilize `MRW > shuffled MRW` separation across multiple finite-sample regimes.

## Current Result

Held-out mixed-process evaluation across lengths is encouraging:

- `T=512` and `T=1024` are both stable,
- `MRW > shuffled MRW` lambda2 and `p_MRW` gaps are positive,
- `Student-t`, `Gaussian`, and `RegimeSwitch` false positives remain controlled,
- `p_MRW` AUC stays high across `256, 512, 1024, 2048`.

However, external negative controls at `T=1024` still expose a serious trade-off:

- synthetic OOD negative-control runs can still push MRW / shuffled / fGn windows into abnormally large lambda2,
- so multi-length training improves in-distribution length generalization more than full distribution-shift robustness.

## Interpretation

This means:

1. length mismatch was part of the problem,
2. but not the whole problem,
3. anti-confounding training plus multi-length exposure improves internal robustness,
4. yet external process mismatch still matters.
