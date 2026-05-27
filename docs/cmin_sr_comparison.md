# CMIN-SR Comparison

## Compared Estimators

The comparison script contrasts:

1. deterministic empirical spectrum + MRW projection baseline
2. legacy statistical proxy estimator
3. `CMIN-Robust`
4. `CMIN-Robust-Multilength`
5. `CMIN-SR`

## Why This Comparison Is Different

Older comparisons focused mostly on:

- `lambda2` false positives
- MRW-vs-nonMRW validity

The new comparison focuses on:

- `p_scaling`
- `p_MRW`
- `residual_norm`
- MRW vs shuffled spectral gaps
- whether monofractal and stress processes are handled more semantically

## Current First-Version Reading

`CMIN-SR` should be judged as a first stable-spectrum learner, not as a final universal solution.

Its most important improvement over old robust CMIN is conceptual:

- it models empirical spectrum first,
- then treats MRW as a projection family,
- instead of forcing every process directly into an MRW-only estimator / classifier framing.

## v1 vs v2 Reading

The main purpose of `CMIN-SR v2` is not to improve every number at once.

Its specific target is:

- reduce false MRW confidence on stable monofractal spectra such as `fGn`

The correct comparison therefore emphasizes:

- `fGn p_MRW`
- Gaussian `p_MRW`
- MRW `p_MRW`
- low-`lambda2` MRW boundary behavior
- `mrw_vs_mono_gain`

The current v2 result should be read as:

- partial monofractal-boundary improvement,
- with a real trade-off in MRW confidence.

## v1 vs v2 vs v3 Reading

`CMIN-SR v3` should be judged by whether it resolves that trade-off.

The desired comparison is:

- `fGn p_scaling` remains high across versions;
- `fGn p_curved` is low in v3;
- `fGn p_MRW` drops relative to v1/v2 or is clearly separated from significant
  MRW;
- significant MRW `p_curved` is high;
- significant MRW `p_MRW` is restored or at least not worse than v2;
- low-`lambda2` MRW shows boundary behavior instead of being forced into a hard
  positive label.

The comparison entry point is:

```bash
conda run -n for_codex python experiments/compare_cmin_sr_versions.py
```

## Boundary-Calibrated Comparison

`CMIN-SR-Calibrated` is included as a non-v4 ablation. It should only be treated
as paper-main-table ready if it lowers `fGn p_MRW` while restoring significant
MRW `p_MRW`.

The current calibrated run does not yet meet that stronger standard: stress
controls remain reasonable, but `fGn` and significant MRW remain too close.
