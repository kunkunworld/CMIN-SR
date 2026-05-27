# Raw Zeta Alignment

## Motivation

Spectrum-space calibration showed that analytic spectral geometry is learnable.
However, applying the calibrator to raw CMIN-SR outputs made `fGn` and Gaussian
look too MRW-compatible. The bottleneck is therefore upstream:

```text
raw signal -> zeta_emp(q)
```

This stage trains the raw encoder to align `zeta_emp(q)` with clean theoretical
targets for MRW, fGn, and Gaussian processes.

## Dataset

`src/mrw_inverse/data/raw_zeta_alignment_dataset.py` generates raw time series
with target spectra:

- MRW low lambda2: `lambda2 in [0, 0.03]`
- MRW medium/high lambda2: `lambda2 in [0.03, 0.20]`
- fGn: linear `zeta(q)=qH`
- iid Gaussian: linear `zeta(q)=0.5q`
- Student-t: masked / low-weight high-q target
- GARCH / regime stress: no strict zeta target, used for robustness

## Current Result

The model trains and reduces fake curvature for monofractal processes. At
`T=1024`, after applying the spectrum-space calibrator:

- `fGn p_MRW_cal` drops from about `0.73` to about `0.36`;
- Gaussian `p_MRW_cal` drops from about `0.60` to about `0.21`;
- Student-t stays low;
- regime-switching stays low;
- but MRW `p_MRW_cal` also drops from about `0.69` to about `0.35`.

So the stage is a partial success: it fixes false monofractal curvature, but
over-smooths / under-recovers MRW curvature.

The curvature-preserving follow-up adds band-specific MRW curvature losses and
third-difference smoothness. It confirms the trade-off: stronger MRW curvature
recovery can reintroduce fGn false positives, while conservative monofractal
preservation keeps fGn clean but leaves MRW under-recovered.

## Interpretation

The next bottleneck is not another validity head. It is better zeta estimation
that preserves true MRW curvature while suppressing finite-sample monofractal
fake curvature.

The finite-sample identifiability study sharpens this diagnosis: deterministic
structure-function estimators also fail to recover `lambda2` reliably at short
T under the current q-grid and scale range. This suggests the alignment problem
is estimator-level, not just neural.
