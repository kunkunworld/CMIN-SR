# Spectrum-Space Calibration

## Motivation

Raw same-H boundary calibration was insufficient. `fGn` and significant MRW
remained too similar in `p_curved` and `p_MRW`, suggesting that finite-sample
raw `zeta_emp` noise can obscure weak curvature.

This stage separates:

```text
raw spectrum estimation
from
spectral geometry interpretation
```

The spectral geometry calibrator is trained directly on analytic / semi-analytic
`zeta(q)` curves before being applied to CMIN-SR outputs.

## Main Result

The analytic spectrum-space task is learnable. The calibrator cleanly separates:

- linear monofractal spectra;
- boundary MRW spectra;
- clearly curved MRW spectra;
- heavy-tail-like unstable spectra;
- regime-like apparent spectra;
- ambiguous mild-curvature spectra.

The controlled analytic lambda2 sweep is monotonic:

- `p_curved_monotonic_fraction = 1.0`
- `p_mrw_monotonic_fraction = 1.0`
- `p_mono_decreasing_fraction = 1.0`

However, applying the calibrator back to raw CMIN-SR outputs does not yet solve
the fGn/MRW separation. This indicates that the raw `zeta_emp` and projection
features are the bottleneck, not the spectrum-space interpretation head.

Raw zeta alignment confirms the diagnosis: making `zeta_emp` more linear for
fGn/Gaussian reduces false MRW compatibility, but the first aligned model also
flattens MRW curvature too much.

Curvature-preserving zeta alignment tests whether band-specific MRW curvature
losses can fix that flattening. The current result is still partial: the
objective is better posed, but finite-sample MRW curvature recovery remains
weak under the current q-grid and scale range.

The finite-sample identifiability study closes the loop: when analytic spectra
are clean, the calibrator separates linear, boundary, and curved spectra; when
zeta noise is injected, the separation degrades; and when raw deterministic
estimators are used, `lambda2` recovery is poor. The raw bottleneck is therefore
empirical spectrum estimation quality.

## Interpretation Policy

`lambda2_proj` remains a projection coordinate, not proof of a true MRW
mechanism. Final interpretation should use:

- empirical spectrum;
- monofractal projection;
- MRW projection;
- calibrated `p_scaling`, `p_curved`, `p_mono`, `p_MRW`;
- projection residuals;
- instability diagnostics.
