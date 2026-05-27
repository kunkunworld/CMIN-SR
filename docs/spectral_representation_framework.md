# Spectral Representation Framework

CMIN-SR is a validity-aware spectral diagnostic framework. It separates
empirical spectrum estimation from spectral geometry interpretation.

## 1. Empirical Spectrum

The raw signal is mapped to an empirical scaling spectrum:

```text
zeta_emp(q)
```

This is the fragile finite-sample component of the framework.

## 2. Monofractal Projection

The empirical spectrum is projected onto a linear monofractal family:

```text
zeta_mono(q) = qH
```

This gives `H_mono` and `mono_residual_norm`.

## 3. MRW Projection

The empirical spectrum is also projected onto the MRW family:

```text
zeta_MRW(q) = qH - 0.5 * lambda2 * q * (q - 2)
```

This gives `H_proj`, `lambda2_proj`, and `residual_norm`.

Important: `lambda2_proj` is a projection coordinate, not proof of a true MRW
mechanism.

## 4. Projection Residuals

The framework compares:

- MRW residual;
- monofractal residual;
- MRW-vs-mono gain;
- normalized gain;
- boundary score.

Projection residuals are essential because high stable scaling alone does not
imply MRW curvature.

## 5. `p_scaling`

Probability-like diagnostic for whether the signal has a stable scaling law.

High `p_scaling` means scaling is present; it does not imply multifractality.

## 6. `p_curved`

Probability-like diagnostic for nonlinear curvature beyond a linear
monofractal spectrum.

## 7. `p_mono`

Probability-like diagnostic for whether the spectrum is better explained by a
linear monofractal projection.

## 8. `p_MRW`

Probability-like diagnostic for whether the spectrum is stable, curved, and
compatible with the MRW projection family.

`p_MRW` should be interpreted together with `p_scaling`, `p_curved`, `p_mono`,
projection residuals, and instability warnings.

## 9. Tail Instability

Heavy tails and high-q moment instability can create apparent curvature. Tail
instability diagnostics prevent such artifacts from being overinterpreted as
MRW compatibility.

## 10. Finite-Sample Identifiability Limitation

The final estimator-level study shows that deterministic structure-function
estimators do not reliably recover `lambda2` at short T under the current
q-grid and scale range.

Therefore:

```text
CMIN-SR is not a guaranteed short-window lambda2 recovery engine.
```

The strongest defensible interpretation is diagnostic:

- is scaling stable?
- is the spectrum curved?
- is a monofractal explanation sufficient?
- does MRW improve residual geometry?
- are tails or regime instability contaminating the spectrum?
- is the sample length sufficient for curvature identification?
