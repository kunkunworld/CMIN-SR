# Mathematical Formulation

## 1. Core View: Inverse Problem, Not Free-Form Regression

The project is reframed as a physics-constrained inverse problem:

Given a finite, noisy, and possibly non-stationary observed path `x(t)`, infer latent MRW-like multiscale mechanism parameters

`theta = (H, lambda2)`

and determine whether the data supports an MRW interpretation at all.

This differs from free-form multifractal spectrum regression. The model should not directly output arbitrary spectrum curves unless a tightly constrained residual is explicitly enabled for mismatch analysis.

## 2. MRW Spectrum Parameterization

For MRW, the scaling exponent is parameterized by:

`zeta(q) = q H - (lambda2 / 2) q (q - 2)`

This implies:

- `H` mainly controls the linear slope of `zeta(q)`,
- `lambda2` controls the curvature of `zeta(q)`,
- `lambda2 = 0` corresponds to monofractal-like linear scaling.

## 3. Legendre-Spectrum Quantities

Define:

`alpha(q) = zeta'(q)`

For MRW specifically:

`alpha(q) = H - lambda2 (q - 1)`

and the Legendre spectrum becomes:

`f(alpha) = 1 + q alpha - zeta(q)`

Therefore `f(alpha)` is not an arbitrary object either. Once `(H, lambda2)` are known, `zeta(q)`, `alpha(q)`, and `f(alpha)` are analytically determined.

## 4. Why Free-Form Spectrum Regression Is Not Ideal

Directly regressing a spectrum curve is mathematically weak in this setting because:

1. The true MRW spectrum is low-dimensional and already parameterized.
2. Free-form regression can violate physical constraints such as concavity.
3. Legendre transforms amplify local noise in `zeta(q)`.
4. A model that freely outputs `f(alpha)` can appear accurate numerically while corresponding to no coherent MRW mechanism.

So the primary inference object should be latent parameters, not an unconstrained curve.

## 5. Finite-Sample Observation Model

The practical setting is:

- finite path length,
- additive noise or perturbations,
- non-ideal scaling ranges,
- possible real-data mismatch from exact MRW assumptions.

Thus the estimator should be:

`x(t) -> multiscale statistics -> finite-sample neural correction -> (H, lambda2) -> analytic MRW decoder`

The neural network is not replacing the mathematics. It is correcting finite-sample bias and noisy-statistic distortion around a constrained physical model.

## 6. Structure Functions

For increments over scale `a`, define:

`S_q(a) = E(|X(t+a) - X(t)|^q)`

Under ideal scaling:

`S_q(a) ~ a^{zeta(q)}`

Therefore:

`log S_q(a) = c_q + zeta(q) log a`

This is a primary multiscale observation equation and should appear both in features and in reconstruction loss.

## 7. Log-Volatility Covariance and lambda2

MRW intermittency is more directly linked to log-volatility dependence than to generic black-box temporal features. A key approximation is:

`Cov(log |Delta X_t|, log |Delta X_{t+tau}|) ≈ C - lambda2 log tau`

So `lambda2` should be supported by a dedicated log-volatility covariance branch. This gives the inverse problem a mathematically interpretable path from observed dependence to intermittency parameter.

## 8. Identifiability Discussion

The project already found a strong asymmetry:

- same-lambda2 / different-H discrimination is easy,
- same-H / different-lambda2 discrimination is hard.

Interpretation:

- `H` is largely identifiable from scale slope.
- `lambda2` is a second-order curvature/intermittency quantity and is more fragile under short windows and real-data mismatch.

This is why:

- explicit curvature features,
- log-volatility covariance features,
- identifiability-aware training,
- and non-MRW negative controls

are all central to the project.

## 9. MRW Validity and Residual Mismatch

Real data need not be exactly MRW.

So a useful framework may include:

- `p_MRW in [0,1]`, an MRW-validity score,
- an optional small residual:

`zeta_real(q) = zeta_MRW(q; H, lambda2) + r(q)`

with constraints such as:

- `r(0) ≈ 0`,
- `r(2) ≈ 0` when applicable,
- small residual norm,
- approximate concavity preserved.

This allows mismatch analysis without abandoning the constrained inverse-problem formulation.
