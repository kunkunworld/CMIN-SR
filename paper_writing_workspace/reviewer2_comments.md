# Reviewer 2 Comments

This file simulates a strict Chaos, Solitons & Fractals reviewer. The tone is
intentionally critical and concrete.

## Overall Assessment

The paper has a potentially interesting CSF-style contribution: it studies
finite-sample multifractal spectral diagnostics and shows that short-window MRW
curvature recovery is limited by empirical spectrum estimation. This is more
interesting than a routine neural-network benchmark. However, the paper must be
careful not to oversell itself as a successful physics-constrained inverse
solver for MRW parameters, because its own evidence shows weak lambda2 recovery
under finite samples.

## Major Concern 1: The paper must not read like a deep learning wrapper

The strongest contribution is the diagnostic decomposition:

- empirical zeta estimation;
- monofractal projection;
- MRW projection;
- spectral geometry calibration;
- finite-sample identifiability.

If the manuscript emphasizes network architecture, heads, training stages, or
loss engineering, it will look like an ordinary engineering paper weakly
wrapped in multifractal language. The CSF framing should emphasize nonlinear
scaling, finite-size effects, spectral curvature, and physical interpretability.

Action:

- Keep neural details concise.
- Move historical v1/v2/v3 head/loss development to appendix.
- Present the neural component as a finite-sample correction/calibration
  mechanism, not as the central claim.

## Major Concern 2: The inverse-problem claim is too strong

The repository contains older language about physics-constrained neural inverse
problems and recovering \(H,\lambda^2\). The final evidence does not support a
strong claim of reliable parameter inversion. Finite-sample lambda2 recovery is
near zero or negative in the deterministic study, and high-lambda detection is
0.0 in the quick diagnostic table.

Action:

- Replace "parameter recovery" with "projection coordinate" and "diagnostic
  spectral evidence" unless an experiment directly supports recovery.
- Explicitly state that \(\lambda^2_{\mathrm{proj}}\) is not proof of an MRW
  mechanism.

## Major Concern 3: Classical multifractal baselines are insufficient

The paper cites MFDFA, wavelet leaders, and bootstrap multifractal analysis, but
the final tables do not compare against MFDFA or WTMM/wavelet-leader baselines.
For CSF, this is a major weakness. Structure-function estimators are useful,
but they are not the full classical baseline set.

Action:

- Add at least one MFDFA baseline if possible.
- If not possible before submission, clearly state this limitation and avoid
  broad claims about outperforming classical multifractal estimators.

## Major Concern 4: Calibration language requires evidence

The paper uses "calibrated probabilities" and "calibrator." A reviewer may ask
whether these scores are probability-calibrated in the statistical sense. The
current evidence supports separation and diagnostic behavior, but not
reliability diagrams, Brier score, or expected calibration error.

Action:

- Use "calibrated diagnostic scores" rather than "probabilities" unless
  statistical calibration metrics are added.
- If possible, add reliability/calibration plots, but this is less important
  than classical multifractal baselines.

## Major Concern 5: Synthetic-only evidence weakens complex-system relevance

The controlled synthetic experiments are appropriate, but CSF readers may
expect at least one real complex-system sanity check. The current paper assets
do not contain a confirmed main real-world table. Existing raw-market pipelines
are warning-safe and optional.

Action:

- Either add a cautious real-world sanity check or explicitly frame the paper
  as a controlled finite-sample identifiability study.
- Do not claim empirical market MRW validation.

## Major Concern 6: Mathematical definitions need precision

The manuscript should define the empirical structure-function estimator,
q-grid, scale set, regression model, and high-q instability handling more
precisely. Current draft language such as "structure-function style
statistics" is too vague for a methods section.

Action:

- Add the formula for \(S_q(a)\) used by the implementation if confirmed.
- State the q-grid and scale sets used in experiments.
- Discuss moment existence and high-q instability for Student-t controls.

## Major Concern 7: \(f(\alpha)\) is underdeveloped

The paper title and framing are multifractal, but final assets do not show a
confirmed \(f(\alpha)\) reconstruction. If \(f(\alpha)\) appears prominently,
reviewers may expect plots or equations.

Action:

- Either add a simple analytic \(f(\alpha)\) figure/table, or keep
  \(f(\alpha)\) in background/future work rather than as a demonstrated result.

## Most Fatal Current Weakness

The most fatal weakness for CSF submission is not the negative lambda2 result.
The negative result is scientifically useful. The most fatal weakness is the
absence of a strong classical multifractal estimator comparison, especially
MFDFA or wavelet leaders, combined with limited real-world validation.

## Minimum Strengthening Before Submission

1. Add one classical baseline comparison, preferably MFDFA, on the same
   finite-sample MRW/fGn/Gaussian grid.
2. Add multiple-seed confidence intervals for the key analytic and finite-sample
   tables.
3. Add either a cautious real-world sanity check or a stronger justification
   that the paper is a controlled identifiability study.
