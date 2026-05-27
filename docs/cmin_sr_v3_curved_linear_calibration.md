# CMIN-SR v3 Curved-Linear Calibration

## Motivation

CMIN-SR v2 introduced a monofractal projection and moved `p_MRW` away from
plain stable-scaling confidence. This was a partial success: `fGn p_MRW`
dropped, Gaussian became less MRW-like, and heavy-tail / regime-switching
controls stayed low. The weakness was that significant MRW samples were also
pulled down. In effect, v2 lowered the global confidence level rather than
cleanly separating stable linear scaling from MRW-like curved scaling.

CMIN-SR v3 keeps the v2 backbone and adds explicit curved-vs-linear
calibration. The main new variable is:

```text
p_curved = probability that the empirical spectrum has stable nonlinear
           curvature beyond a monofractal linear spectrum.
```

`p_curved` is not `p_MRW`. It only answers whether curvature is present.
`p_MRW` then asks whether that stable curvature is compatible with the
two-parameter MRW projection.

## Probability Semantics

- `p_scaling`: is there a stable scaling law?
- `p_curved`: is the empirical spectrum significantly curved?
- `p_mono`: is a linear monofractal spectrum a better explanation?
- `p_MRW`: is there stable curved scaling compatible with MRW projection?

This separation is the main v3 contribution. `fGn` and iid Gaussian should keep
high `p_scaling`, low `p_curved`, high `p_mono`, and low-to-medium `p_MRW`.
Significant-`lambda2` MRW should keep high `p_scaling`, high `p_curved`, low
`p_mono`, and high `p_MRW`.

## Curvature Diagnostics

The deterministic diagnostics live in:

- `src/mrw_inverse/models/curvature_diagnostics.py`

They consume empirical, monofractal, and MRW projected spectra, plus residuals
and projected `lambda2`, and return normalized scores:

- `curvature_score`: mean absolute second derivative of `zeta_emp`, squashed to
  `[0, 1]`.
- `linearity_score`: monofractal fit quality, adjusted by the MRW-vs-linear
  curve gap.
- `mrw_vs_mono_gain`: positive residual improvement from monofractal to MRW.
- `normalized_mrw_gain`: signed residual improvement.
- `curvature_significance`: combined projected `lambda2`, positive gain, and
  empirical curvature.
- `curvature_confidence`: softer curvature confidence for explanation and
  feature input.
- `boundary_mrw_score`: high when `lambda2` is small and MRW/monofractal fits
  are close.

The implementation is batch-compatible and NaN-safe.

The trainable v3 wrapper also adds a lightweight `p_boundary_head`. It consumes
the deterministic diagnostics plus the fused multiscale representation, then
blends the learned boundary score with the deterministic boundary score. This
keeps the diagnostic interpretation while giving the model a way to avoid
marking every low-curvature monofractal spectrum as boundary MRW.

## Boundary MRW

Low-`lambda2` MRW is not treated as a failed positive. It is the scientific
boundary where MRW and monofractal explanations are almost indistinguishable at
finite sample sizes.

The expected behavior is:

- high `p_scaling`
- low-to-medium `p_curved`
- medium `p_mono`
- medium `p_MRW`
- high `boundary_mrw_score`

This makes low-`lambda2` MRW a boundary class, not a strong-MRW class.

## Training Entry Points

```bash
conda run -n for_codex python experiments/train_cmin_sr_v3.py
conda run -n for_codex python experiments/evaluate_cmin_sr_v3.py
conda run -n for_codex python experiments/compare_cmin_sr_versions.py
```

Smoke tests:

```bash
conda run -n for_codex python scripts/smoke_test_curvature_diagnostics.py
conda run -n for_codex python scripts/smoke_test_cmin_sr_v3_training.py
conda run -n for_codex python scripts/smoke_test_cmin_sr_v3_checkpoint.py
conda run -n for_codex python scripts/smoke_test_cmin_sr_v3_eval.py
```

## Success Criterion

v3 is successful if it lowers `fGn` / Gaussian MRW over-compatibility while
preserving or restoring significant-MRW `p_MRW`. The goal is not to drive
`fGn p_MRW` to zero. The goal is a cleaner semantic decomposition:

```text
stable scaling != curved scaling != MRW-compatible curved scaling
```

## Follow-up: Boundary Calibration Ablation

The v3 run showed that adding `p_curved` alone was not enough. MRW and `fGn`
remained too similar in `p_curved` / `p_MRW`, so the next ablation adds same-H
contrastive calibration:

- `fGn(H)` versus `MRW(H, lambda2)`
- curvature ranking over lambda2 levels
- MRW compatibility ranking
- monofractal ranking in the opposite direction

See:

- `docs/cmin_sr_boundary_calibration.md`
