# CMIN-SR Evaluation

## Evaluation Lengths

The current evaluation runs on:

- `T=256`
- `T=512`
- `T=1024`
- `T=2048`

where:

- `512` and `1024` are the current train lengths,
- `256` and `2048` are OOD stress lengths.

## Main Quantities

For each process family the evaluation reports:

- `H_proj`
- `lambda2_proj`
- `p_scaling`
- `p_MRW`
- `residual_norm`
- `spectrum_stability`
- MRW vs shuffled surrogate gaps

## Interpretation Policy

The most important outputs are now:

- `p_scaling`: is there a stable empirical scaling law?
- `p_MRW`: is that scaling law well explained by MRW?
- `residual_norm`: how much mismatch remains after MRW projection?

`lambda2_proj` should not be read in isolation.

## v2 Boundary Calibration Focus

For `CMIN-SR v2`, the most important extra diagnostics are:

- `p_mono`
- `mono_residual_norm`
- `mrw_vs_mono_gain`
- `curvature_significance`

The core evaluation question is no longer just:

- `is p_MRW high?`

but:

- `is p_MRW high because MRW genuinely beats a monofractal baseline?`

This is especially important for:

- `fGn`
- `iid Gaussian`
- low-`lambda2` MRW

## v3 Curved-Linear Focus

For `CMIN-SR v3`, evaluation adds:

- `p_curved`
- `p_mono`
- `boundary_mrw_score`
- `curvature_score`
- `curvature_significance`
- `linearity_score`
- `normalized_mrw_gain`
- `mono_residual_norm`

The key readout is:

- significant MRW should have high `p_scaling`, high `p_curved`, high `p_MRW`,
  and low `p_mono`;
- `fGn` / Gaussian should have high `p_scaling`, low `p_curved`, high `p_mono`,
  and lower `p_MRW`;
- low-`lambda2` MRW should be reported as a boundary case with high
  `boundary_mrw_score`;
- Student-t and regime-switching controls should keep low `p_MRW`;
- GARCH should remain cautious / ambiguous.

## Boundary-Calibrated Evaluation

The boundary-calibrated evaluation adds a same-H sweep:

```text
H in [0.2, 0.4, 0.6, 0.8]
lambda2 in [0.00, 0.01, 0.03, 0.06, 0.10, 0.15]
plus fGn(H)
```

The important diagnostics are:

- `p_curved` monotonicity over `lambda2_true`;
- `p_MRW` smooth increase over `lambda2_true`;
- `p_mono` decrease over `lambda2_true`;
- boundary score peaking near low lambda2;
- residual preference switching from monofractal to MRW as lambda2 increases.
