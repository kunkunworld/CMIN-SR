# CMIN Tiny Synthetic Training

## Purpose

This stage introduces the first minimal trained checkpoint for the inverse-problem framework:

- model: `CMINRegressor`
- checkpoint: `checkpoints/cmin/cmin_tiny_synthetic.pt`

The goal is not SOTA synthetic performance.
The goal is to replace pure proxy mode with a trained physics-constrained baseline and test whether learning improves selectivity under negative controls.

## Training Setup

Default tiny configuration:

- `T = 512`
- `num_train = 2000`
- `num_val = 500`
- `batch_size = 32`
- `epochs = 10`
- `lr = 1e-3`
- `weight_decay = 1e-4`
- `seed = 2026`

Parameter ranges:

- `H in [0.1, 0.9]`
- `lambda2 in [0.0, 0.2]`

Loss weights:

- `param_weight = 1.0`
- `zeta_weight = 0.5`
- `falpha_weight = 0.2`
- `constraint_weight = 0.1`
- `logvol_weight = 0.0`
- `scaling_weight = 0.0`
- `contrast_weight = 0.0`

## Current Outcome

The tiny checkpoint trains stably and is useful as a first model-mode baseline.

Current synthetic evaluation is roughly:

- `MAE(H) ~ 0.027`
- `MAE(lambda2) ~ 0.050`
- `MAE(zeta) ~ 0.057`
- `MAE(f(alpha)) ~ 0.089`
- `lambda2 boundary hit rate = 0`

So:

- `H` recovery is already usable,
- `lambda2` recovery is only moderate,
- the checkpoint is sufficient for stage-3 stress testing,
- but it is not yet a strong intermittency-identifiability model.

## Outputs

- `outputs/reports/cmin_training/cmin_tiny_training_summary.md`
- `outputs/tables/cmin_training/train_history.csv`
- `outputs/figures/cmin_training/loss_curve.png`
- `outputs/tables/cmin_training/val_predictions.csv`
- `outputs/reports/cmin_eval/cmin_eval_summary.md`

## Important Limitation

This checkpoint is trained only on positive MRW samples.

That means it has not been taught to reject:

- heavy-tailed iid data
- GARCH volatility clustering
- regime-switching nonstationarity
- shuffled temporal-dependence-destroyed surrogates

Therefore poor selectivity under negative controls should not be surprising.
