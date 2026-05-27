# CMIN-SR Boundary Calibration Ablation

## Motivation

CMIN-SR v3 added `p_curved` and a boundary head, but the result was only a
partial success. At `T=1024`, MRW and `fGn` still had very similar scores:

- both had `p_scaling` near 1;
- both had `p_curved` around 0.51-0.53;
- both had `p_MRW` around 0.64-0.66;
- both had `p_mono` around 0.62-0.64.

This showed that adding a head alone was insufficient. The missing signal was
targeted same-H supervision:

```text
same-H stable monofractal scaling
vs
same-H curved MRW scaling
```

## Boundary Calibration Dataset

The targeted dataset lives in:

- `src/mrw_inverse/data/boundary_calibration_dataset.py`

Each group fixes `H` and generates:

- `fGn(H)`
- iid Gaussian baseline
- MRW with `lambda2 = 0.00`
- MRW with `lambda2 = 0.01`
- MRW with `lambda2 = 0.03`
- MRW with `lambda2 = 0.06`
- MRW with `lambda2 = 0.10`
- MRW with `lambda2 = 0.15`

Each sample carries:

- `group_id`
- `H_group`
- `lambda2_true`
- `rank_curvature_target`
- `target_p_scaling`
- `target_p_curved`
- `target_p_mrw`
- `target_p_mono`
- `target_boundary_mrw`
- `curvature_class`

The dataset is an ablation / calibration dataset. It does not replace the main
spectral representation dataset.

## Boundary Calibration Loss

The loss lives in:

- `src/mrw_inverse/losses/boundary_calibration_losses.py`

It combines:

- curvature ranking within same-H groups;
- MRW compatibility ranking;
- monofractal ranking in the opposite direction;
- residual margin for `fGn` and significant MRW;
- boundary smoothness;
- scaling-curvature decoupling;
- soft curvature regression against true `lambda2`.

The training entry point fine-tunes the existing v3 architecture:

```bash
conda run -n for_codex python experiments/train_cmin_sr_boundary_calibrated.py
```

The checkpoint is:

- `checkpoints/cmin/cmin_sr_calibrated_synthetic.pt`

## Current Result

The ablation is useful diagnostically, but it is not yet the paper-table final
model. Same-H calibration lowered some stress-process false positives, but it
did not make `p_curved` a clean monotone lambda2 probe, and it did not restore
significant-MRW `p_MRW`.

This suggests the next calibration step should use a stronger analytic target,
for example:

- pretrain `p_curved` directly on analytic `zeta(q)` curves;
- freeze the encoder and train only the interpretation heads;
- increase the boundary batch ratio further;
- add direct lambda2-order supervision on analytic spectra before raw series
  fine-tuning.

This follow-up is implemented as spectrum-space calibration. The result is
diagnostic: analytic geometry is learnable, but raw CMIN-SR `zeta_emp` remains
the bottleneck when the calibrator is applied back to raw signals.
