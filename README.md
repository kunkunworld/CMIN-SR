# CMIN-SR

CMIN-SR is a validity-aware spectral diagnostic framework for stochastic
signals. It asks whether a finite noisy signal supports a linear monofractal,
boundary-MRW, or curved MRW-like spectral interpretation.

## Overview

The final project should be read as a diagnostic pipeline, not as a guaranteed
short-window `lambda2` recovery engine.

```text
raw signal
  -> empirical spectrum zeta(q)
  -> monofractal and MRW projections
  -> residuals, curvature, and instability diagnostics
  -> calibrated interpretation probabilities
```

The main probabilities are:

- `p_scaling`: stable scaling law?
- `p_curved`: significant nonlinear curvature?
- `p_mono`: better explained by a monofractal linear spectrum?
- `p_MRW`: stable curved spectrum compatible with MRW projection?

`lambda2_proj` is a projection coordinate, not proof of a true MRW mechanism.

## What This Repository Contains

- Core CMIN-SR modules under `src/mrw_inverse/`
- Analytic spectral geometry calibration
- Raw zeta alignment and curvature-preserving alignment ablations
- Finite-sample MRW curvature identifiability study
- Historical CMIN / CMIN-SR versions and ablations
- Paper asset generation scripts under `experiments/paper/`
- Documentation under `docs/`

## Installation

Use the project Python environment with PyTorch, NumPy, pandas, and matplotlib.
In the development workspace, commands were run with:

```bash
conda run -n for_codex python <script.py>
```

## Quick Start

Run a lightweight reproducibility check:

```bash
python scripts/smoke_test_paper_pipeline.py
```

Generate paper tables and figures from existing outputs:

```bash
python experiments/paper/generate_all_paper_assets.py --quick
```

## Reproduce Main Experiments

```bash
python experiments/paper/run_exp1_spectral_geometry_calibration.py --quick
python experiments/paper/run_exp2_process_family_diagnostics.py --quick
python experiments/paper/run_exp3_boundary_projection.py --quick
python experiments/paper/run_exp4_finite_sample_identifiability.py --quick
python experiments/paper/run_exp5_real_world_sanity_check.py --quick
python experiments/paper/generate_all_paper_assets.py --quick
```

## Main Results

- Analytic spectral geometry is learnable: linear monofractal, boundary MRW,
  and curved MRW spectra separate cleanly in controlled `zeta(q)` space.
- Applying the calibrator to raw CMIN-SR spectra exposes the upstream
  bottleneck: empirical spectrum quality.
- Raw zeta alignment suppresses fake fGn/Gaussian curvature but can also
  flatten true MRW curvature.
- Curvature-preserving alignment confirms a trade-off between rejecting
  monofractal false positives and recovering MRW curvature.
- Deterministic finite-sample estimators also fail to reliably recover
  `lambda2` at short T under the current q-grid and scale range.

## Repository Structure

- `src/mrw_inverse/models/`: empirical spectra, projections, calibrators, CMIN-SR models
- `src/mrw_inverse/data/`: analytic and raw synthetic datasets
- `src/mrw_inverse/losses/`: spectral and zeta-alignment losses
- `src/mrw_inverse/analysis/`: finite-sample identifiability analysis
- `experiments/`: training, evaluation, comparison, and paper wrappers
- `experiments/paper/`: paper-level reproducibility and asset scripts
- `paper_assets/`: generated figures, tables, summaries, and LaTeX snippets
- `docs/`: framework notes, manifests, experiment maps, and final summaries
- `checkpoints/`: saved model checkpoints
- `outputs/`: generated reports, tables, and figures

## Documentation

Start here:

- `docs/project_implementation_guide_cn.md`
- `docs/theory_review_multiagent_cn.md`
- `docs/final_project_manifest.md`
- `docs/core_code_index.md`
- `docs/experiment_plan.md`
- `docs/spectral_representation_framework.md`
- `docs/finite_sample_curvature_identifiability.md`
- `docs/classical_multifractal_baselines.md`
- `docs/legacy_experiments_map.md`
- `docs/final_summary_for_paper.md`

## Legacy Experiments

Historical experiments are preserved for reproducibility and appendix analysis:

- proxy estimators
- Tiny CMIN
- CMIN-Robust
- CMIN-SR v1/v2/v3
- boundary calibration
- raw zeta alignment
- curvature-preserving zeta alignment

See `docs/legacy_experiments_map.md`.

## Citation

Citation information will be added after manuscript preparation.
