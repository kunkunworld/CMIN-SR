# Project Audit

## 1. Existing Model Code

Current primary model implementation file:

- `src/mrw_dl/models.py`

Key model classes identified:

- `PhysicsHybridCNNRegressor`
- `MRWStatisticalFrontend`
- `PCSMINRegressor`
- `PCSMINV2Regressor`
- `PCSMINV3Regressor`
- `LMMINetRegressor`
- `LMMICurvatureRegressor`
- supporting raw CNN / ResNet / scale-CNN baselines

Current final synthetic estimator is not a standalone trainable class. It is a structured expert combination:

- `lambda2` from `PC-SMIN`
- `H` from `LMMI-Net`
- analytic decoder downstream

This logic currently lives in:

- `scripts/evaluate_final_hybrid.py`

## 2. Existing Data / Preprocessing Code

Synthetic:

- `src/mrw_dl/generation.py`
- `src/mrw_dl/data.py`
- `scripts/generate_dataset_robust.py`
- `scripts/generate_dataset_wide.py`
- `scripts/make_split.py`

Real data:

- Fama-French preprocessing:
  - `scripts/preprocess_real_csvs.py`
- raw market preprocessing:
  - `scripts/preprocess_market_price_csvs.py`
- rolling inference:
  - `scripts/rolling_real_estimation.py`
- factor analysis:
  - `scripts/analyze_factor_real_world.py`
- raw market analysis:
  - `scripts/analyze_market_real_world.py`

## 3. Existing Outputs

Synthetic outputs:

- many `outputs/dl_spectrum_*` checkpoint folders
- `outputs/reports/*.md`, `*.csv`, `*.json`
- baseline and ablation comparison files

Real-data outputs:

- `outputs/real_world/rolling_mrw_estimates.csv`
- `outputs/reports/factor_real_world/*`
- preprocessing summaries under `outputs/reports/`

## 4. What Still Has Clear Value

Must keep:

- `src/mrw_dl/models.py`
- `src/mrw_dl/baselines.py`
- `src/mrw_dl/spectrum_baseline.py`
- all trained checkpoint folders under `outputs/dl_spectrum_*`
- final hybrid outputs
- factor real-world outputs
- preprocessing scripts

Should keep but treat as legacy experiment evidence:

- curvature variants
- identifiability control variants
- classical baseline scripts and demo runners

Potential cleanup / legacy candidates:

- early demo scripts
- duplicate plotting scripts
- `__pycache__/`

See:

- `docs/cleanup_candidates.md`

## 5. New Refactor Layer

A new additive framework has been introduced under:

- `src/mrw_inverse/`

This allows the project to evolve toward a mathematically explicit inverse-problem formulation without breaking the existing benchmark and reporting stack.
