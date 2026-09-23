# Corrected Implementation and Reproduction

Run commands from the repository root after installing `requirements.txt`.
No manuscript, private correspondence, raw data or generated results are bundled.
All numerical modules below originate from the corrected experiments; public
packaging adds path options, dependency declarations and checkpoint-free tests.

## Code Map

| Module | Purpose |
| --- | --- |
| `generation.py` | Strict-PSD covariance sampling with sqrt(2T) correction and paired legacy/corrected innovations |
| `estimators/structure_input_contract.py` | Explicit increments/path adapter; unchanged historical regression definitions |
| `analysis_tools.py` | OLS, signed/clipped projection coordinates and frozen RMSE-feature interpretation |
| `retraining.py` | Existing neural architecture, fixed loss configurations, independent data streams |
| `linear_control.py` | Matched log-volatility features for the Ridge control |
| `backend_ablation.py` | Label-blind features, MLP/affine readouts, rules and independent holdouts |

The structure-bootstrap label in the input adapter is a historical alias of the
trimmed estimate with scale-subset uncertainty, not a separate independent
estimator. Projection comparison residuals use MAE; the analytic calibrator uses
its training-consistent RMSE residual features. Do not mix these conventions.

## 1. Numerical Gates

```bash
python scripts/smoke_test_public_release.py
python revision/scripts/validate_generator_correction.py --output-dir revision/results/generator_check
```

Checks include covariance identity and independent Cholesky sampling, legacy
pairing, explicit input equivalence, projection semantics, split independence,
finite gradients, frozen validity heads and matched readout forward paths.

## 2. Geometry Checkpoint Prerequisite

Corrected evaluations include a frozen historical geometry MLP as a reference.
Train the original analytic calibrator to create its expected checkpoint:

```bash
python experiments/train_spectral_geometry_calibrator.py --device cpu
```

This writes `checkpoints/cmin/spectral_geometry_calibrator.pt` and can overwrite
that destination. Do not run it over a checkpoint you intend to preserve. To use
a trusted existing file elsewhere, set `CMINSR_GEOMETRY_CHECKPOINT`:

```powershell
$env:CMINSR_GEOMETRY_CHECKPOINT = 'C:/path/to/spectral_geometry_calibrator.pt'
```

On POSIX shells use `export CMINSR_GEOMETRY_CHECKPOINT=/path/to/file.pt`.
Only load trusted PyTorch files. Regenerated weights need not have the archived
hash. The legacy analytic trainer includes synthetic tail-target features;
corrected raw evaluations set that channel to zero. The matched backend training
in step 5 also sets it to zero and is the fair MLP/affine comparison.

## 3. Corrected Deterministic Recovery

Start with the small check (requires step 2):

```bash
python revision/scripts/run_corrected_recovery.py --quick --output-dir revision/results/recovery_smoke
```

Full settings use five seeds, eight lambda2 levels and T=512..8192. Classical
MFDFA, quadratic MFDFA, Haar-leader and WTMM comparisons run at T<=2048.

```bash
python revision/scripts/run_corrected_recovery.py --output-dir revision/results/corrected_recovery_formal
python revision/scripts/run_corrected_recovery.py --seeds 32001 32002 32003 --num-samples 10 --output-dir revision/results/corrected_recovery_validation
python revision/scripts/run_scope_sensitivity.py --input-dir revision/results/corrected_recovery_formal --output-dir revision/results/scope_sensitivity
```

The sample CSV records generator, input convention, seed, target validity, signed
coordinates and projection metrics. Other-H modulated sequences are explicitly
sensitivity controls rather than confirmatory analytic targets.

## 4. Matched Neural Correction

Smoke training needs no geometry checkpoint. Smoke evaluation does require one:

```bash
python revision/scripts/train_corrected_neural_comparison.py --quick --device cpu --output-dir revision/results/neural_smoke
python revision/scripts/evaluate_corrected_neural_comparison.py --quick --device cpu --training-dir revision/results/neural_smoke --output-dir revision/results/neural_eval_smoke
```

Full fixed-budget runs (use `--device cuda` where available):

```bash
python revision/scripts/train_corrected_neural_comparison.py --device cpu --output-dir revision/results/neural_retrain_formal_attempt3
python revision/scripts/evaluate_corrected_neural_comparison.py --device cpu --training-dir revision/results/neural_retrain_formal_attempt3 --output-dir revision/results/neural_eval_formal
python revision/scripts/summarize_neural_retraining.py --evaluation-dir revision/results/neural_eval_formal --output-dir revision/results/neural_summary_formal
```

The retained `attempt3` directory name is an artifact-compatibility convention,
not three training attempts to perform. See `NEURAL_RETRAIN_PROTOCOL.md` and
`LINEAR_CONTROL_AMENDMENT.md` for settings, scope and independent controls.

## 5. MLP Versus Affine Readouts

The full backend evaluation uses frontend seed51001, full_c025, from step 4:

```bash
python revision/scripts/train_backend_ablation.py --output-dir revision/results/backend_ablation_training_formal
python revision/scripts/evaluate_backend_ablation.py --training-dir revision/results/backend_ablation_training_formal --frontend-dir revision/results/neural_retrain_formal_attempt3 --output-dir revision/results/backend_ablation_eval_formal
python revision/scripts/summarize_backend_ablation.py --training-dir revision/results/backend_ablation_training_formal --evaluation-dir revision/results/backend_ablation_eval_formal --output-dir revision/results/backend_ablation_summary_formal
python revision/scripts/plot_backend_ablation.py
```

A small end-to-end test can instead use the smoke frontend seed51999:

```bash
python revision/scripts/train_backend_ablation.py --quick --output-dir revision/results/backend_smoke
python revision/scripts/evaluate_backend_ablation.py --quick --training-dir revision/results/backend_smoke --frontend-dir revision/results/neural_smoke --frontend-seed 51999 --output-dir revision/results/backend_eval_smoke
python revision/scripts/summarize_backend_ablation.py --quick --training-dir revision/results/backend_smoke --evaluation-dir revision/results/backend_eval_smoke --output-dir revision/results/backend_summary_smoke
```

Do not mix smoke and full outputs in scientific tables. `BACKEND_ABLATION_PROTOCOL.md`
defines soft-target fitting, independent discrimination and stress acceptance as
different endpoints, with validation-only threshold selection.

## 6. Noise, Stress and Local Data

```bash
python revision/scripts/run_corrected_noise_bridge.py --evaluation-dir revision/results/neural_eval_formal --output-dir revision/results/corrected_noise_bridge
python revision/scripts/evaluate_stress_controls.py --output-dir revision/results/stress_controls
python revision/scripts/evaluate_retrained_stress.py --device cpu --training-dir revision/results/neural_retrain_formal_attempt3 --output-dir revision/results/retrained_stress
python revision/scripts/summarize_stress_decisions.py
```

The optional local factor-data check does not download or redistribute data:

```bash
python revision/scripts/run_real_surrogate_inference.py --quick --input-csv data/real_processed/all_processed_returns.csv --output-dir revision/results/real_surrogate_smoke
```

Input columns: `date`, `series_name`, `return`, `source_name`. Supply decimal
returns, unique dates within each of `Mkt-RF`, `SMB`, `HML`, `Mom`, and at least
512 observations per factor. Full runs omit `--quick` (999 surrogate replicates);
`--blocks 1 16` selects full shuffling and 16-day block shuffling. The declared
Holm family covers four factors, selected block sizes and two endpoints.
Changing that family after inspecting results is not part of this protocol.

## Output and Historical Dependencies

Generated `revision/results/`, `revision/e0_provenance/`, `outputs/`, data and
checkpoints are git-ignored. A new output directory is required per run.
Script defaults retain original seeds, budgets and numerical definitions.

`evaluate_frozen_encoders.py` additionally requires historical raw checkpoints
and is a distribution-shift comparison, not retraining on corrected data.
`summarize_corrected_experiments.py`, `summarize_scope_controls.py`,
`summarize_stress_decisions.py` and `plot_revision_evidence.py` are aggregate
helpers with fixed `revision/results/` paths. They require their corresponding
full runs and are not standalone fresh-clone or quick-test commands. Missing
artifacts must be generated or provided explicitly, not replaced with invented
results. Full reruns and exact archived numerical replay are distinct checks.
