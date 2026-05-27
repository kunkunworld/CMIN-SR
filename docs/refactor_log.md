# Refactor Log

## 2026-04-30

### Step 1: Project audit
- Scanned `src/`, `scripts/`, `outputs/`, and current real-data folders.
- Confirmed that the existing project backbone is centered on `src/mrw_dl/` and numerous experiment scripts.
- Confirmed that no raw SPY / QQQ / BTC / ETH market CSVs are currently present in the scanned data folders.

### Step 2: Inverse-problem framework layer
- Added a new parallel package `src/mrw_inverse/` rather than modifying or deleting `src/mrw_dl/`.
- Reason:
  - preserve backward compatibility with existing synthetic benchmarks and checkpoints,
  - avoid breaking old scripts,
  - create a clearer home for the new physics-constrained inverse-problem framing.

### Step 3: New core modules
- Added:
  - `src/mrw_inverse/models/mrw_decoder.py`
  - `src/mrw_inverse/models/multiscale_features.py`
  - `src/mrw_inverse/models/logvol_covariance.py`
  - `src/mrw_inverse/models/cmin.py`
- Purpose:
  - formalize the analytic MRW decoder,
  - isolate structure-function extraction,
  - isolate log-volatility covariance features for `lambda2`,
  - define a unified `CMINRegressor` as a constrained inverse-problem model.

### Step 4: New loss interfaces
- Added:
  - `src/mrw_inverse/losses/constraint_losses.py`
  - `src/mrw_inverse/losses/contrastive_losses.py`
  - `src/mrw_inverse/losses/mrw_losses.py`
- Purpose:
  - make the synthetic-to-real project mathematically structured around parameter, spectrum, scaling, log-vol covariance, constraint, and contrastive losses.

### Step 5: Experiment entry skeletons
- Added:
  - `experiments/run_synthetic_recovery.py`
  - `experiments/run_identifiability_test.py`
  - `experiments/run_negative_controls.py`
  - `experiments/run_real_surrogate_validation.py`
  - `experiments/run_vol_forecasting_secondary.py`
- Purpose:
  - provide stable entry points for the new experiment organization,
  - document what is already implemented versus what is still planned.

### Step 6: Real-data market phase preparation
- Added `scripts/preprocess_market_price_csvs.py`.
- Updated `scripts/rolling_real_estimation.py` so it can read processed market return files containing optional `price` columns and write outputs to a configurable directory.
- Added `scripts/analyze_market_real_world.py` to support raw-market reporting once SPY / QQQ / BTC / ETH CSVs are available.

### Step 7: Documentation scaffolding
- Added:
  - `docs/cleanup_candidates.md`
  - `docs/mathematical_formulation.md`
  - `docs/experiment_plan.md`
  - `docs/real_data_interpretation.md`
- Updated `README.md` to shift the project framing from benchmark chasing toward a physics-constrained neural inverse problem.

### Step 8: Safety policy
- No old files were deleted.
- No checkpoint directories were modified.
- Any potentially obsolete content is being recorded in `docs/cleanup_candidates.md` before any deletion is considered.

### Step 9: Smoke tests
- Ran `conda run -n for_codex python scripts/smoke_test_inverse_framework.py`
  - import succeeded,
  - `H_hat`, `lambda2_hat`, `zeta_hat`, `alpha_hat`, `f_alpha_hat`, and `p_MRW` all had valid shapes,
  - no NaN appeared in forward outputs or losses,
  - a smoke report was written to `outputs/reports/inverse_framework_smoke_test.json`.
- Ran `conda run -n for_codex python experiments/run_synthetic_recovery.py`
  - confirmed the new experiment-entry skeleton can generate a markdown report.
- Ran `conda run -n for_codex python scripts/analyze_market_real_world.py`
  - confirmed that missing raw market CSVs produce a structured warning instead of crashing.

### Step 10: README reframing
- Updated `README.md` so the project is explicitly described as a physics-constrained neural inverse problem rather than only a predictive deep-learning benchmark.

### Step 11: Proxy fallback estimator
- Added `src/mrw_inverse/proxy.py`.
- Purpose:
  - provide a no-checkpoint fallback estimator,
  - let non-MRW negative controls and surrogate validation run end-to-end immediately,
  - keep the framework operational before a trained `CMIN` checkpoint exists.
- Proxy outputs:
  - `H_proxy`
  - `lambda2_proxy`
  - `p_MRW` heuristic
  - residual norm
  - log-vol covariance slope
  - empirical zeta curvature
  - `f(alpha)` width

### Step 12: Negative controls experiment implementation
- Replaced the skeleton `experiments/run_negative_controls.py` with a minimal runnable experiment.
- Implemented generators for:
  - MRW
  - Shuffled MRW
  - fGn
  - iid Gaussian
  - iid Student-t
  - GARCH(1,1)
  - regime-switching Gaussian
- Added outputs under:
  - `outputs/tables/negative_controls/`
  - `outputs/figures/negative_controls/`
  - `outputs/reports/negative_controls/`

### Step 13: Real surrogate validation implementation
- Replaced the skeleton `experiments/run_real_surrogate_validation.py` with a minimal runnable experiment.
- Implemented variants:
  - original
  - shuffled
  - block-shuffled
- Added outputs under:
  - `outputs/tables/real_surrogate_validation/`
  - `outputs/figures/real_surrogate_validation/`
  - `outputs/reports/real_surrogate_validation/`
- Phase-randomized surrogates are still not implemented in this minimal version.

### Step 14: Smoke tests for phase 2
- Added:
  - `scripts/smoke_test_negative_controls.py`
  - `scripts/smoke_test_surrogate_validation.py`
- Verified:
  - CSV generation works,
  - markdown summary generation works,
  - fallback proxy mode works,
  - no NaN in key prediction columns during smoke runs.

### Step 15: New documentation for phase 2
- Added:
  - `docs/negative_controls.md`
  - `docs/surrogate_validation.md`
- Updated:
  - `docs/experiment_plan.md`
  - `docs/real_data_interpretation.md`

### Step 16: Minimal CMIN synthetic training loop
- Added:
  - `experiments/train_cmin_synthetic.py`
  - `experiments/evaluate_cmin_synthetic.py`
  - `scripts/smoke_test_cmin_training.py`
  - `scripts/smoke_test_cmin_checkpoint.py`
- Purpose:
  - train a first minimal `CMINRegressor` checkpoint on synthetic MRW data,
  - save a reusable checkpoint with config metadata,
  - evaluate synthetic recovery before reusing model mode in negative controls and surrogate validation.

### Step 17: Checkpoint and mode standardization
- Updated `src/mrw_inverse/proxy.py`.
- Added support for:
  - checkpoint dictionaries with `model_state_dict`, `config`, and `model_name`,
  - `proxy`, `auto`, and `model` estimation modes,
  - explicit `model_name` and `checkpoint_path` fields in output rows.
- Reason:
  - make proxy mode and trained-model mode directly comparable in later experiments.

### Step 18: Negative controls and surrogate validation in model mode
- Updated:
  - `experiments/run_negative_controls.py`
  - `experiments/run_real_surrogate_validation.py`
- Added:
  - model-mode checkpoint loading
  - proxy / auto / model CLI modes
  - separate output tags for proxy and `CMIN` runs
- Result:
  - both experiments now run either with the statistical proxy estimator or with the trained tiny `CMIN` checkpoint.

### Step 19: Proxy-vs-CMIN comparison layer
- Added:
  - `experiments/compare_proxy_vs_cmin.py`
  - `docs/cmin_training.md`
  - `docs/proxy_vs_cmin.md`
- Purpose:
  - compare whether trained `CMIN` improves over proxy mode on negative controls and factor-return surrogate gaps.

### Step 20: Stage-3 empirical outcome
- Ran:
  - `conda run -n for_codex python experiments/train_cmin_synthetic.py`
  - `conda run -n for_codex python experiments/evaluate_cmin_synthetic.py`
  - proxy and model versions of negative controls
  - proxy and model versions of real surrogate validation
  - `conda run -n for_codex python experiments/compare_proxy_vs_cmin.py`
- Main outcome:
  - tiny `CMIN` trains stably and gives usable synthetic `H` / `zeta` recovery,
  - but it does **not** reduce heavy-tail / nonstationarity false positives relative to the proxy estimator,
  - so the next major need is anti-confounding training data rather than more architecture complexity.

### Step 21: Anti-confounded mixed-process dataset
- Added:
  - `src/mrw_inverse/data/__init__.py`
  - `src/mrw_inverse/data/anti_confounded_dataset.py`
- Purpose:
  - build mixed-process training data with:
    - MRW
    - shuffled MRW
    - iid Gaussian
    - iid Student-t
    - GARCH(1,1)
    - regime-switching Gaussian
  - provide `target_p_mrw`, `target_lambda2`, and paired MRW/shuffled IDs for anti-confounding supervision.

### Step 22: Robust losses and training scripts
- Added:
  - `src/mrw_inverse/losses/robust_losses.py`
  - `experiments/train_cmin_robust.py`
  - `experiments/evaluate_cmin_robust.py`
  - `scripts/smoke_test_cmin_robust_training.py`
  - `scripts/smoke_test_cmin_robust_checkpoint.py`
- Updated:
  - `src/mrw_inverse/losses/__init__.py`
  - `src/mrw_inverse/__init__.py`
- Purpose:
  - train `CMIN-Robust` with parameter loss on MRW samples,
  - validity supervision on all samples,
  - negative lambda suppression on non-MRW samples,
  - paired MRW vs shuffled ordering loss.

### Step 23: Robust model-mode integration
- Updated:
  - `experiments/run_negative_controls.py`
  - `experiments/run_real_surrogate_validation.py`
  - `src/mrw_inverse/proxy.py`
- Changes:
  - added model-name override support,
  - made checkpoint loading tolerant with `strict=False`,
  - enabled direct use of `cmin_robust_synthetic.pt` in existing experiment entry points.

### Step 24: Estimator comparison layer
- Added:
  - `experiments/compare_estimators.py`
  - `docs/cmin_robust_training.md`
  - `docs/estimator_comparison.md`
- Purpose:
  - compare `proxy`, `cmin_tiny`, and `cmin_robust` on both negative controls and real-data surrogate gaps.

### Step 25: Stage-4 empirical outcome
- Ran:
  - `conda run -n for_codex python experiments/train_cmin_robust.py`
  - `conda run -n for_codex python experiments/evaluate_cmin_robust.py`
  - `conda run -n for_codex python experiments/run_negative_controls.py --mode model --checkpoint checkpoints/cmin/cmin_robust_synthetic.pt --model-name cmin_robust --output-tag negative_controls_cmin_robust`
  - `conda run -n for_codex python experiments/run_real_surrogate_validation.py --mode model --checkpoint checkpoints/cmin/cmin_robust_synthetic.pt --model-name cmin_robust --output-tag real_surrogate_validation_cmin_robust`
  - `conda run -n for_codex python experiments/compare_estimators.py`
- Main outcome:
  - held-out mixed-process evaluation improved strongly over tiny CMIN,
  - Student-t / Gaussian / regime-switching false positives were substantially reduced,
  - `p_MRW` became meaningful,
  - but real factor-return surrogate gaps often became smaller or negative,
  - so robust training appears genuinely useful but more conservative.

### Step 26: Multi-length robust training and phase-randomized surrogate support
- Added:
  - `src/mrw_inverse/surrogates.py`
  - `experiments/train_cmin_robust_multilength.py`
  - `experiments/evaluate_cmin_robust_multilength.py`
  - `scripts/smoke_test_cmin_robust_multilength_training.py`
  - `scripts/smoke_test_cmin_robust_multilength_checkpoint.py`
- Updated:
  - `experiments/run_real_surrogate_validation.py`
  - `scripts/preprocess_market_price_csvs.py`
- Purpose:
  - extend anti-confounded training across `T=512` and `T=1024`,
  - add optional phase-randomized surrogate support,
  - make raw-market preprocessing output more standardized.

### Step 27: Raw-market validation readiness
- Added:
  - `experiments/run_raw_market_surrogate_validation.py`
  - `experiments/analyze_raw_market_dynamics.py`
  - `scripts/smoke_test_raw_market_preprocessing.py`
  - `scripts/smoke_test_raw_market_surrogate_validation.py`
- Result:
  - when SPY / QQQ / BTC / ETH CSVs are absent, the pipeline now writes structured warnings instead of crashing,
  - raw-market validation remains blocked only by missing data, not by missing code paths.

### Step 28: Phase-5 empirical outcome
- Ran:
  - `conda run -n for_codex python experiments/train_cmin_robust_multilength.py`
  - `conda run -n for_codex python experiments/evaluate_cmin_robust_multilength.py`
  - `conda run -n for_codex python experiments/run_negative_controls.py --length 1024 --mode model --checkpoint checkpoints/cmin/cmin_robust_multilength.pt --model-name cmin_robust_multilength --output-tag negative_controls_multilength`
  - `conda run -n for_codex python experiments/run_real_surrogate_validation.py --mode model --checkpoint checkpoints/cmin/cmin_robust_multilength.pt --model-name cmin_robust_multilength --output-tag real_surrogate_validation_cmin_robust_multilength`
  - `conda run -n for_codex python experiments/compare_estimators.py`
- Main outcome:
  - held-out by-length evaluation is strong at `T=512` and `T=1024`,
  - OOD held-out `T=256` and `T=2048` are also reasonable,
- but external `T=1024` negative controls still expose amplified MRW / shuffled / fGn behavior,
- factor-return surrogate gaps become even more conservative,
- so multi-length training fixes internal length generalization more than full external OOD robustness.

### Step 29: Empirical spectrum / MRW projection split
- Added:
  - `src/mrw_inverse/models/empirical_spectrum.py`
  - `src/mrw_inverse/models/mrw_projection.py`
  - `src/mrw_inverse/models/spectral_representation_model.py`
  - `src/mrw_inverse/losses/spectral_representation_losses.py`
  - `src/mrw_inverse/data/process_generators.py`
  - `src/mrw_inverse/data/normalization.py`
  - `experiments/run_spectral_representation_diagnostics.py`
  - smoke tests for empirical spectrum / MRW projection / spectral representation
  - `docs/spectral_representation_framework.md`
  - `docs/generator_alignment_audit.md`
- Updated:
  - `anti_confounded_dataset.py`
  - `run_negative_controls.py`
  - `README.md`
  - interpretation documents
- Purpose:
  - separate process-agnostic empirical spectral representation from MRW-family parametric projection,
  - distinguish `p_scaling` from `p_MRW`,
  - reinterpret older robust CMIN outputs as validity-aware MRW projection components rather than a universal final model.

### Step 30: First trainable CMIN-SR
- Added:
  - `src/mrw_inverse/data/spectral_representation_dataset.py`
  - trainable `CMINSRModel` in `spectral_representation_model.py`
  - `experiments/train_cmin_sr.py`
  - `experiments/evaluate_cmin_sr.py`
  - `experiments/compare_sr_estimators.py`
  - smoke tests for `CMIN-SR` training / checkpoint / eval
  - `docs/cmin_sr_training.md`
  - `docs/cmin_sr_eval.md`
  - `docs/cmin_sr_comparison.md`
- Purpose:
  - move from deterministic empirical-spectrum diagnostics to a first trainable stable spectral representation learner,
  - keep MRW as a projection family instead of the only target process,
  - evaluate process semantics through `p_scaling`, `p_MRW`, and `residual_norm`.

### Step 31: CMIN-SR v2 monofractal-vs-MRW calibration
- Added:
  - `src/mrw_inverse/models/monofractal_projection.py`
  - `src/mrw_inverse/losses/monofractal_calibration_losses.py`
  - `experiments/train_cmin_sr_v2.py`
  - `experiments/evaluate_cmin_sr_v2.py`
  - `experiments/compare_cmin_sr_v1_v2.py`
  - `scripts/smoke_test_monofractal_projection.py`
  - `scripts/smoke_test_cmin_sr_v2_training.py`
  - `scripts/smoke_test_cmin_sr_v2_checkpoint.py`
  - `scripts/smoke_test_cmin_sr_v2_eval.py`
  - `docs/cmin_sr_v2_monofractal_calibration.md`
- Updated:
  - `src/mrw_inverse/models/spectral_representation_model.py`
  - `src/mrw_inverse/data/spectral_representation_dataset.py`
  - `src/mrw_inverse/losses/__init__.py`
  - `README.md`
  - `docs/experiment_plan.md`
  - `docs/spectral_representation_framework.md`
  - `docs/cmin_sr_training.md`
  - `docs/cmin_sr_eval.md`
  - `docs/cmin_sr_comparison.md`
- Purpose:
  - explicitly separate stable monofractal scaling from MRW-like curved multifractal scaling,
  - redefine `p_MRW` as MRW-vs-mono explanatory gain instead of generic stable-scaling confidence,
  - treat low-`lambda2` MRW as a soft boundary case instead of forcing it into strong MRW validity.
- Result:
  - `fGn p_MRW` drops relative to v1,
  - Gaussian `p_MRW` also drops,
  - `Student-t` / regime-switching stay low-`p_MRW`,
  - but MRW `p_MRW` is also pulled down, so v2 improves boundary calibration only partially.
- Engineering note:
  - smoke tests originally overwrote the main `cmin_sr_v2` checkpoint / eval outputs,
  - smoke scripts now restore backed-up artifacts after running, so future tiny tests do not clobber the main v2 results.

### Step 32: CMIN-SR v3 curved-vs-linear calibration
- Added:
  - `src/mrw_inverse/models/curvature_diagnostics.py`
  - `src/mrw_inverse/losses/curved_linear_calibration_losses.py`
  - `CMINSRv3Model` in `src/mrw_inverse/models/spectral_representation_model.py`
  - v3 dataset labels in `src/mrw_inverse/data/spectral_representation_dataset.py`
  - `experiments/train_cmin_sr_v3.py`
  - `experiments/evaluate_cmin_sr_v3.py`
  - `experiments/compare_cmin_sr_versions.py`
  - `scripts/smoke_test_curvature_diagnostics.py`
  - `scripts/smoke_test_cmin_sr_v3_training.py`
  - `scripts/smoke_test_cmin_sr_v3_checkpoint.py`
  - `scripts/smoke_test_cmin_sr_v3_eval.py`
  - `docs/cmin_sr_v3_curved_linear_calibration.md`
  - `docs/cmin_sr_version_comparison.md`
- Updated:
  - `README.md`
  - `docs/experiment_plan.md`
  - `docs/spectral_representation_framework.md`
  - `docs/cmin_sr_v2_monofractal_calibration.md`
  - `docs/cmin_sr_eval.md`
  - `docs/cmin_sr_comparison.md`
- Purpose:
  - separate stable scaling, explicit curvature, monofractal explanation, and MRW compatibility,
  - add `p_curved` and `boundary_mrw_score`,
  - add a lightweight `p_boundary_head` so boundary-MRW is not purely deterministic,
  - treat low-`lambda2` MRW as a boundary case rather than a strong positive,
  - avoid the v2 failure mode where monofractal competition lowers both `fGn p_MRW` and MRW `p_MRW`.
- Engineering note:
  - v3 smoke tests now preserve existing main artifacts by backing them up and restoring them after tiny runs.

### Step 33: CMIN-SR boundary calibration ablation
- Added:
  - `src/mrw_inverse/data/boundary_calibration_dataset.py`
  - `src/mrw_inverse/losses/boundary_calibration_losses.py`
  - `experiments/train_cmin_sr_boundary_calibrated.py`
  - `experiments/evaluate_cmin_sr_boundary_calibrated.py`
  - `scripts/smoke_test_boundary_calibration_dataset.py`
  - `scripts/smoke_test_boundary_calibration_losses.py`
  - `scripts/smoke_test_cmin_sr_boundary_calibrated_training.py`
  - `scripts/smoke_test_cmin_sr_boundary_calibrated_checkpoint.py`
  - `scripts/smoke_test_cmin_sr_boundary_calibrated_eval.py`
  - `docs/cmin_sr_boundary_calibration.md`
  - `docs/cmin_sr_final_version_comparison.md`
- Updated:
  - `experiments/compare_cmin_sr_versions.py`
  - `README.md`
  - boundary calibration exports in `src/mrw_inverse/data/__init__.py`
  - boundary calibration loss exports in `src/mrw_inverse/losses/__init__.py`
- Purpose:
  - keep the v3 architecture unchanged,
  - fine-tune from `cmin_sr_v3_synthetic.pt`,
  - add same-H fGn/MRW lambda2-sweep contrastive calibration,
  - directly test whether `p_curved` can become a sharp linear-vs-curved probe.
- Result:
  - engineering path runs end-to-end and the calibrated checkpoint is saved,
  - Gaussian, Student-t, and regime-switching remain controlled,
  - but fGn and significant MRW remain too close and the same-H `p_curved` sweep is not monotonic,
  - so this is a useful ablation, not yet the final paper main-table model.

### Step 34: Spectrum-space calibration
- Added:
  - `src/mrw_inverse/data/analytic_spectrum_dataset.py`
  - `src/mrw_inverse/models/spectral_geometry_calibrator.py`
  - `src/mrw_inverse/losses/spectrum_space_calibration_losses.py`
  - `experiments/train_spectral_geometry_calibrator.py`
  - `experiments/evaluate_spectral_geometry_calibrator.py`
  - `experiments/apply_spectral_calibrator_to_cmin_sr.py`
  - `scripts/smoke_test_analytic_spectrum_dataset.py`
  - `scripts/smoke_test_spectral_geometry_calibrator.py`
  - `scripts/smoke_test_spectral_geometry_training.py`
  - `scripts/smoke_test_spectral_geometry_checkpoint.py`
  - `scripts/smoke_test_apply_spectral_calibrator_to_cmin_sr.py`
  - `docs/spectrum_space_calibration.md`
  - `docs/spectral_geometry_calibrator.md`
  - `docs/cmin_sr_spectrum_calibrated_comparison.md`
- Updated:
  - `README.md`
  - data/model/loss package exports
- Result:
  - analytic spectrum-space calibrator trains successfully,
  - analytic lambda2 sweep is monotonic for `p_curved`, `p_MRW`, and `p_mono`,
  - applying it to raw CMIN-SR outputs does not yet separate fGn and MRW,
  - current bottleneck is raw `zeta_emp` quality rather than the spectrum-space interpretation head.

### Step 35: Raw zeta alignment
- Added:
  - `src/mrw_inverse/data/raw_zeta_alignment_dataset.py`
  - `src/mrw_inverse/models/robust_zeta_estimator.py`
  - `src/mrw_inverse/models/zeta_aligned_encoder.py`
  - `src/mrw_inverse/losses/zeta_alignment_losses.py`
  - `experiments/train_raw_zeta_alignment.py`
  - `experiments/evaluate_raw_zeta_alignment.py`
  - `experiments/compare_zeta_alignment_effect.py`
  - raw zeta alignment smoke tests
  - `docs/raw_zeta_alignment.md`
  - `docs/zeta_alignment_comparison.md`
- Purpose:
  - directly train `raw signal -> zeta_emp(q)`,
  - suppress fake monofractal curvature before applying the pretrained spectral geometry calibrator,
  - avoid adding new validity heads.
- Result:
  - training runs and checkpoint saves successfully,
  - fGn/Gaussian calibrated MRW confidence drops strongly after alignment,
  - MRW calibrated MRW confidence also drops, indicating over-linearization and under-recovered true MRW curvature,
  - next work should improve curvature-preserving zeta estimation rather than validity-head design.

### Step 36: Curvature-preserving zeta alignment
- Added:
  - `src/mrw_inverse/losses/curvature_preserving_zeta_losses.py`
  - `experiments/train_curvature_preserving_zeta_alignment.py`
  - `experiments/evaluate_curvature_preserving_zeta_alignment.py`
  - `experiments/compare_curvature_preserving_effect.py`
  - curvature-preserving smoke tests
  - `docs/curvature_preserving_zeta_alignment.md`
  - `docs/curvature_preserving_zeta_comparison.md`
- Purpose:
  - keep fGn/Gaussian fake curvature suppressed,
  - restore medium/high MRW curvature using band-specific D2 matching,
  - use third-difference smoothness so true quadratic curvature is not penalized,
  - avoid new validity heads or larger backbones.
- Result:
  - training/eval run successfully and checkpoint saves,
  - fGn/Gaussian remain controlled in the final conservative run,
  - MRW medium/high curvature is still under-recovered,
  - a more MRW-favoring run restores MRW but reintroduces fGn false positives,
  - this points to q-grid / scale-range / estimator identifiability limits rather than head design.

### Step 37: Finite-sample curvature identifiability
- Added:
  - `src/mrw_inverse/analysis/curvature_identifiability.py`
  - `experiments/run_finite_sample_curvature_identifiability.py`
  - `experiments/run_scale_length_sensitivity.py`
  - `experiments/run_qgrid_sensitivity.py`
  - `experiments/run_zeta_noise_bridge.py`
  - `experiments/run_cmin_sr_failure_attribution.py`
  - finite-sample identifiability smoke tests
  - `docs/finite_sample_curvature_identifiability.md`
  - `docs/estimator_curvature_recovery.md`
  - `docs/zeta_noise_bridge.md`
  - `docs/cmin_sr_failure_attribution.md`
- Purpose:
  - stop adding validity heads or larger backbones,
  - test whether deterministic structure-function estimators can recover MRW
    `lambda2` from finite raw samples,
  - quantify T, scale-range, and q-grid sensitivity,
  - bridge analytic zeta noise to raw-CMIN-SR calibration failure,
  - attribute failure across analytic, deterministic, and neural zeta levels.
- Result:
  - all smoke tests pass and quick diagnostic outputs are generated,
  - deterministic OLS/trimmed/bootstrap/smoothed estimators have near-zero or
    negative `lambda2_true` vs `lambda2_proj` correlation at short T,
  - high-lambda MRW detection is effectively zero in the quick run,
  - q-grid and scale-set sweeps do not fix recovery under the current estimator,
  - zeta noise bridge shows that q-space noise can push linear spectra toward
    MRW-compatible calibrated scores,
  - failure attribution indicates an estimator-level finite-sample
    identifiability bottleneck rather than merely a neural head failure.

### Step 38: Project closure and paper asset organization
- Added:
  - `docs/final_project_manifest.md`
  - `docs/core_code_index.md`
  - `docs/legacy_experiments_map.md`
  - `docs/final_summary_for_paper.md`
  - `experiments/paper/` wrapper scripts for the final paper experiments
  - `experiments/paper/collect_paper_assets.py`
  - `experiments/paper/generate_latex_tables.py`
  - `experiments/paper/generate_paper_figures.py`
  - `experiments/paper/generate_all_paper_assets.py`
  - `paper_assets/figures`, `paper_assets/tables`, `paper_assets/summaries`, `paper_assets/latex`
  - `scripts/smoke_test_paper_pipeline.py`
- Updated:
  - `README.md` into a concise paper-repository overview
  - `docs/experiment_plan.md` into the final paper experiment plan
  - `docs/spectral_representation_framework.md` into a compact final definition
- Purpose:
  - stop new model development,
  - organize the core code and historical ablations,
  - provide reproducible paper-level entry points,
  - collect/generate paper-ready tables and figures.

### Step 39: Supplemental TODO evidence pass for paper writing
- Added:
  - `experiments/paper/summarize_todo_evidence.py`
  - `paper_assets/tables/todo_supplemental_baseline_summary.csv`
  - `paper_assets/figures/fig7_multifractal_spectrum_shapes.png`
  - `paper_assets/figures/fig7_multifractal_spectrum_shapes.pdf`
  - `paper_assets/summaries/todo_evidence_completion_status.md`
  - `paper_assets/summaries/real_world_data_necessity_assessment.md`
- Updated:
  - `paper_writing_workspace/TODO_missing_evidence.md`
  - `paper_writing_workspace/claim_evidence_table.md`
  - `paper_writing_workspace/next_experiment_plan.md`
  - paper assets via `experiments/paper/generate_paper_figures.py`,
    `generate_latex_tables.py`, and `collect_paper_assets.py`
- Commands run:
  - `D:\anaconda\envs\for_codex\python.exe scripts\run_baselines_improved.py data\raw\mrw_dataset_robust_fgn.npz baseline_results_robust_improved_10`
  - `D:\anaconda\envs\for_codex\python.exe scripts\run_baselines_ensemble.py`
  - `D:\anaconda\envs\for_codex\python.exe experiments\run_finite_sample_curvature_identifiability.py --quick --num-samples 12 --seed 2026`
  - `D:\anaconda\envs\for_codex\python.exe experiments\run_qgrid_sensitivity.py --quick --num-samples 8 --seed 2026`
  - `D:\anaconda\envs\for_codex\python.exe experiments\run_scale_length_sensitivity.py --quick --num-samples 8 --seed 2026`
  - `D:\anaconda\envs\for_codex\python.exe experiments\run_zeta_noise_bridge.py --quick --num-samples 30 --seed 2026`
  - `D:\anaconda\envs\for_codex\python.exe experiments\run_cmin_sr_failure_attribution.py --quick --num-samples 16 --seed 2026`
  - `D:\anaconda\envs\for_codex\python.exe scripts\smoke_test_paper_pipeline.py`
- Result:
  - classical structure-function/MFDFA evidence is now summarized for appendix
    use,
  - historical unconstrained neural baselines are summarized,
  - analytic `zeta(q)`, `alpha(q)`, and `f(alpha)` spectrum-shape visualization
    is available,
  - finite-sample identifiability, q-grid sensitivity, scale-length
    sensitivity, zeta-noise bridge, and failure-attribution quick outputs were
    refreshed,
  - paper pipeline smoke test passed.
- Remaining caveats:
  - WTMM/wavelet-leader comparison is still absent,
  - full 3--5 seed aggregation is still recommended,
  - real-world data should be used only as optional sanity-check evidence unless
    a stronger application study is added.

### Step 40: Multi-seed stability and local real-data sanity supplement
- Added:
  - `experiments/paper/run_seed_stability_supplement.py`
  - `paper_assets/tables/seed_stability_spectral_geometry_by_seed.csv`
  - `paper_assets/tables/seed_stability_spectral_geometry_summary.csv`
  - `paper_assets/tables/seed_stability_identifiability_by_seed.csv`
  - `paper_assets/tables/seed_stability_identifiability_summary.csv`
  - `paper_assets/tables/seed_stability_zeta_noise_bridge_by_seed.csv`
  - `paper_assets/tables/seed_stability_zeta_noise_bridge_summary.csv`
  - `paper_assets/tables/seed_stability_failure_attribution_by_seed.csv`
  - `paper_assets/tables/seed_stability_failure_attribution_summary.csv`
  - `paper_assets/summaries/seed_stability_supplement.md`
  - `paper_assets/tables/table5_real_world_sanity_famafrench_factors_proxy.csv`
  - `paper_assets/figures/fig8_real_world_sanity_famafrench_factors_proxy.png`
  - `paper_assets/summaries/real_world_sanity_famafrench_factors_proxy.md`
- Updated:
  - `experiments/run_real_surrogate_validation.py`
    - fixed optional phase-surrogate aggregation when phase columns are absent,
    - made relative/absolute input-path display safe,
    - added `--series-names` so paper sanity checks can select only relevant
      factor-return series.
  - `paper_writing_workspace/TODO_missing_evidence.md`
  - `paper_writing_workspace/claim_evidence_table.md`
  - `paper_writing_workspace/next_experiment_plan.md`
  - `paper_assets/figures/figure_manifest.md`
- Commands run:
  - `D:\anaconda\envs\for_codex\python.exe experiments\paper\run_seed_stability_supplement.py --quick --seeds 2024,2025,2026 --num-samples-ident 8 --num-samples-spectral 600 --num-samples-noise 30 --num-samples-attribution 16`
  - `D:\anaconda\envs\for_codex\python.exe experiments\run_real_surrogate_validation.py --input data\real_processed\all_processed_returns.csv --window 512 --step 252 --mode proxy --output-tag real_world_sanity_famafrench_factors_proxy --series-names "F-F_Research_Data_Factors_daily:Mkt-RF,F-F_Research_Data_Factors_daily:SMB,F-F_Research_Data_Factors_daily:HML,F-F_Momentum_Factor_daily:Mom" --seed 2026`
- Result:
  - analytic spectral-geometry separation remains stable over three seeds,
  - finite-sample identifiability conclusions remain weak/limited over three
    seeds in quick mode,
  - zeta-noise bridge degradation is stable over three seeds,
  - local Fama-French factor returns show small positive original-vs-shuffled
    proxy gaps, suitable only as cautious application sanity evidence.
- Attempt/failure notes:
  - first real-data run failed because optional phase-surrogate columns were
    referenced when phase randomization was disabled; fixed by dynamic
    aggregation,
  - second real-data run failed because a relative input path was passed to
    `Path.relative_to(ROOT)`; fixed with safe display-path handling,
  - a preliminary run included `RF`; reran a cleaner factor-only sanity check
    over `Mkt-RF`, `SMB`, `HML`, and `Mom`.

### Step 41: CSF-style manuscript expansion and MRW simulation assets
- Added:
  - `experiments/paper/generate_mrw_simulation_assets.py`
  - `paper_assets/figures/fig9_mrw_simulation_example.png`
  - `paper_assets/figures/fig9_mrw_simulation_example.pdf`
  - `paper_assets/tables/mrw_simulation_example_timeseries.csv`
  - `paper_assets/tables/mrw_simulation_example_zeta.csv`
  - `paper_assets/summaries/mrw_simulation_figure_prompt.md`
  - `paper_writing_workspace/literature/citation_verification_update_2026-05-21.md`
  - `paper_writing_workspace/reviewer_audit_update_2026-05-21.md`
- Updated:
  - `paper_writing_workspace/main.tex`
  - `paper_writing_workspace/sections/01_introduction.tex`
  - `paper_writing_workspace/sections/02_framework.tex`
  - `paper_writing_workspace/sections/03_methods.tex`
  - `paper_writing_workspace/sections/04_experiments.tex`
  - `paper_writing_workspace/sections/05_results.tex`
  - `paper_writing_workspace/sections/06_failure_analysis.tex`
  - `paper_writing_workspace/sections/07_discussion.tex`
  - `paper_writing_workspace/sections/08_conclusion.tex`
- Purpose:
  - expand the manuscript into a more readable CSF-style draft,
  - define the finite-sample structure-function setting more explicitly,
  - explain how MRW samples are simulated,
  - update the text to include the newer MFDFA, three-seed, analytic
    `f(alpha)`, and Fama-French sanity-check evidence,
  - incorporate hallucination-audit and CSF-reviewer cautions.
- Verification:
  - checked that all `\cite{...}` keys used by the manuscript exist in
    `references.bib`,
  - checked that all referenced figure files exist in
    `paper_writing_workspace/figures`,
  - attempted local `pdflatex` build, but this shell could not find
    `pdflatex` on PATH; user-side VS Code/MiKTeX build should be used for final
    PDF verification.

### Step 42: Top-journal figure/table audit and reproducible figure-data export
- Added:
  - `experiments/paper/export_figure_data_and_revise_tables.py`
  - `paper_assets/figure_data/fig1_pipeline_structure.md`
  - `paper_assets/figure_data/fig1_pipeline_mermaid.md`
  - `paper_assets/figure_data/fig1_redesign_prompt.md`
  - `paper_assets/figure_data/fig2_mrw_simulation_timeseries.csv`
  - `paper_assets/figure_data/fig2_mrw_simulation_zeta.csv`
  - `paper_assets/figure_data/fig3_analytic_spectrum_geometry.csv`
  - `paper_assets/figure_data/fig4_spectral_geometry_map.csv`
  - `paper_assets/figure_data/fig5_lambda2_recovery.csv`
  - `paper_assets/figure_data/fig6_zeta_noise_bridge.csv`
  - `paper_assets/figure_data/fig7_failure_attribution.csv`
  - `paper_assets/figure_data/fig8_projection_residual_geometry.csv`
  - `paper_assets/figure_data/fig9_real_data_sanity_check.csv`
  - `paper_writing_workspace/paper_assets/summaries/top_journal_figure_table_audit.md`
  - `paper_writing_workspace/paper_assets/summaries/latex_float_and_layout_fix_report.md`
  - `paper_writing_workspace/paper_assets/summaries/top_journal_writing_audit.md`
- Updated:
  - `paper_writing_workspace/latex_tables/table1_process_family_diagnostics.tex`
  - `paper_writing_workspace/latex_tables/table2_mrw_mono_projection.tex`
  - `paper_writing_workspace/latex_tables/table3_finite_sample_lambda2_recovery.tex`
  - `paper_writing_workspace/latex_tables/table4_ablation.tex`
  - `paper_writing_workspace/sections/04_experiments.tex`
  - `paper_writing_workspace/sections/05_results.tex`
- Purpose:
  - audit every main figure/table from a top-journal perspective,
  - compact the finite-sample lambda2 recovery table for the main text while
    preserving full estimator-level CSV data,
  - export reproducible plotting data for all manuscript figures,
  - make conservative wording edits so CMIN-SR remains framed as a
    validity-aware diagnostic framework rather than an MRW mechanism prover.
- Verification:
  - generated all requested figure-data CSV/Markdown files,
  - synchronized key table/figure data into `paper_writing_workspace/paper_assets`,
  - checked that no experiment values were changed.

### Step 43: Chinese project implementation and concept guide
- Added:
  - `docs/project_implementation_guide_cn.md`
- Updated:
  - `README.md`
- Purpose:
  - provide a single Chinese project map explaining the research problem,
    concepts, final CMIN-SR pipeline, code organization, experiment evolution,
    reproduction entry points, paper storyline, and conservative interpretation.
- Notes:
  - no code, experiment values, checkpoints, or paper claims were changed.

### Step 44: Multi-agent theoretical review
- Added:
  - `docs/theory_review_multiagent_cn.md`
- Updated:
  - `README.md`
- Purpose:
  - synthesize three independent theory reviews covering mathematical
    consistency, MRW/multifractal modeling, and claim-evidence risk.
- Key findings:
  - the project theory is defensible as a spectral-geometry-constrained
    diagnostic framework,
  - the highest-priority mathematical risk is inconsistent structure-function
    semantics across text and estimator implementations,
  - `calibrated` scores, `lambda2_proj`, and finite-sample identifiability
    claims require conservative wording.
- Notes:
  - no code or manuscript text was changed beyond documentation/index updates.

### Step 45: Classical multifractal baseline supplement
- Added:
  - `src/mrw_inverse/analysis/classical_multifractal_estimators.py`
  - `experiments/run_classical_multifractal_baseline_comparison.py`
  - `scripts/smoke_test_classical_multifractal_baselines.py`
  - `docs/classical_multifractal_baselines.md`
  - `paper_writing_workspace/latex_tables/table5_classical_multifractal_baselines.tex`
- Updated:
  - `src/mrw_inverse/analysis/curvature_identifiability.py`
  - `src/mrw_inverse/analysis/__init__.py`
  - `experiments/run_finite_sample_curvature_identifiability.py`
  - `paper_writing_workspace/sections/04_experiments.tex`
  - `paper_writing_workspace/sections/05_results.tex`
  - `paper_writing_workspace/sections/07_discussion.tex`
  - `paper_writing_workspace/TODO_missing_evidence.md`
  - `docs/core_code_index.md`
  - `docs/finite_sample_curvature_identifiability.md`
  - `docs/project_implementation_guide_cn.md`
  - `README.md`
- Estimators added:
  - `structure_aggregated_ols`
  - `mfdfa`
  - `mfdfa_quadratic`
  - `wavelet_leader_haar`
  - `wtmm_haar`
- Outputs:
  - `outputs/tables/classical_multifractal_baselines/classical_baseline_lambda2_recovery_by_T.csv`
  - `outputs/tables/classical_multifractal_baselines/classical_baseline_by_estimator.csv`
  - `outputs/reports/classical_multifractal_baselines/classical_multifractal_baseline_summary.md`
  - `outputs/figures/classical_multifractal_baselines/classical_lambda2_corr_vs_T.png`
  - `outputs/figures/classical_multifractal_baselines/classical_high_lambda_detection_vs_T.png`
- Interpretation:
  - first-order MFDFA improves some quick-grid metrics and detects many
    high-lambda samples, but lambda2 correlation remains weak and changes sign
    across sample lengths,
  - compact Haar wavelet controls do not remove the short-window recovery
    bottleneck,
  - `wtmm_haar` degenerates to low-curvature estimates on the quick grid, so it
    must be described honestly as a lightweight WTMM-style control rather than
    a full WTMM package.
- Verification:
  - `D:\anaconda\envs\for_codex\python.exe scripts\smoke_test_classical_multifractal_baselines.py`
  - `D:\anaconda\envs\for_codex\python.exe experiments\run_classical_multifractal_baseline_comparison.py --quick --num-samples 10 --output-dir outputs --seed 2026`
  - `D:\anaconda\envs\for_codex\python.exe experiments\run_finite_sample_curvature_identifiability.py --quick --num-samples 2 --estimator-name all --output-dir outputs\finite_sample_all_estimators_smoke --seed 2026`

### Step 46: Fix clipped titles in manuscript figures
- Updated:
  - `experiments/paper/generate_mrw_simulation_assets.py`
  - `experiments/paper/summarize_todo_evidence.py`
  - `paper_writing_workspace/figures/fig9_mrw_simulation_example.png`
  - `paper_writing_workspace/figures/fig9_mrw_simulation_example.pdf`
  - `paper_writing_workspace/figures/fig7_multifractal_spectrum_shapes.png`
  - `paper_writing_workspace/figures/fig7_multifractal_spectrum_shapes.pdf`
- Purpose:
  - fix clipped internal titles in the MRW simulation figure and analytic
    spectrum geometry figure by lowering `suptitle` placement and reserving
    top layout space before saving.
- Verification:
  - visually inspected regenerated PNGs in the LaTeX figure directory.

### Step 47: Final LaTeX reference/title/layout cleanup
- Added:
  - `paper_writing_workspace/paper_assets/summaries/final_latex_publishability_check.md`
- Updated:
  - `paper_writing_workspace/sections/02_framework.tex`
  - `paper_writing_workspace/sections/03_methods.tex`
  - `paper_writing_workspace/sections/05_results.tex`
  - `paper_writing_workspace/main.pdf`
- Purpose:
  - remove unresolved citation/reference warnings,
  - avoid math-heavy subsection titles that triggered hyperref bookmark warnings,
  - eliminate overfull hbox warnings,
  - keep wording conservative and publication-ready.
- Verification:
  - ran `pdflatex -> bibtex -> pdflatex -> pdflatex` and a final `pdflatex`
    pass with MiKTeX,
  - final log scan found no undefined references, no undefined citations, no
    `??` warnings, no hyperref bookmark warnings, and no overfull hbox warnings.

### Step 48: Journal-style figure unification
- Added:
  - `experiments/paper/generate_journal_style_replacement_figures.py`
  - `paper_writing_workspace/figures/fig1_journal_cmin_sr_pipeline.png`
  - `paper_writing_workspace/figures/fig1_journal_cmin_sr_pipeline.pdf`
  - `paper_writing_workspace/figures/fig5_journal_finite_sample_identifiability.png`
  - `paper_writing_workspace/figures/fig5_journal_finite_sample_identifiability.pdf`
  - `paper_writing_workspace/paper_assets/summaries/figure_style_unification_review.md`
- Updated:
  - `paper_writing_workspace/sections/02_framework.tex`
  - `paper_writing_workspace/sections/05_results.tex`
- Purpose:
  - replace the black-background handmade framework and identifiability figures
    with white-background, publication-style versions that match the rest of
    the manuscript while preserving the scientific content.
- Verification:
  - regenerated the replacement figures with the `for_codex` Python
    environment,
  - compiled `main.tex` with MiKTeX `pdflatex`,
  - log scan found no undefined references, no undefined citations, no `??`
    warnings, and no overfull hbox warnings.

### Step 49: Citation strengthening and hallucination audit
- Added references:
  - Muzy, Bacry, and Arneodo (1991), wavelets and multifractal formalism,
  - Muzy, Bacry, and Arneodo (1993), structure functions vs WTMM,
  - Arneodo, Bacry, and Muzy (1995), wavelet thermodynamics of fractals,
  - Mallat and Hwang (1992), wavelet singularity detection,
  - Bacry, Gloter, Hoffmann, and Muzy (2010), mixed asymptotic multifractal
    analysis.
- Updated:
  - `paper_writing_workspace/references.bib`
  - `paper_writing_workspace/literature/references_curated.bib`
  - `paper_writing_workspace/literature/source_links.md`
  - `paper_writing_workspace/sections/01_introduction.tex`
  - `paper_writing_workspace/sections/04_experiments.tex`
  - `paper_writing_workspace/sections/07_discussion.tex`
- Added:
  - `paper_writing_workspace/literature/citation_hallucination_audit_2026-05-26.md`
- Purpose:
  - strengthen the WTMM/wavelet multifractal background and finite-sample
    identifiability discussion without adding unsupported claims.
- Verification:
  - ran `pdflatex -> bibtex -> pdflatex -> pdflatex`,
  - final log scan found no undefined references, no undefined citations, no
    `??` warnings, and no overfull hbox warnings.

### Step 50: Deep-learning inverse-problem citation support
- Added references:
  - Arridge, Maass, {\"O}ktem, and Sch{\"o}nlieb (2019), data-driven inverse
    problems,
  - Brunton, Noack, and Koumoutsakos (2020), scientific machine learning in
    physical modeling,
  - Cranmer, Brehmer, and Louppe (2020), simulation-based inference.
- Updated:
  - `paper_writing_workspace/references.bib`
  - `paper_writing_workspace/literature/references_curated.bib`
  - `paper_writing_workspace/literature/source_links.md`
  - `paper_writing_workspace/literature/citation_hallucination_audit_2026-05-26.md`
  - `paper_writing_workspace/sections/01_introduction.tex`
- Purpose:
  - support the manuscript's claim that the neural component is used as a
    constrained inverse-modeling and simulation-based diagnostic component,
    not as an unrestricted time-series predictor.
- Verification:
  - checked official DOI/publisher metadata rather than arXiv-only records,
  - ran `pdflatex -> bibtex -> pdflatex -> pdflatex`,
  - final log scan found no undefined references, no undefined citations, no
    `??` warnings, and no overfull hbox warnings.
