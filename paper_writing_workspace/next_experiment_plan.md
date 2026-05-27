# Next Experiment Plan for CSF Submission

## Supplemental pass update: 2026-05-11

The following items were addressed without training a new CMIN-SR model:

- MFDFA / structure-function baseline evidence was added from existing
  classical baseline scripts. Remaining gap: WTMM or wavelet-leader baseline.
- Historical direct neural and unconstrained parameter-regression baselines were
  summarized from `outputs/dl_spectrum_*`.
- Analytic \(\zeta(q)\), \(\alpha(q)\), and \(f(\alpha)\) spectrum-shape figure
  was generated.
- Quick q-grid, scale-length, zeta-noise bridge, and failure-attribution outputs
  were refreshed.
- Real-world data were assessed as useful for CSF persuasion but not strictly
  required for the core finite-sample identifiability claim.

Updated highest-priority remaining work before submission:

1. If time allows, add WTMM or wavelet-leader baseline; otherwise state the
   classical comparison is limited to structure functions and MFDFA.
2. Optionally expand the current quick 3-seed supplement into a larger
   full-grid run.
3. Optionally expand the real-world sanity check beyond Fama-French factors,
   but keep interpretation diagnostic rather than mechanistic.

## Priority 1: Submission-Critical

### Experiment-1: MFDFA baseline comparison
- Purpose: Compare CMIN-SR / deterministic structure-function diagnostics with a standard multifractal estimator.
- Claim supported: The paper's finite-sample limitation and diagnostic framing are not artifacts of one estimator family.
- Required data: Synthetic MRW, fGn, Gaussian, Student-t, GARCH/regime controls.
- Required baseline: MFDFA; optionally wavelet leaders if feasible.
- Required metric: lambda2/curvature proxy correlation, zeta MAE, fGn false curvature, MRW high-lambda detection.
- Expected figure/table: Table comparing structure-function, MFDFA, and CMIN-SR diagnostic outputs; figure of lambda2 recovery vs T.
- Suggested script: Extend `experiments/run_finite_sample_curvature_identifiability.py` or create `experiments/run_mfdfa_baseline_comparison.py`.
- Supplemental-pass status: Partially completed. MFDFA and structure-function
  baseline evidence was generated and summarized; WTMM/wavelet-leader remains
  missing.
- Estimated difficulty: Medium for WTMM/wavelet-leader.
- Risk if omitted: Medium after MFDFA evidence; still mention baseline scope clearly.

### Experiment-2: Multiple-seed stability for main results
- Purpose: Show that analytic separation and finite-sample failure are not one-seed artifacts.
- Claim supported: Robustness of the main diagnostic conclusions.
- Required data: Existing analytic spectrum dataset and finite-sample MRW grid.
- Required baseline: Existing calibrator and deterministic estimators.
- Required metric: Mean/std of \(p_{\mathrm{MRW}}\), \(p_{\mathrm{curved}}\), lambda2 correlation, high-lambda detection.
- Expected figure/table: Error-bar table for analytic calibration and identifiability.
- Suggested script: Add seed loop to `experiments/paper/run_exp1_spectral_geometry_calibration.py` and `run_exp4_finite_sample_identifiability.py`.
- Estimated difficulty: Low-Medium.
- Supplemental-pass status: Completed in lightweight form for seeds 2024,
  2025, and 2026. See `paper_assets/tables/seed_stability_*_summary.csv`.
- Risk if omitted: Low after supplement; larger full-grid aggregation remains
  optional polish.

### Experiment-3: Real-world sanity check with cautious interpretation
- Purpose: Demonstrate relevance to a real complex-system signal without claiming MRW validation.
- Claim supported: Framework can be applied as warning-aware diagnostics beyond synthetic data.
- Required data: One available financial, turbulence, physiological, or other complex-system dataset.
- Required baseline: Original vs shuffled/surrogate comparison; deterministic zeta diagnostics.
- Required metric: \(p_{\mathrm{scaling}}\), \(p_{\mathrm{curved}}\), \(p_{\mathrm{MRW}}\), residuals, instability, surrogate gaps.
- Expected figure/table: One cautious real-world diagnostic table and one spectrum/residual plot.
- Suggested script: `experiments/paper/run_exp5_real_world_sanity_check.py` or raw-market pipeline.
- Estimated difficulty: Medium, depending on data availability.
- Supplemental-pass status: Completed using local Fama-French factor returns
  with original-vs-shuffled proxy diagnostics. See
  `paper_assets/tables/table5_real_world_sanity_famafrench_factors_proxy.csv`.
- Risk if omitted: Low if this is included as a cautious appendix/sanity check.

## Priority 2: Strongly Recommended

### Experiment-4: Direct neural spectrum regression baseline
- Purpose: Test whether constrained spectral representation improves over free zeta regression.
- Claim supported: Physical/analytic constraints are useful for finite-sample spectral diagnostics.
- Required data: Synthetic MRW/fGn/Gaussian raw series and target zeta curves.
- Required baseline: Unconstrained neural model predicting \(\zeta(q)\).
- Required metric: zeta MAE, curvature error, fGn false curvature, MRW curvature retention.
- Expected figure/table: Free regression vs constrained projection diagnostics.
- Suggested script: New lightweight baseline or recover historical outputs if present.
- Supplemental-pass status: Completed from historical MLP/CNN outputs; use as
  appendix evidence.
- Estimated difficulty: Done for historical baseline; medium for a newly
  standardized final baseline.
- Risk if omitted: Low if appendix table is used.

### Experiment-5: Unconstrained \(H,\lambda^2\) regression baseline
- Purpose: Compare projection-constrained inference with direct parameter regression.
- Claim supported: The constrained diagnostic pipeline is not replaceable by a simple regressor.
- Required data: Synthetic MRW parameter-labeled samples.
- Required baseline: MLP/CNN parameter regressor.
- Required metric: H MAE, lambda2 MAE/correlation, boundary accuracy.
- Expected figure/table: Parameter recovery comparison table.
- Suggested script: `experiments/compare_estimators.py` or new paper wrapper.
- Supplemental-pass status: Completed from historical MLP/CNN and constrained
  decoder outputs; use cautiously as appendix evidence.
- Estimated difficulty: Done for historical baseline; medium for a newly
  standardized final baseline.
- Risk if omitted: Low-Medium.

### Experiment-6: \(f(\alpha)\) / alpha-spectrum visualization
- Purpose: Connect the paper more directly to multifractal spectrum tradition.
- Claim supported: Analytic decoder gives physically interpretable spectrum geometry.
- Required data: Analytic linear, boundary, and curved MRW spectra.
- Required baseline: Not required; analytic plotting enough.
- Required metric: Shape consistency, concavity, spectrum width.
- Expected figure/table: Example \(\zeta(q)\), \(\alpha(q)\), \(f(\alpha)\) curves.
- Suggested script: Extend `experiments/paper/generate_paper_figures.py`.
- Supplemental-pass status: Completed for analytic spectrum geometry:
  `paper_assets/figures/fig7_multifractal_spectrum_shapes.pdf`.
- Estimated difficulty: Done for analytic figure; empirical reconstruction
  accuracy remains optional.
- Risk if omitted: Low after figure inclusion.

### Experiment-7: Full q-grid and scale-range sensitivity
- Purpose: Strengthen central finite-sample limitation.
- Claim supported: Curvature recovery depends on q-grid and scale support.
- Required data: Existing synthetic MRW grid.
- Required baseline: Existing deterministic estimators.
- Required metric: lambda2 correlation, MAE, warning rate, high-lambda detection.
- Expected figure/table: q-grid bars and scale/T heatmaps.
- Suggested script: `experiments/run_qgrid_sensitivity.py`, `experiments/run_scale_length_sensitivity.py`.
- Estimated difficulty: Low-Medium.
- Risk if omitted: Sensitivity conclusion remains quick-run only.

## Priority 3: If Time Allows

### Experiment-8: Raw time-series noise robustness
- Purpose: Complement zeta-space noise bridge with raw signal perturbations.
- Claim supported: Noise affects empirical spectrum estimation upstream.
- Required data: Synthetic MRW/fGn/Gaussian with measurement noise/outliers.
- Required baseline: Existing zeta alignment / deterministic estimators.
- Required metric: p_curved, p_MRW, zeta MAE, high-q instability.
- Expected figure/table: Diagnostic degradation vs raw noise level.
- Suggested script: Extend raw zeta alignment eval.
- Estimated difficulty: Medium.
- Risk if omitted: Noise discussion remains partly indirect.

### Experiment-9: Bootstrap confidence intervals for diagnostics
- Purpose: Provide uncertainty estimates for diagnostic scores and residuals.
- Claim supported: Diagnostic framework is uncertainty-aware.
- Required data: Existing predictions and sample-level tables.
- Required baseline: None.
- Required metric: bootstrap CIs for p_MRW gaps, residual gaps, lambda2 correlations.
- Expected figure/table: Confidence interval table.
- Suggested script: Paper table generation script.
- Estimated difficulty: Low.
- Risk if omitted: Less polished but not fatal.

### Experiment-10: Expanded surrogate/shuffle controls
- Purpose: Test whether temporal dependence, tails, and shuffling alter spectral geometry as expected.
- Claim supported: Framework distinguishes MRW-like temporal structure from distributional artifacts.
- Required data: Original and shuffled MRW / financial-like samples.
- Required baseline: Existing surrogate pipeline if available.
- Required metric: p_MRW gap, p_curved gap, residual gap, tail instability.
- Expected figure/table: Original-vs-shuffled diagnostic table.
- Suggested script: Existing surrogate validation scripts.
- Estimated difficulty: Medium.
- Risk if omitted: Surrogate claims should remain limited.
