# TODO Missing Evidence

## Completion update: 2026-05-11

This TODO list was revisited as a supplemental evidence pass. No new CMIN-SR
version, validity head, loss family, or backbone was trained. The pass used the
existing `for_codex` Python environment and refreshed existing diagnostic
experiments.

Status summary:

- [x] Classical structure-function / MFDFA evidence was partially completed via
  `scripts/run_baselines_improved.py` and `scripts/run_baselines_ensemble.py`.
  Summary table:
  `paper_assets/tables/todo_supplemental_baseline_summary.csv`.
- [x] Direct neural and unconstrained parameter-regression baselines were
  recovered from historical outputs under `outputs/dl_spectrum_*` and summarized
  in `paper_assets/tables/todo_supplemental_baseline_summary.csv`.
- [x] Analytic \(f(\alpha)\) / spectrum-shape visualization was generated:
  `paper_assets/figures/fig7_multifractal_spectrum_shapes.pdf`.
- [x] Finite-sample identifiability quick tables were refreshed:
  `outputs/tables/finite_sample_identifiability/lambda2_recovery_by_T.csv`,
  `lambda2_recovery_by_qgrid.csv`, and
  `lambda2_recovery_by_scale_range.csv`.
- [x] Zeta-space noise bridge was refreshed:
  `outputs/tables/zeta_noise_bridge/separation_margin_vs_noise.csv`.
- [x] Failure attribution was refreshed:
  `outputs/tables/cmin_sr_failure_attribution/failure_attribution_summary_table.csv`.
- [x] Paper asset generation and smoke test passed:
  `scripts/smoke_test_paper_pipeline.py`.
- [x] Multiple-seed stability was completed for a lightweight paper supplement
  over seeds 2024, 2025, and 2026. See
  `paper_assets/tables/seed_stability_*_summary.csv` and
  `paper_assets/summaries/seed_stability_supplement.md`.
- [x] Real-world sanity check was completed on local Fama-French factor returns
  using original-vs-shuffled proxy diagnostics. See
  `paper_assets/tables/table5_real_world_sanity_famafrench_factors_proxy.csv`
  and `paper_assets/summaries/real_world_sanity_famafrench_factors_proxy.md`.
- [x] Lightweight wavelet controls were added through dependency-free Haar
  wavelet-leader and WTMM-style modulus-maxima baselines. These are useful
  classical controls, but they are not full production WTMM/wavelet-leader
  packages.

Detailed status file:
`paper_assets/summaries/todo_evidence_completion_status.md`.

## TODO-1: Classical MFDFA / WTMM baseline comparison
- Missing item: Full production WTMM/wavelet-leader package comparison remains optional; lightweight Haar wavelet controls are now available.
- Why it matters for CSF: CSF readers expect comparison with established multifractal estimation methods, not only neural/internal ablations.
- Current evidence: Structure-function deterministic estimators exist in `src/mrw_inverse/analysis/curvature_identifiability.py`; MFDFA evidence was added through `outputs/baselines/baseline_results_robust_improved_10.json` and `outputs/baselines/baseline_ensemble_result.json`. Final quick-grid classical baselines now include aggregated structure functions, MFDFA, quadratic MFDFA, Haar wavelet leader, and Haar WTMM-style maxima in `outputs/tables/classical_multifractal_baselines/classical_baseline_lambda2_recovery_by_T.csv`.
- Risk if not addressed: Lower than before; a reviewer may still ask for a production WTMM/wavelet-leader package, but the manuscript now has wavelet-style controls.
- Suggested experiment / check: If time allows, compare against a full external WTMM/wavelet-leader implementation. Otherwise describe the current Haar baselines honestly as lightweight controls.
- Suggested script or file to modify: `experiments/run_classical_multifractal_baseline_comparison.py`.
- Priority: Low / future benchmark

## TODO-2: Direct neural spectrum regression baseline
- Missing item: A clean final baseline where a neural model freely regresses \(\zeta(q)\) without MRW/monofractal projection constraints.
- Why it matters for CSF: Supports the claim that constrained spectral geometry is preferable to arbitrary spectrum regression.
- Current evidence: Historical MLP/CNN outputs exist and are now summarized in `paper_assets/tables/todo_supplemental_baseline_summary.csv`.
- Risk if not addressed: The comparison is historical rather than a newly standardized final split.
- Suggested experiment / check: Train or evaluate a direct zeta regression baseline on the same synthetic MRW/fGn grid.
- Suggested script or file to modify: Add a lightweight baseline under `experiments/` or summarize existing historical outputs if available.
- Priority: Low/appendix

## TODO-3: Unconstrained \(H,\lambda^2\) regression baseline
- Missing item: Direct unconstrained parameter regression baseline.
- Why it matters for CSF: Tests whether the physical projection/decoder adds stability beyond a simple supervised regressor.
- Current evidence: Older MLP/CNN and constrained PC-SMIN/final-hybrid outputs are now summarized in `paper_assets/tables/todo_supplemental_baseline_summary.csv`.
- Risk if not addressed: Use as appendix evidence only; not a central final-architecture claim.
- Suggested experiment / check: Compare projection-constrained inference with an unconstrained MLP/CNN parameter regressor.
- Suggested script or file to modify: `experiments/compare_estimators.py` or a new final baseline wrapper.
- Priority: Low/appendix

## TODO-4: \(f(\alpha)\) reconstruction visualization
- Missing item: Final paper figure showing reconstructed \(f(\alpha)\) or alpha-spectrum shape consistency.
- Why it matters for CSF: Multifractal papers often expect singularity spectra, not only \(\zeta(q)\) and calibrated probabilities.
- Current evidence: Analytic spectrum shape figure generated at `paper_assets/figures/fig7_multifractal_spectrum_shapes.pdf`.
- Risk if not addressed: Empirical \(f(\alpha)\) reconstruction accuracy is still not a main result.
- Suggested experiment / check: Generate example \(\zeta(q)\), \(\alpha(q)\), and \(f(\alpha)\) curves for linear, boundary, and curved MRW cases.
- Suggested script or file to modify: Add plotting to `experiments/paper/generate_paper_figures.py`.
- Priority: Completed for analytic visualization; empirical accuracy remains optional

## TODO-5: Multiple random seed stability
- Missing item: Main paper tables with multiple seeds and uncertainty intervals.
- Why it matters for CSF: Synthetic finite-sample studies should report robustness, not only one seed.
- Current evidence: Multi-seed supplement completed for seeds 2024, 2025, and 2026. Output tables are stored under `paper_assets/tables/seed_stability_*`.
- Risk if not addressed: Remaining risk is only that the aggregation uses quick settings; still adequate for a stability appendix.
- Suggested experiment / check: Run key experiments over 3-5 seeds and report mean/std.
- Suggested script or file to modify: Paper wrapper scripts in `experiments/paper/`.
- Priority: High

## TODO-6: Real-world complex system validation
- Missing item: Main final validation on a real financial, turbulence, physiological, or other complex-system dataset.
- Why it matters for CSF: CSF favors complex-system relevance beyond synthetic diagnostics.
- Current evidence: Local Fama-French factor-return sanity check completed in proxy mode. Main outputs: `outputs/tables/real_world_sanity_famafrench_factors_proxy/real_surrogate_gap_summary.csv`, `paper_assets/tables/table5_real_world_sanity_famafrench_factors_proxy.csv`, and `paper_assets/figures/fig8_real_world_sanity_famafrench_factors_proxy.png`.
- Risk if not addressed: Use as cautious application evidence only; do not claim a real MRW mechanism.
- Suggested experiment / check: Run one warning-safe real-world sanity check and present it as exploratory, not proof of MRW mechanism.
- Suggested script or file to modify: `experiments/paper/run_exp5_real_world_sanity_check.py`.
- Priority: High

## TODO-7: More complete scale-length sensitivity
- Missing item: Full-scale sensitivity results beyond quick grids.
- Why it matters for CSF: The central limitation concerns finite-sample and scale-range identifiability.
- Current evidence: `lambda2_recovery_by_scale_range.csv` was refreshed in quick mode with `for_codex`; quick/moderate run remains small.
- Risk if not addressed: Sensitivity conclusions may be underpowered.
- Suggested experiment / check: Run full or larger grid over \(T=512,1024,2048,4096,8192\) and scale sets A-D.
- Suggested script or file to modify: `experiments/run_scale_length_sensitivity.py`.
- Priority: Medium

## TODO-8: More complete q-grid sensitivity
- Missing item: Full q-grid sensitivity with dense q-grid and low/high q ranges.
- Why it matters for CSF: Curvature estimation depends strongly on q-range.
- Current evidence: `lambda2_recovery_by_qgrid.csv` was refreshed in quick mode with `for_codex`.
- Risk if not addressed: Claims about q-grid limitations must remain cautious.
- Suggested experiment / check: Run all Q1-Q5 grids with more samples and seeds.
- Suggested script or file to modify: `experiments/run_qgrid_sensitivity.py`.
- Priority: Medium

## TODO-9: Noise robustness study on raw time series
- Missing item: Systematic raw time-series noise perturbation study.
- Why it matters for CSF: The paper emphasizes noise and finite-sample instability.
- Current evidence: Zeta-space noise bridge exists; raw signal noise robustness is less explicit in final assets.
- Risk if not addressed: Noise claims may look indirect.
- Suggested experiment / check: Add measurement noise / outlier / colored noise tests and evaluate spectral diagnostics.
- Suggested script or file to modify: raw zeta alignment or process-family diagnostics scripts.
- Priority: Medium

## TODO-10: Statistical significance / confidence intervals
- Missing item: Confidence intervals or significance tests for key diagnostic gaps.
- Why it matters for CSF: Strengthens claims about separability and finite-sample failure.
- Current evidence: Supplemental baseline summary includes mean/std where sample-level baseline rows exist; main diagnostic CIs remain limited.
- Risk if not addressed: Reviewer may ask whether gaps are statistically reliable.
- Suggested experiment / check: bootstrap over samples or seeds for key tables.
- Suggested script or file to modify: paper asset generation scripts.
- Priority: Medium

## TODO-11: Abstract numerical claims need exact source comments
- Missing item: The abstract should cite exact tables internally during writing notes.
- Why it matters for CSF: Prevents untraceable numerical claims.
- Current evidence: Analytic calibrator, identifiability, zeta-noise bridge, and failure-attribution numbers are table-backed. Raw alignment \(0.73\rightarrow0.36\) can be linked to `outputs/tables/zeta_alignment_comparison/zeta_alignment_comparison.csv`.
- Risk if not addressed: Harder to audit final manuscript.
- Suggested experiment / check: Locate or regenerate raw zeta alignment before/after comparison CSV.
- Suggested script or file to modify: `experiments/compare_zeta_alignment_effect.py`.
- Priority: Medium

## TODO-12: Clarify whether the paper is about parameter recovery or diagnostic validity
- Missing item: A final decision in the manuscript text.
- Why it matters for CSF: Mixed framing weakens the paper.
- Current evidence: Final docs and refreshed identifiability tables strongly recommend diagnostic framing.
- Risk if not addressed: Reviewer may criticize the paper for failing at parameter recovery if it overclaims recovery.
- Suggested experiment / check: No new experiment; edit language consistently.
- Suggested script or file to modify: `paper_draft.md` / `main.tex`.
- Priority: High
