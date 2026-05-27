# Paper Material Inventory

This inventory records what can be safely used for a Chaos, Solitons & Fractals
style paper draft from the current repository. It is written conservatively:
claims are included only when supported by existing code, tables, figures, docs,
or notes.

## Supplemental Evidence Update: 2026-05-11

Additional paper-supporting evidence was generated without training a new
CMIN-SR model:

- Classical structure-function / MFDFA baseline outputs:
  `outputs/baselines/baseline_results_robust_improved_10.json` and
  `outputs/baselines/baseline_ensemble_result.json`.
- Consolidated baseline table:
  `paper_assets/tables/todo_supplemental_baseline_summary.csv`.
- Analytic multifractal spectrum-shape figure:
  `paper_assets/figures/fig7_multifractal_spectrum_shapes.pdf`.
- Refreshed finite-sample identifiability, q-grid, scale-range, zeta-noise
  bridge, and failure-attribution outputs under `outputs/tables/`.
- Real-world data necessity assessment:
  `paper_assets/summaries/real_world_data_necessity_assessment.md`.

## Available Writing Materials

- Paper workspace:
  - `paper_writing_workspace/main.tex`
  - `paper_writing_workspace/sections/*.tex`
  - `paper_writing_workspace/ARTICLE_STRUCTURE_CN.md`
  - `paper_writing_workspace/literature/references_curated.bib`
  - `paper_writing_workspace/literature/citation_usage_guide.md`
- Final project notes:
  - `paper_writing_workspace/notes/final_project_manifest.md`
  - `paper_writing_workspace/notes/final_summary_for_paper.md`
  - `paper_writing_workspace/notes/spectral_representation_framework.md`
  - `paper_writing_workspace/notes/finite_sample_curvature_identifiability.md`
  - `paper_writing_workspace/notes/zeta_noise_bridge.md`
  - `paper_writing_workspace/notes/cmin_sr_failure_attribution.md`
  - `paper_writing_workspace/notes/core_code_index.md`
  - `paper_writing_workspace/notes/legacy_experiments_map.md`
- Paper assets:
  - `paper_writing_workspace/figures/*.pdf`
  - `paper_writing_workspace/latex_tables/*.tex`
  - `paper_writing_workspace/tables/*.csv`

## Confirmed Method Modules

- Empirical spectrum and robust zeta estimation:
  - `src/mrw_inverse/models/empirical_spectrum.py`
  - `src/mrw_inverse/models/robust_zeta_estimator.py`
- Projection geometry:
  - `src/mrw_inverse/models/mrw_projection.py`
  - `src/mrw_inverse/models/monofractal_projection.py`
- Curvature and spectral interpretation:
  - `src/mrw_inverse/models/curvature_diagnostics.py`
  - `src/mrw_inverse/models/spectral_geometry_calibrator.py`
- CMIN-SR model family and zeta-aligned encoders:
  - `src/mrw_inverse/models/spectral_representation_model.py`
  - `src/mrw_inverse/models/zeta_aligned_encoder.py`
- Datasets:
  - `src/mrw_inverse/data/analytic_spectrum_dataset.py`
  - `src/mrw_inverse/data/raw_zeta_alignment_dataset.py`
  - `src/mrw_inverse/data/boundary_calibration_dataset.py`
- Losses:
  - `src/mrw_inverse/losses/spectrum_space_calibration_losses.py`
  - `src/mrw_inverse/losses/zeta_alignment_losses.py`
  - `src/mrw_inverse/losses/curvature_preserving_zeta_losses.py`
  - `src/mrw_inverse/losses/spectral_representation_losses.py`
- Analysis:
  - `src/mrw_inverse/analysis/curvature_identifiability.py`

## Confirmed Mathematical / Physical Constraints

- Monofractal projection:
  - \(\zeta_{\mathrm{mono}}(q)=qH\)
  - Evidence: `docs/spectral_representation_framework.md`,
    `src/mrw_inverse/models/monofractal_projection.py`.
- MRW projection:
  - \(\zeta_{\mathrm{MRW}}(q)=qH-\frac{1}{2}\lambda^2 q(q-2)\)
  - Evidence: `docs/spectral_representation_framework.md`,
    `src/mrw_inverse/models/mrw_projection.py`.
- `lambda2_proj` is a projection coordinate, not proof of a true MRW mechanism.
  - Evidence: `docs/final_project_manifest.md`,
    `docs/spectral_representation_framework.md`.
- Spectral interpretation is decomposed into:
  - \(p_{\mathrm{scaling}}\): scaling stability;
  - \(p_{\mathrm{curved}}\): nonlinear curvature;
  - \(p_{\mathrm{mono}}\): monofractal compatibility;
  - \(p_{\mathrm{MRW}}\): MRW-compatible curved spectral evidence;
  - boundary and instability diagnostics.
  - Evidence: `docs/spectral_representation_framework.md`,
    `src/mrw_inverse/models/spectral_geometry_calibrator.py`.

## Confirmed Data Generation and Process Families

The repository supports controlled synthetic families used across CMIN-SR
experiments:

- MRW / low-lambda MRW / medium-high lambda MRW.
- fGn / fractional Gaussian noise style monofractal controls.
- iid Gaussian controls.
- iid Student-t heavy-tail controls.
- GARCH(1,1) volatility-clustering controls.
- Regime-switching Gaussian controls.
- Shuffled MRW / surrogate variants in historical diagnostics.

Evidence:

- `src/mrw_inverse/data/analytic_spectrum_dataset.py`
- `src/mrw_inverse/data/raw_zeta_alignment_dataset.py`
- `src/mrw_inverse/data/boundary_calibration_dataset.py`
- `outputs/tables/curvature_preserving_zeta_eval/process_by_T_band.csv`
- `outputs/tables/cmin_sr_v3_eval/process_by_T.csv`

## Confirmed Experimental Settings

Settings explicitly present in experiment scripts and reports include:

- q-grid commonly used:
  - \([0.5,1.0,1.5,2.0,2.5,3.0]\)
- finite-sample evaluation lengths:
  - \(T=256,512,1024,2048\) in several evaluations;
  - finite-sample identifiability quick study includes \(T=512,1024,2048\).
- deterministic estimator comparison:
  - `structure_ols`
  - `structure_trimmed`
  - `structure_bootstrap`
  - `structure_smoothed`
- scale sets and q-grid sensitivity:
  - implemented by `experiments/run_scale_length_sensitivity.py`
  - implemented by `experiments/run_qgrid_sensitivity.py`

## Confirmed Metrics

- Calibrated spectral probabilities:
  - \(p_{\mathrm{scaling}}\), \(p_{\mathrm{curved}}\),
    \(p_{\mathrm{mono}}\), \(p_{\mathrm{MRW}}\), \(p_{\mathrm{boundary}}\).
- Projection quantities:
  - \(H_{\mathrm{proj}}\), \(\lambda^2_{\mathrm{proj}}\), \(H_{\mathrm{mono}}\).
- Residual geometry:
  - MRW residual norm;
  - monofractal residual norm;
  - MRW-vs-mono gain.
- Curvature diagnostics:
  - curvature score;
  - second-difference norm;
  - third-difference norm;
  - linearity score;
  - boundary score.
- Finite-sample identifiability metrics:
  - lambda2 MAE/RMSE;
  - correlation and Spearman correlation between true and projected lambda2;
  - high-lambda detection rate;
  - boundary accuracy;
  - warning rate.

## Confirmed Experimental Results

### Analytic spectrum-space calibration

Source:

- `outputs/tables/spectral_geometry_calibrator_eval/summary_by_spectrum_type.csv`

Representative values:

- `linear_mono`: \(p_{\mathrm{curved}}\approx0.026\),
  \(p_{\mathrm{MRW}}\approx0.130\), \(p_{\mathrm{mono}}\approx0.969\).
- `boundary_mrw`: \(p_{\mathrm{MRW}}\approx0.352\),
  \(p_{\mathrm{mono}}\approx0.684\), \(p_{\mathrm{curved}}\approx0.276\).
- `curved_mrw`: \(p_{\mathrm{curved}}\approx0.888\),
  \(p_{\mathrm{MRW}}\approx0.935\), \(p_{\mathrm{mono}}\approx0.178\).
- `heavy_tail_distorted`: \(p_{\mathrm{MRW}}\approx0.055\).
- `regime_apparent`: \(p_{\mathrm{MRW}}\approx0.125\).

Conclusion supported:

- The spectral geometry calibrator works in clean analytic spectrum space.

### Zeta noise bridge

Source:

- `outputs/tables/zeta_noise_bridge/separation_margin_vs_noise.csv`

Representative values:

- MRW-vs-linear \(p_{\mathrm{MRW}}\) separation margin decreases from
  approximately \(0.805\) at noise level 0 to approximately \(0.392\) at noise
  level 0.1.

Conclusion supported:

- Noise in zeta-space can degrade MRW/fGn separation and helps explain why raw
  calibrated outputs may fail.

### Failure attribution

Source:

- `outputs/tables/cmin_sr_failure_attribution/failure_attribution_summary_table.csv`

Representative values:

- High-lambda MRW analytic spectrum: \(p_{\mathrm{MRW}}\approx0.931\).
- High-lambda MRW deterministic zeta: \(p_{\mathrm{MRW}}\approx0.142\).
- High-lambda MRW neural zeta: \(p_{\mathrm{MRW}}\approx0.242\).
- fGn analytic spectrum: \(p_{\mathrm{MRW}}\approx0.132\).
- fGn deterministic zeta: \(p_{\mathrm{MRW}}\approx0.135\).
- fGn neural zeta: \(p_{\mathrm{MRW}}\approx0.290\).

Conclusion supported:

- Analytic spectral geometry is separable, but deterministic and neural raw
  zeta estimates do not recover strong MRW curvature reliably in the tested
  setting.

### Finite-sample curvature identifiability

Source:

- `outputs/tables/finite_sample_identifiability/lambda2_recovery_by_T.csv`

Representative values:

- At \(T=512\), lambda2 correlation is negative for all implemented estimators.
- At \(T=1024\), best lambda2 correlation is approximately \(0.136\).
- At \(T=2048\), lambda2 correlation is near zero or negative.
- High-lambda detection rate is 0.0 in the quick diagnostic grid.

Conclusion supported:

- Under the current q-grid, scale range, and structure-function estimator
  family, short-window MRW curvature recovery is unreliable.

## Confirmed Figures and Visualizations

- `paper_writing_workspace/figures/fig1_cmin_sr_framework.pdf`
- `paper_writing_workspace/figures/fig2_spectral_geometry_calibration.pdf`
- `paper_writing_workspace/figures/fig3_process_family_map.pdf`
- `paper_writing_workspace/figures/fig4_mrw_vs_mono_projection.pdf`
- `paper_writing_workspace/figures/fig5_finite_sample_identifiability.pdf`
- `paper_writing_workspace/figures/fig6_zeta_noise_bridge.pdf`
- `paper_writing_workspace/figures/fig6_failure_attribution.pdf`

## Best Current Paper Story

The strongest CSF-style story is not that a neural network solves MRW parameter
recovery. The strongest story is:

1. Complex stochastic signals can exhibit apparent multifractal scaling.
2. Finite samples, high-order moments, heavy tails, and regime changes make
   empirical spectral curvature ambiguous.
3. CMIN-SR separates empirical spectrum estimation from spectral geometry
   interpretation.
4. Analytic spectral geometry is learnable and physically interpretable.
5. Raw empirical spectrum estimation is the bottleneck.
6. Deterministic estimators also fail to recover lambda2 reliably at short T,
   indicating an estimator-level finite-sample identifiability limitation.
7. Therefore CMIN-SR is best framed as a calibrated spectral diagnostic
   framework, not as a guaranteed short-window MRW parameter recovery engine.

## Uncertain or Missing Information

- No confirmed MFDFA or WTMM baseline result tables are present.
- No strong direct comparison against unconstrained neural spectrum regression
  is confirmed in the final paper assets.
- No robust real-world complex-system validation should be claimed as a main
  result unless additional data and reports are added.
- Current main evidence is synthetic / controlled.
- No strong table of \(H\) and \(\lambda^2\) estimation accuracy for a final
  successful model should be claimed; identifiability results are mostly
  negative for lambda2.
- No confirmed \(f(\alpha)\) reconstruction figure/table is available in the
  final paper assets.
