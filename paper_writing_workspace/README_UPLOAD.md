# Overleaf / LaTeX Upload Bundle

This folder is the compact writing package for the CMIN-SR paper.

## What to Upload

Upload the whole `paper_upload_bundle` folder to Overleaf, or upload the zip:

`paper_upload_bundle.zip`

Recommended Overleaf root file:

`main.tex`

## Folder Contents

- `main.tex`: paper skeleton with figure and table calls.
- `sections/`: section-level writing prompts and draft structure.
- `figures/`: paper-ready PNG/PDF figures.
- `tables/`: CSV source tables.
- `latex_tables/`: generated booktabs-style LaTeX table snippets.
- `notes/`: project manifest, final paper summary, framework notes, and failure-analysis notes.

## Suggested Paper Thesis

CMIN-SR is a validity-aware spectral diagnostic framework for finite stochastic signals. It can learn clean spectral geometry, but short-window MRW curvature recovery is limited by empirical spectrum estimation and finite-sample identifiability.

## Claims To Use

- Analytic spectral geometry is learnable.
- Monofractal, boundary MRW, and curved MRW spectra are separable in clean zeta space.
- Raw empirical zeta estimation is the bottleneck.
- Deterministic estimators also struggle to recover lambda2 at short T.
- CMIN-SR should report calibrated diagnostics, not overclaim mechanism recovery.

## Claims To Avoid

- Do not claim guaranteed short-window lambda2 recovery.
- Do not claim lambda2_proj proves a true MRW mechanism.
- Do not make real-world volatility forecasting the central validation.
- Do not present failed ablations as final models.

## Main Figures

- `fig1_cmin_sr_framework`
- `fig2_spectral_geometry_calibration`
- `fig3_process_family_map`
- `fig4_mrw_vs_mono_projection`
- `fig5_finite_sample_identifiability`
- `fig6_zeta_noise_bridge` or `fig6_failure_attribution`

## Main Tables

- `table1_process_family_diagnostics`
- `table2_mrw_mono_projection`
- `table3_ablation`
- `table4_identifiability`

