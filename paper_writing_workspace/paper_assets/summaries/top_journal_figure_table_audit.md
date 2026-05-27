# Top-Journal Figure and Table Audit

This audit treats the manuscript as a CSF-style paper on finite-sample spectral diagnostics. The standard used here is not whether a figure is merely readable, but whether it can survive a skeptical reviewer who asks what each visual contributes to the scientific argument.

## Figure 1: CMIN-SR diagnostic pipeline
- Current problems:
  - The current figure reads like a temporary block diagram rather than a polished conceptual figure.
  - It does not visually separate spectrum estimation, projection geometry, and calibrated diagnostic interpretation.
  - The conservative interpretation, namely diagnostic evidence organization rather than MRW mechanism proof, is not prominent enough.
- Top-journal judgment: needs redraw.
- Concrete recommendation:
  - Redraw as a three-stage left-to-right pipeline:
    1. finite increments to empirical zeta estimation;
    2. monofractal and MRW projections with residual geometry;
    3. calibrated diagnostic scores plus warnings and conservative interpretation.
  - Use parallel branches for monofractal and MRW projection, then merge into residual/curvature/boundary/instability features.
  - Add a final callout: "diagnostic evidence, not mechanism proof."
- Raw data export required: yes.
- Exported files:
  - `paper_assets/figure_data/fig1_pipeline_structure.md`
  - `paper_assets/figure_data/fig1_pipeline_mermaid.md`
  - `paper_assets/figure_data/fig1_redesign_prompt.md`

## Figure 2: MRW simulation example
- Current problems:
  - Useful, but should avoid looking like a generic time-series plot.
  - The MRW construction should make clear which quantities are simulated and which are analytic/reference curves.
  - If too many panels are squeezed, labels may become too small.
- Top-journal judgment: keep concept, redraw for polish.
- Concrete recommendation:
  - Use a four-panel figure: volatility/log-volatility driver, increments, cumulative path, and analytic zeta curves for linear/boundary/curved cases.
  - Show only a representative time window if the full series is visually dense.
  - Caption should state that the figure is illustrative and not an empirical result.
- Raw data export required: yes.
- Exported files:
  - `paper_assets/figure_data/fig2_mrw_simulation_timeseries.csv`
  - `paper_assets/figure_data/fig2_mrw_simulation_zeta.csv`
  - `paper_assets/figure_data/fig2_redraw_notes.md`

## Figure 3: Analytic monofractal and MRW spectrum geometry
- Current problems:
  - Scientifically important, but it may overlap with Figure 2 if both show zeta curves.
  - If alpha and f(alpha) are included, axes must be clearly labeled and the MRW concavity convention must be visible.
- Top-journal judgment: keep as an independent theory figure if page budget allows; otherwise merge with Figure 2 as a multi-panel method figure.
- Concrete recommendation:
  - Draw zeta(q), alpha(q), and f(alpha) in aligned panels.
  - Use line styles rather than many saturated colors, so it survives grayscale.
- Raw data export required: yes.
- Exported file:
  - `paper_assets/figure_data/fig3_analytic_spectrum_geometry.csv`

## Figure 4: Spectral geometry map
- Current problems:
  - Scatter maps can look persuasive without communicating uncertainty.
  - If many points overlap, process-family separation may be visually overstated or understated.
- Top-journal judgment: redraw or refine.
- Concrete recommendation:
  - Use p_curved vs p_MRW with process color and marker shape.
  - Add mean markers and, if seed/run data are available, error bars or translucent ellipses.
  - Keep individual points faint and show category centers prominently.
- Raw data export required: yes.
- Exported file:
  - `paper_assets/figure_data/fig4_spectral_geometry_map.csv`

## Figure 5: Finite-sample curvature recovery
- Current problems:
  - This is the strongest negative/diagnostic result, but the figure should carry the main message more directly than a large estimator table.
  - Error bars or seed variability should be added when available.
- Top-journal judgment: should be a main-text figure; likely more important than the full Table 3.
- Concrete recommendation:
  - Plot lambda2 correlation vs T, with estimator lines.
  - Add a second panel for lambda2 MAE or high-lambda detection rate.
  - Use a horizontal zero-correlation line.
  - Caption should say that weak recovery under short windows supports an estimator-level finite-sample limitation.
- Raw data export required: yes.
- Exported file:
  - `paper_assets/figure_data/fig5_lambda2_recovery.csv`

## Figure 6: Zeta noise bridge
- Current problems:
  - Needs to state whether curves are averaged over perturbation types/seeds.
  - Without uncertainty bands, a reviewer may question robustness.
- Top-journal judgment: keep, but redraw if possible.
- Concrete recommendation:
  - Plot MRW/fGn separation margin vs noise level.
  - Separate smooth, jagged, and high-q perturbation types if available.
  - Add error bands if seed/run columns support it.
- Raw data export required: yes.
- Exported file:
  - `paper_assets/figure_data/fig6_zeta_noise_bridge.csv`

## Figure 7: Failure attribution
- Current problems:
  - This figure must make the three-level comparison visually obvious: analytic zeta, deterministic zeta, neural zeta.
  - If shown as many bars without grouping, the message becomes weak.
- Top-journal judgment: keep, redraw as grouped bar/point plot.
- Concrete recommendation:
  - Use grouped bars or connected points for p_MRW and p_curved across analytic/deterministic/neural sources.
  - Add zeta MAE as a secondary panel rather than a secondary axis if possible.
- Raw data export required: yes.
- Exported file:
  - `paper_assets/figure_data/fig7_failure_attribution.csv`

## Figure 8: Projection residual geometry
- Current problems:
  - Needs a clear y=x reference line and consistent axis limits.
  - Process labels should not crowd the plot.
- Top-journal judgment: keep, redraw for polish.
- Concrete recommendation:
  - Plot MRW residual vs monofractal residual with a diagonal reference line.
  - Color by process family and optionally size/shape by lambda2 band.
  - The caption should explain which side of the diagonal favors which projection.
- Raw data export required: yes.
- Exported file:
  - `paper_assets/figure_data/fig8_projection_residual_geometry.csv`

## Figure 9: Real-data sanity check
- Current problems:
  - The figure is not central to the synthetic/identifiability thesis.
  - If the display is only a PNG with limited explanation, it may look weaker than the controlled experiments.
- Top-journal judgment: move to appendix or keep as a small sanity-check figure only if layout is clean.
- Concrete recommendation:
  - Use a compact paired bar plot: original vs shuffled values by factor/metric.
  - State explicitly that this is not real-world validation of an MRW mechanism.
  - If space is tight, convert to a small table in appendix.
- Raw data export required: yes.
- Exported file:
  - `paper_assets/figure_data/fig9_real_data_sanity_check.csv`

## Table 1: Process-family calibrated diagnostics
- Current problems:
  - The original table was too wide and partially unhelpful for main text.
  - Numeric columns used code-like names and too many process/T combinations.
- Top-journal judgment: keep as compact summary; full data belongs in assets/supplement.
- Concrete recommendation:
  - Main text should show T=1024 process-family means with p_curved, p_MRW, p_mono, and p_boundary.
  - Full T/process table should remain as CSV.
- Raw data export required: yes.
- Exported files:
  - `paper_assets/tables/table1_process_family_diagnostics_compact.csv`
  - `paper_assets/tables/table1_process_family_diagnostics.csv`

## Table 2: MRW and monofractal projection diagnostics
- Current problems:
  - The previous LaTeX table was effectively truncated to a few identifier columns.
  - It did not expose residual geometry.
- Top-journal judgment: keep after compaction; consider appendix if Figure 8 carries the residual argument.
- Concrete recommendation:
  - Keep lambda2 projection, MRW residual, monofractal residual, gain, and instability.
  - Use Figure 8 as the visual version.
- Raw data export required: yes.
- Exported files:
  - `paper_assets/tables/table2_mrw_mono_projection_compact.csv`
  - `paper_assets/tables/table2_mrw_mono_projection.csv`

## Table 3: Finite-sample lambda2 recovery
- Current problems:
  - The full estimator table is too wide and weak as a main-text table.
  - The important message is not the exact value for every estimator, but the lack of stable short-window recovery.
- Top-journal judgment: compact in main text; full table to supplement/assets; Figure 5 should carry the main result.
- Concrete recommendation:
  - Main text: one row per T with mean/best/worst correlation and detection rate.
  - Supplement/assets: full estimator-level MAE/RMSE/correlation table.
- Raw data export required: yes.
- Exported files:
  - `paper_assets/tables/table3_finite_sample_lambda2_recovery_compact.csv`
  - `paper_assets/tables/table3_finite_sample_lambda2_recovery_full.csv`
  - `paper_assets/tables/fig_table3_lambda2_recovery_plot_data.csv`

## Table 4: CMIN-SR ablation and version comparison
- Current problems:
  - The previous table showed only identifiers and not the relevant diagnostic score.
  - A full historical version table is too much for the main paper.
- Top-journal judgment: keep compact version in main or move to appendix depending page budget.
- Concrete recommendation:
  - Main text can show p_MRW at T=1024 for MRW, fGn, Gaussian, and Student-t across v1/v2/v3/calibrated.
  - Full version history belongs in appendix or repository assets.
- Raw data export required: yes.
- Exported files:
  - `paper_assets/tables/table4_ablation_compact.csv`
  - `paper_assets/tables/table3_ablation_or_version_comparison.csv`
