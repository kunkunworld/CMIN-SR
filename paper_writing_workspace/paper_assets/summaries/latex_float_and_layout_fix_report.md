# LaTeX Float and Layout Fix Report

## Fixes Applied

- Replaced the oversized finite-sample estimator table in the Results section with a compact finite-sample lambda2 recovery table:
  - `latex_tables/table3_finite_sample_lambda2_recovery.tex`
  - Source CSV: `paper_assets/tables/table3_finite_sample_lambda2_recovery_compact.csv`
- Kept the full estimator-level data without changing values:
  - `paper_assets/tables/table3_finite_sample_lambda2_recovery_full.csv`
  - `paper_assets/tables/fig_table3_lambda2_recovery_plot_data.csv`
- Replaced the historical ablation table input with a compact version-comparison table:
  - `latex_tables/table4_ablation.tex`
  - Source CSV: `paper_assets/tables/table4_ablation_compact.csv`
- Rebuilt Table 1 into a compact process-family diagnostic table:
  - `latex_tables/table1_process_family_diagnostics.tex`
  - Source CSV: `paper_assets/tables/table1_process_family_diagnostics_compact.csv`
- Rebuilt Table 2 into a compact projection/residual table:
  - `latex_tables/table2_mrw_mono_projection.tex`
  - Source CSV: `paper_assets/tables/table2_mrw_mono_projection_compact.csv`
- Confirmed that Figure 9 has an available source file and exported its underlying sanity-check CSV:
  - Figure file: `figures/fig8_real_world_sanity_famafrench_factors_proxy.png`
  - Data: `paper_assets/figure_data/fig9_real_data_sanity_check.csv`
- Updated two risk-prone phrases:
  - "validates the analytic spectral geometry" -> "checks the analytic spectral geometry"
  - "true MRW curvature" -> "curvature in generated MRW samples"

## Layout Recommendations

- Figure 1 should be redrawn; the current diagram is too schematic for a main conceptual figure.
- Figure 5 should be visually emphasized because it communicates the finite-sample identifiability limitation more effectively than the compact Table 3.
- Full estimator-level Table 3 should not be in the main text. Keep it in assets/supplement.
- The real-data sanity check should be treated as appendix-level evidence unless it is redrawn into a compact, clear paired comparison.
- If any table still overflows after local compilation, use `\small` plus `tabular*` or move the table to appendix rather than shrinking it further.

## Build Note

The local shell used for this audit could not run `pdflatex` because MiKTeX was not on the command PATH in this environment. The LaTeX source was therefore checked and edited directly, but the final visual compile should be verified in VS Code/LaTeX Workshop.
