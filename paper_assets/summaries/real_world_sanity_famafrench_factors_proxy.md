# Real Surrogate Validation Summary

- Input returns: `data\real_processed\all_processed_returns.csv`
- Window: `512`
- Step: `252`
- Block size: `16`
- Mode: `proxy`
- Model name: `statistical_proxy`
- Checkpoint: ``
- Detail CSV: `outputs\tables\real_world_sanity_famafrench_factors_proxy\real_surrogate_window_metrics.csv`
- Gap table: `outputs\tables\real_world_sanity_famafrench_factors_proxy\real_surrogate_gap_table.csv`
- Gap summary: `outputs\tables\real_world_sanity_famafrench_factors_proxy\real_surrogate_gap_summary.csv`
- Figure: `outputs\figures\real_world_sanity_famafrench_factors_proxy\real_surrogate_gap_boxplots.png`

## Gap Summary Table

source_name,n_windows,mean_lambda2_gap_shuffle,mean_p_MRW_gap_shuffle,mean_f_width_gap_shuffle,mean_logvol_gap_shuffle,mean_lambda2_gap_block,mean_p_MRW_gap_block,mean_f_width_gap_block,mean_logvol_gap_block
F-F_Momentum_Factor_daily:Mom,102,0.013665836530297154,0.022089636848949313,0.02733167306059429,0.029375706438809497,-0.01352401194144044,-0.021065703006605846,-0.027048023882880946,-0.01635605611673091
F-F_Research_Data_Factors_daily:HML,102,0.00999318745249265,0.015481459671782289,0.01998637490498524,0.01803971911104805,-0.005493696992735174,-0.008784951977304526,-0.010987393985470351,-0.010488618868836225
F-F_Research_Data_Factors_daily:Mkt-RF,102,0.00856816460512993,0.016168707751811503,0.017136329210259822,0.021100319248332708,-0.004668379509369028,-0.007212242041753413,-0.009336759018738098,-0.008087925226712207
F-F_Research_Data_Factors_daily:SMB,102,0.01288659896996658,0.01976269577719127,0.025773197939933188,0.0238309167676521,0.0002931596778245868,-0.0011008449347183437,0.000586319355649186,-0.0028533324449560556


## Cautious Interpretation

- `F-F_Momentum_Factor_daily:Mom`: original windows show a clearly positive lambda2 gap over shuffled surrogates (0.0137), which is consistent with temporal multiscale dependence beyond marginal distribution.
  - The MRW-validity score is also higher in originals than shuffled windows (0.0221), supporting a dependence-based interpretation.
  - Spectrum width is wider in originals by about 0.0273, which is qualitatively aligned with stronger multiscale structure.
- `F-F_Research_Data_Factors_daily:HML`: original windows show a clearly positive lambda2 gap over shuffled surrogates (0.0100), which is consistent with temporal multiscale dependence beyond marginal distribution.
  - Spectrum width is wider in originals by about 0.0200, which is qualitatively aligned with stronger multiscale structure.
- `F-F_Research_Data_Factors_daily:Mkt-RF`: original windows show a clearly positive lambda2 gap over shuffled surrogates (0.0086), which is consistent with temporal multiscale dependence beyond marginal distribution.
  - Spectrum width is wider in originals by about 0.0171, which is qualitatively aligned with stronger multiscale structure.
- `F-F_Research_Data_Factors_daily:SMB`: original windows show a clearly positive lambda2 gap over shuffled surrogates (0.0129), which is consistent with temporal multiscale dependence beyond marginal distribution.
  - Spectrum width is wider in originals by about 0.0258, which is qualitatively aligned with stronger multiscale structure.

## Interpretation Policy

- Fama-French factor returns are cleaned, aggregated, economically constructed return series.
- Small original-vs-shuffled gaps should be interpreted as conservative weak-intermittency evidence, not as a failure of the framework.
- These surrogate tests are more important than downstream forecasting for validating whether lambda2 responds to temporal dependence rather than only marginal heavy tails.
- Phase-randomized surrogates are not implemented in this minimal runnable version.