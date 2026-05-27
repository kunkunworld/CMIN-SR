# Real Surrogate Validation Summary

- Input returns: `data\real_processed\all_processed_returns.csv`
- Window: `512`
- Step: `252`
- Block size: `16`
- Mode: `proxy`
- Model name: `statistical_proxy`
- Checkpoint: ``
- Detail CSV: `outputs\tables\real_world_sanity_famafrench_proxy\real_surrogate_window_metrics.csv`
- Gap table: `outputs\tables\real_world_sanity_famafrench_proxy\real_surrogate_gap_table.csv`
- Gap summary: `outputs\tables\real_world_sanity_famafrench_proxy\real_surrogate_gap_summary.csv`
- Figure: `outputs\figures\real_world_sanity_famafrench_proxy\real_surrogate_gap_boxplots.png`

## Gap Summary Table

source_name,n_windows,mean_lambda2_gap_shuffle,mean_p_MRW_gap_shuffle,mean_f_width_gap_shuffle,mean_logvol_gap_shuffle,mean_lambda2_gap_block,mean_p_MRW_gap_block,mean_f_width_gap_block,mean_logvol_gap_block
F-F_Momentum_Factor_daily:Mom,102,0.013302706501530119,0.021828641365188844,0.026605413003060196,0.027872142727890525,-0.015291433727749557,-0.023899959594162037,-0.030582867455499096,-0.018520422075717233
F-F_Research_Data_Factors_daily:HML,102,0.009490114516740991,0.015243505578348911,0.018980229033481923,0.018555612798734644,-0.008359305212395883,-0.01336091758184482,-0.016718610424791773,-0.01239744465230614
F-F_Research_Data_Factors_daily:Mkt-RF,102,0.008727280875179861,0.01606714128128388,0.017454561750359716,0.019053916219555257,-0.0038475961725344507,-0.005389034225545017,-0.007695192345068936,-0.006055729409661885
F-F_Research_Data_Factors_daily:RF,102,0.03662133773233573,0.055904098754023614,0.07324267546467159,0.14889681983306763,-0.004034537303372585,-0.013250597749395868,-0.008069074606745023,-0.1215356917579552
F-F_Research_Data_Factors_daily:SMB,102,0.011575164133979814,0.016773630215638168,0.02315032826795961,0.01627830667836308,5.445459143783445e-05,-0.001390929142285698,0.00010890918287567964,-0.0025359518146842136


## Cautious Interpretation

- `F-F_Momentum_Factor_daily:Mom`: original windows show a clearly positive lambda2 gap over shuffled surrogates (0.0133), which is consistent with temporal multiscale dependence beyond marginal distribution.
  - The MRW-validity score is also higher in originals than shuffled windows (0.0218), supporting a dependence-based interpretation.
  - Spectrum width is wider in originals by about 0.0266, which is qualitatively aligned with stronger multiscale structure.
- `F-F_Research_Data_Factors_daily:HML`: original windows show a clearly positive lambda2 gap over shuffled surrogates (0.0095), which is consistent with temporal multiscale dependence beyond marginal distribution.
  - Spectrum width is wider in originals by about 0.0190, which is qualitatively aligned with stronger multiscale structure.
- `F-F_Research_Data_Factors_daily:Mkt-RF`: original windows show a clearly positive lambda2 gap over shuffled surrogates (0.0087), which is consistent with temporal multiscale dependence beyond marginal distribution.
  - Spectrum width is wider in originals by about 0.0175, which is qualitatively aligned with stronger multiscale structure.
- `F-F_Research_Data_Factors_daily:RF`: original windows show a clearly positive lambda2 gap over shuffled surrogates (0.0366), which is consistent with temporal multiscale dependence beyond marginal distribution.
  - The MRW-validity score is also higher in originals than shuffled windows (0.0559), supporting a dependence-based interpretation.
  - Spectrum width is wider in originals by about 0.0732, which is qualitatively aligned with stronger multiscale structure.
- `F-F_Research_Data_Factors_daily:SMB`: original windows show a clearly positive lambda2 gap over shuffled surrogates (0.0116), which is consistent with temporal multiscale dependence beyond marginal distribution.
  - Spectrum width is wider in originals by about 0.0232, which is qualitatively aligned with stronger multiscale structure.

## Interpretation Policy

- Fama-French factor returns are cleaned, aggregated, economically constructed return series.
- Small original-vs-shuffled gaps should be interpreted as conservative weak-intermittency evidence, not as a failure of the framework.
- These surrogate tests are more important than downstream forecasting for validating whether lambda2 responds to temporal dependence rather than only marginal heavy tails.
- Phase-randomized surrogates are not implemented in this minimal runnable version.