# TODO Evidence Completion Status

This file records which missing-evidence items were addressed using existing project scripts and outputs. No new CMIN-SR model, validity head, or backbone was trained.

| TODO | Status | Evidence / action | Remaining risk |
|---|---|---|---|
| Classical MFDFA / structure-function baseline comparison | Partially completed | Ran `scripts/run_baselines_improved.py` and `scripts/run_baselines_ensemble.py`; see `paper_assets/tables/todo_supplemental_baseline_summary.csv`. | WTMM/wavelet-leader comparison is still missing. |
| Direct neural spectrum regression baseline | Completed from historical outputs | Summarized `outputs/dl_spectrum_mlp`, `outputs/dl_spectrum_cnn`, and constrained PC-SMIN/final-hybrid metrics. | Historical split, not a newly standardized CMIN-SR final split. |
| Unconstrained H/lambda2 regression baseline | Completed from historical outputs | Same summary table includes unconstrained MLP/CNN parameter metrics and constrained PC-SMIN/final-hybrid metrics. | Use as appendix/ablation evidence, not main claim. |
| f(alpha) reconstruction visualization | Completed | Generated `paper_assets/figures/fig7_multifractal_spectrum_shapes.{png,pdf}`. | Analytic spectrum-shape figure, not empirical reconstruction accuracy. |
| Multiple random seed stability | Partially completed | Key scripts have seed controls and quick supplemental runs were refreshed at seed 2026. | Full 3-5 seed aggregation remains recommended before final submission. |
| Real-world complex-system validation | Assessed, not promoted to main | Existing Fama-French data are present, but current paper claim is diagnostic/identifiability; real data are useful as sanity check, not required to prove the core synthetic identifiability result. | CSF reviewers may still prefer one cautious real-world example. |
| Scale-length sensitivity | Completed quick refresh | `outputs/tables/finite_sample_identifiability/lambda2_recovery_by_scale_range.csv`. | Full grid with larger T and more samples would strengthen it. |
| q-grid sensitivity | Completed quick refresh | `outputs/tables/finite_sample_identifiability/lambda2_recovery_by_qgrid.csv`. | Full Q1-Q5 grid with more samples would strengthen it. |
| Zeta-space noise bridge | Completed refresh | `outputs/tables/zeta_noise_bridge/separation_margin_vs_noise.csv`. | Raw signal noise perturbation remains optional. |
| Statistical confidence intervals | Partially completed | Baseline table includes mean/std where sample-level data are available. | Bootstrap CIs for all main diagnostics remain optional polish. |
| Abstract numerical source comments | Improved | Refreshed tables for spectral geometry, noise bridge, failure attribution, identifiability. | Raw zeta alignment before/after numbers should stay sourced to comparison CSV/docs. |
| Diagnostic framing vs parameter recovery | Completed as writing decision | Evidence supports diagnostic framing; avoid guaranteed short-window `lambda2` recovery claim. | Must keep language consistent in the manuscript. |

## Baseline Evidence Summary

evidence_block,method_metric,mean,std,n,source
single_sample_classical_baseline,zeta_mae_sf,0.1143941492997356,0.10890088265699095,10.0,outputs/baselines/baseline_results_robust_improved_10.json
single_sample_classical_baseline,zeta_mae_mfdfa,0.07228891711525168,0.07766035830161473,10.0,outputs/baselines/baseline_results_robust_improved_10.json
single_sample_classical_baseline,spectrum_mae_sf,0.04128364772906182,0.025350523918286863,10.0,outputs/baselines/baseline_results_robust_improved_10.json
single_sample_classical_baseline,spectrum_mae_mfdfa,0.034190463092511406,0.02465647489631399,10.0,outputs/baselines/baseline_results_robust_improved_10.json
ensemble_high_lambda_mrw_baseline,zeta_mae_sf,0.29566064773295486,,128.0,outputs/baselines/baseline_ensemble_result.json
ensemble_high_lambda_mrw_baseline,zeta_mae_mfdfa,0.2986464388473657,,128.0,outputs/baselines/baseline_ensemble_result.json
ensemble_high_lambda_mrw_baseline,spectrum_mae_sf,0.2606751225614694,,128.0,outputs/baselines/baseline_ensemble_result.json
ensemble_high_lambda_mrw_baseline,spectrum_mae_mfdfa,0.2476248857299372,,128.0,outputs/baselines/baseline_ensemble_result.json
unconstrained_mlp,parameter_lambda2_mae,0.03936256095767021,,,outputs/dl_spectrum_mlp/metrics.json
unconstrained_mlp,parameter_lambda2_rmse,0.04546542465686798,,,outputs/dl_spectrum_mlp/metrics.json
unconstrained_mlp,parameter_lambda2_bias,-0.005439944099634886,,,outputs/dl_spectrum_mlp/metrics.json
unconstrained_mlp,parameter_H_mae,0.12199527025222778,,,outputs/dl_spectrum_mlp/metrics.json
unconstrained_mlp,parameter_H_rmse,0.14763690531253815,,,outputs/dl_spectrum_mlp/metrics.json
unconstrained_mlp,parameter_H_bias,0.015789184719324112,,,outputs/dl_spectrum_mlp/metrics.json
unconstrained_mlp,spectrum_zeta_mae,0.21432391727032174,,,outputs/dl_spectrum_mlp/metrics.json
unconstrained_mlp,spectrum_zeta_rmse,0.2845319800176814,,,outputs/dl_spectrum_mlp/metrics.json
unconstrained_mlp,spectrum_f_mae,0.07145646475096741,,,outputs/dl_spectrum_mlp/metrics.json
unconstrained_mlp,spectrum_f_rmse,0.10264910227608325,,,outputs/dl_spectrum_mlp/metrics.json
unconstrained_mlp,spectrum_alpha_mae,0.1266350486260519,,,outputs/dl_spectrum_mlp/metrics.json
unconstrained_cnn,parameter_lambda2_mae,0.03360655531287193,,,outputs/dl_spectrum_cnn/metrics.json
unconstrained_cnn,parameter_lambda2_rmse,0.041142575442790985,,,outputs/dl_spectrum_cnn/metrics.json
unconstrained_cnn,parameter_lambda2_bias,0.004572968930006027,,,outputs/dl_spectrum_cnn/metrics.json
unconstrained_cnn,parameter_H_mae,0.01814924366772175,,,outputs/dl_spectrum_cnn/metrics.json
unconstrained_cnn,parameter_H_rmse,0.023598376661539078,,,outputs/dl_spectrum_cnn/metrics.json
unconstrained_cnn,parameter_H_bias,-0.0014129268238320947,,,outputs/dl_spectrum_cnn/metrics.json
unconstrained_cnn,spectrum_zeta_mae,0.03821505907258005,,,outputs/dl_spectrum_cnn/metrics.json
unconstrained_cnn,spectrum_zeta_rmse,0.05338886641930567,,,outputs/dl_spectrum_cnn/metrics.json
unconstrained_cnn,spectrum_f_mae,0.061007356946092016,,,outputs/dl_spectrum_cnn/metrics.json
unconstrained_cnn,spectrum_f_rmse,0.09288924365958615,,,outputs/dl_spectrum_cnn/metrics.json
unconstrained_cnn,spectrum_alpha_mae,0.03729513898377796,,,outputs/dl_spectrum_cnn/metrics.json
pc_smin_constrained,parameter_lambda2_mae,0.03309771046042442,,,outputs/dl_spectrum_pc_smin/metrics.json
pc_smin_constrained,parameter_lambda2_rmse,0.0406157523393631,,,outputs/dl_spectrum_pc_smin/metrics.json
pc_smin_constrained,parameter_lambda2_bias,0.0016936000902205706,,,outputs/dl_spectrum_pc_smin/metrics.json
pc_smin_constrained,parameter_H_mae,0.017195114865899086,,,outputs/dl_spectrum_pc_smin/metrics.json
pc_smin_constrained,parameter_H_rmse,0.02155548334121704,,,outputs/dl_spectrum_pc_smin/metrics.json
pc_smin_constrained,parameter_H_bias,-0.0007748497882857919,,,outputs/dl_spectrum_pc_smin/metrics.json
pc_smin_constrained,spectrum_zeta_mae,0.03627019078178846,,,outputs/dl_spectrum_pc_smin/metrics.json
pc_smin_constrained,spectrum_zeta_rmse,0.0495613617421614,,,outputs/dl_spectrum_pc_smin/metrics.json
pc_smin_constrained,spectrum_f_mae,0.06008363352554046,,,outputs/dl_spectrum_pc_smin/metrics.json
pc_smin_constrained,spectrum_f_rmse,0.09169981139654873,,,outputs/dl_spectrum_pc_smin/metrics.json
pc_smin_constrained,spectrum_alpha_mae,0.03555129443576343,,,outputs/dl_spectrum_pc_smin/metrics.json
final_hybrid_constrained,parameter_lambda2_mae,0.0330977700650692,,,outputs/dl_spectrum_final_hybrid/metrics.json
final_hybrid_constrained,parameter_lambda2_rmse,0.040615808218717575,,,outputs/dl_spectrum_final_hybrid/metrics.json
final_hybrid_constrained,parameter_lambda2_bias,0.0016935454914346337,,,outputs/dl_spectrum_final_hybrid/metrics.json
final_hybrid_constrained,parameter_H_mae,0.012810206040740013,,,outputs/dl_spectrum_final_hybrid/metrics.json
final_hybrid_constrained,parameter_H_rmse,0.016289984807372093,,,outputs/dl_spectrum_final_hybrid/metrics.json
final_hybrid_constrained,parameter_H_bias,0.002053131116554141,,,outputs/dl_spectrum_final_hybrid/metrics.json
final_hybrid_constrained,spectrum_zeta_mae,0.029959992374510462,,,outputs/dl_spectrum_final_hybrid/metrics.json
final_hybrid_constrained,spectrum_zeta_rmse,0.041379495595842385,,,outputs/dl_spectrum_final_hybrid/metrics.json
final_hybrid_constrained,spectrum_f_mae,0.060083738675695546,,,outputs/dl_spectrum_final_hybrid/metrics.json
final_hybrid_constrained,spectrum_f_rmse,0.0916999320943589,,,outputs/dl_spectrum_final_hybrid/metrics.json
final_hybrid_constrained,spectrum_alpha_mae,0.03316710797336066,,,outputs/dl_spectrum_final_hybrid/metrics.json
