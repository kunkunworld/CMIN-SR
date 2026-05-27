# Legacy Experiments Map

## 1. Proxy Estimator
- Purpose: deterministic baseline before CMIN models.
- Key result: useful sanity check, not final inference framework.
- Main paper: no.
- Appendix/failure analysis: optional.
- Outputs: `outputs/tables/proxy_vs_cmin/`, `outputs/tables/estimator_comparison/`.
- Checkpoint: none.

## 2. Tiny CMIN
- Purpose: minimal neural validity baseline.
- Key result: exposes always-MRW bias.
- Main paper: no.
- Appendix/failure analysis: yes.
- Outputs: `outputs/tables/cmin_eval/`.
- Checkpoint: `checkpoints/cmin/cmin_tiny_synthetic.pt`.

## 3. CMIN-Robust
- Purpose: anti-confounding with shuffled, heavy-tail, GARCH, and regime controls.
- Key result: improves caution under negative controls.
- Main paper: optional background.
- Appendix/failure analysis: yes.
- Outputs: `outputs/tables/cmin_robust_eval/`, `outputs/tables/cmin_robust_multilength_eval/`.
- Checkpoints: `cmin_robust_synthetic.pt`, `cmin_robust_multilength.pt`.

## 4. CMIN-SR v1
- Purpose: first spectral-representation model.
- Key result: fGn over-compatible with MRW.
- Main paper: no.
- Appendix/failure analysis: yes.
- Outputs: `outputs/tables/cmin_sr_eval/`.
- Checkpoint: `cmin_sr_synthetic.pt`.

## 5. CMIN-SR v2
- Purpose: add monofractal competition.
- Key result: lowers fGn `p_MRW`, but also lowers MRW `p_MRW`.
- Main paper: no.
- Appendix/failure analysis: yes.
- Outputs: `outputs/tables/cmin_sr_v2_eval/`.
- Checkpoint: `cmin_sr_v2_synthetic.pt`.

## 6. CMIN-SR v3
- Purpose: add explicit `p_curved` and boundary diagnostics.
- Key result: engineering success, but fGn/MRW remain too similar.
- Main paper: projection definitions can be cited.
- Appendix/failure analysis: yes.
- Outputs: `outputs/tables/cmin_sr_v3_eval/`.
- Checkpoint: `cmin_sr_v3_synthetic.pt`.

## 7. Boundary Calibration
- Purpose: same-H fGn/MRW contrastive calibration.
- Key result: raw finite-sample contrast is insufficient.
- Main paper: no.
- Appendix/failure analysis: yes.
- Outputs: `outputs/tables/cmin_sr_boundary_calibrated_eval/`.
- Checkpoint: `cmin_sr_calibrated_synthetic.pt`.

## 8. Spectrum-Space Calibration
- Purpose: train geometry interpretation directly on analytic spectra.
- Key result: succeeds cleanly; proves interpretation layer is learnable.
- Main paper: yes.
- Appendix/failure analysis: no, except detailed sweeps.
- Outputs: `outputs/tables/spectral_geometry_calibrator_eval/`.
- Checkpoint: `spectral_geometry_calibrator.pt`.

## 9. Raw Zeta Alignment
- Purpose: improve raw signal to `zeta_emp(q)`.
- Key result: suppresses fake fGn/Gaussian curvature but over-linearizes MRW.
- Main paper: no.
- Appendix/failure analysis: yes.
- Outputs: `outputs/tables/raw_zeta_alignment_eval/`.
- Checkpoint: `cmin_sr_zeta_aligned.pt`.

## 10. Curvature-Preserving Zeta Alignment
- Purpose: preserve MRW curvature while keeping fGn/Gaussian clean.
- Key result: confirms fGn rejection vs MRW recovery trade-off.
- Main paper: optional supporting diagnostic.
- Appendix/failure analysis: yes.
- Outputs: `outputs/tables/curvature_preserving_zeta_eval/`.
- Checkpoint: `cmin_sr_zeta_curvature_preserved.pt`.

## 11. Finite-Sample Identifiability
- Purpose: test whether deterministic estimators recover MRW curvature at all.
- Key result: short-window `lambda2` recovery is poor under current estimators.
- Main paper: yes.
- Appendix/failure analysis: detailed q-grid and noise bridge.
- Outputs: `outputs/tables/finite_sample_identifiability/`,
  `outputs/tables/zeta_noise_bridge/`,
  `outputs/tables/cmin_sr_failure_attribution/`.
- Checkpoint: uses `spectral_geometry_calibrator.pt`; no new CMIN-SR checkpoint.
