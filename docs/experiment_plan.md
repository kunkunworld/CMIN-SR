# Final Experiment Plan

This is the final paper-oriented experiment map. No new CMIN-SR versions,
validity heads, large backbones, or new research directions are planned.

## Main Experiments

### 1. Spectral Geometry Calibration

Purpose:
- show that linear monofractal, boundary MRW, curved MRW, and unstable spectra
  are separable in clean `zeta(q)` space.

Entry:

```bash
python experiments/paper/run_exp1_spectral_geometry_calibration.py --quick
```

Primary outputs:
- `outputs/tables/spectral_geometry_calibrator_eval/summary_by_spectrum_type.csv`
- `outputs/tables/spectral_geometry_calibrator_eval/lambda2_sweep.csv`

### 2. Controlled Stochastic Family Diagnostics

Purpose:
- evaluate calibrated spectral diagnostics across MRW, low-lambda MRW, fGn,
  Gaussian, Student-t, GARCH, and regime-switching processes.

Entry:

```bash
python experiments/paper/run_exp2_process_family_diagnostics.py --quick
```

Primary outputs:
- `outputs/tables/curvature_preserving_zeta_eval/process_by_T_band.csv`

### 3. Monofractal vs MRW Projection / Boundary

Purpose:
- report projection residuals, monofractal competition, MRW-vs-mono gain, and
  boundary behavior.

Entry:

```bash
python experiments/paper/run_exp3_boundary_projection.py --quick
```

Primary outputs:
- `outputs/tables/cmin_sr_v3_eval/process_by_T.csv`

### 4. Finite-Sample Curvature Identifiability

Purpose:
- test whether deterministic estimators can recover MRW `lambda2` from finite
  raw samples under the current q-grid and scale range.

Entry:

```bash
python experiments/paper/run_exp4_finite_sample_identifiability.py --quick
```

Primary outputs:
- `outputs/tables/finite_sample_identifiability/lambda2_recovery_by_T.csv`
- `outputs/tables/finite_sample_identifiability/lambda2_recovery_by_qgrid.csv`
- `outputs/tables/zeta_noise_bridge/separation_margin_vs_noise.csv`
- `outputs/tables/cmin_sr_failure_attribution/failure_attribution_summary_table.csv`

## Optional Experiment

### 5. Real-World Sanity Check

Purpose:
- run warning-safe exploratory validation if market data are available.
- This should not be treated as the central claim.

Entry:

```bash
python experiments/paper/run_exp5_real_world_sanity_check.py --quick
```

If data are missing, the script writes a warning summary and exits cleanly.

## Ablation / Appendix

- Tiny CMIN always-MRW bias.
- CMIN-Robust anti-confounding.
- CMIN-SR v1/v2/v3 progression.
- Boundary calibration same-H contrast failure.
- Raw zeta alignment trade-off.
- Curvature-preserving zeta alignment trade-off.
- Zeta noise bridge.
- Analytic vs deterministic vs neural failure attribution.

## Paper Asset Generation

```bash
python experiments/paper/generate_all_paper_assets.py --quick
```

Outputs:
- `paper_assets/figures/`
- `paper_assets/tables/`
- `paper_assets/summaries/`
- `paper_assets/latex/`
