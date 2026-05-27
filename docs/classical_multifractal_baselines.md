# Classical Multifractal Baselines

## Purpose

This supplement adds stronger classical estimator controls to the finite-sample
curvature recovery study. It does not train a new CMIN-SR model, add a new
validity head, or change the main diagnostic framework.

The goal is to answer:

```text
Do representative classical multifractal estimators remove the short-window
lambda2 recovery bottleneck?
```

## Implemented Estimators

The implementation lives in:

```text
src/mrw_inverse/analysis/classical_multifractal_estimators.py
```

The estimators are:

- `structure_aggregated_ols`
  - overlapping aggregate-increment structure functions;
  - aligns more directly with the paper's intended return/increment convention.
- `mfdfa`
  - first-order multifractal detrended fluctuation analysis;
  - reports zeta under the project convention `zeta(q)=q h(q)`.
- `mfdfa_quadratic`
  - second-order MFDFA.
- `wavelet_leader_haar`
  - compact dependency-free Haar wavelet-leader approximation.
- `wtmm_haar`
  - compact Haar WTMM-style modulus-maxima approximation.

Important limitation:

```text
The Haar wavelet rows are lightweight classical controls, not full production
WTMM or wavelet-leader packages.
```

This wording should be preserved in the manuscript.

## Experiment Entry Point

```bash
python experiments/run_classical_multifractal_baseline_comparison.py --quick
```

Smoke test:

```bash
python scripts/smoke_test_classical_multifractal_baselines.py
```

The experiment also integrates with:

```text
experiments/run_finite_sample_curvature_identifiability.py
```

Supported estimator groups:

```bash
python experiments/run_finite_sample_curvature_identifiability.py --quick --estimator-name classical
python experiments/run_finite_sample_curvature_identifiability.py --quick --estimator-name all
```

## Outputs

Main outputs:

```text
outputs/reports/classical_multifractal_baselines/
outputs/tables/classical_multifractal_baselines/
outputs/figures/classical_multifractal_baselines/
```

Paper assets are copied to:

```text
paper_assets/tables/
paper_assets/figures/
paper_assets/summaries/
paper_writing_workspace/paper_assets/tables/
paper_writing_workspace/paper_assets/figures/
paper_writing_workspace/paper_assets/summaries/
```

Key files:

- `classical_baseline_lambda2_recovery_by_T.csv`
- `classical_baseline_by_estimator.csv`
- `classical_baseline_sample_level.csv`
- `classical_multifractal_baseline_summary.md`
- `classical_lambda2_corr_vs_T.png/pdf`
- `classical_high_lambda_detection_vs_T.png/pdf`

## Quick-Run Result

In the quick grid with `T in {512, 1024, 2048}`, `H in {0.4,0.6}`,
`lambda2 in {0,0.03,0.10,0.20}`, and 10 samples per cell:

- first-order MFDFA has the lowest average lambda2 MAE among the added
  classical baselines, around `0.069`;
- MFDFA detects many high-lambda samples, but its lambda2 correlation remains
  weak and changes sign across T;
- aggregated structure functions and Haar wavelet-leader controls show near-zero
  or unstable correlation;
- the compact Haar WTMM-style baseline degenerates to a low-curvature estimate
  on this grid, so its correlation is undefined.

Interpretation:

```text
The stronger classical baselines improve the benchmark coverage, but they do
not remove the short-window lambda2 recovery limitation under the tested grid.
```

## Manuscript Use

This supplement supports a cautious statement:

```text
Representative classical baselines, including MFDFA and lightweight Haar
wavelet controls, do not eliminate the finite-sample curvature recovery
bottleneck in the tested short-window setting.
```

Do not claim:

```text
CMIN-SR outperforms all classical multifractal estimators.
```

Do not claim:

```text
The implemented Haar WTMM-style baseline is a full WTMM package.
```

The remaining future-work statement should be:

```text
A full production WTMM or wavelet-leader benchmark would still be valuable,
but the present results already include lightweight wavelet-style controls.
```
