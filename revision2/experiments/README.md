# Validation and Matched Comparisons

This suite checks the corrected simulator and separates spectrum estimation,
signed/clipped curvature, and thresholded diagnostic decisions. Run commands
from the repository root. No manuscript files, raw data, checkpoints or generated
results are included. The numerical scripts and vendored reference match the
experiment versions; only descriptive metadata in `protocol.json` was adapted
for public use, so its hash differs from the internal experiment record.

## Independent Simulator Check

This check needs no trained model or private input. Install the root requirements
first. The exact independent implementation is vendored from
[`neurospin/pymultifracs`](https://github.com/neurospin/pymultifracs) at commit
`ad254f31500a71374eb28c85a0905d9fa2ae2c28`, with its MIT license and SHA-256
provenance. No download is needed to run it.

```bash
python revision2/experiments/smoke_test.py
python revision2/experiments/validate_simulator.py --quick --output revision2/experiments/results/E1_smoke
python revision2/experiments/validate_simulator.py --output revision2/experiments/results/E1_formal
```

The full run checks 108 covariance cases/lags, uses an independent Cholesky
reference and the pinned public FFT implementation, and compares long-series
scaling estimates with the analytic reference. MRW spectral targets are checked
at H=0.5; the other H values are pure-fGn controls. Finite-scale discrepancies are
retained, not treated as exact equality with asymptotic exponents. The older
normalization/input audit remains in
`revision/scripts/validate_generator_correction.py`.

## Neural Evaluation Prerequisites

Follow steps 2 and 4 of [`revision/README.md`](../../revision/README.md), including
the full evaluation and summary commands. The expected generated paths are:

- `checkpoints/cmin/spectral_geometry_calibrator.pt` (or the configured geometry checkpoint).
- `revision/results/neural_retrain_formal_attempt3/` with `training_runs.csv` and its checkpoints.
- `revision/results/neural_eval_formal/` with independent test sequences, manifests and predictions.
- `revision/results/neural_summary_formal/heldout_roc_thresholds.csv`.

For this validation suite, use `--device cuda` in both the training and evaluation
commands in step 4: the replay scripts intentionally retain the CUDA device and
64-sample batches used by the experiment. Checkpoint hashes are checked against
the generated training manifest. No random-weight fallback is used. Exact
archived replay additionally requires the same checkpoints and numerical
environment; independently retrained models need not produce identical values.

## Common Samples and Signed Curvature

```bash
python revision2/experiments/compare_estimators.py --output revision2/experiments/results/E2_E3_formal
python revision2/experiments/nonwhite_fgn.py --output revision2/experiments/results/E4_formal
python revision2/experiments/audit_results.py --output revision2/experiments/results/verification
```

The common-sample comparison evaluates nine estimators on the same stored test
sequences. It reports spectrum MAE, signed and clipped curvature error, bias,
correlation and bootstrap intervals. Haar controls are approximations, not full
wavelet-leader or WTMM implementations. A training-seed prediction is not a new
independent sequence. The nonwhite-fGn check retains the original frozen
thresholds, including poor threshold transfer when it occurs.

`audit_results.py` reads the three fixed formal folders above; it does not
automatically switch to custom output names. It verifies input hashes, numerical
identities, operating counts and neural prediction replay. Do not edit protected
source files between generating results and running this audit.

## Independent Null Calibration and Threshold Transfer

The subsequent threshold adjustment is specified in
`THRESHOLD_CORRECTION_PROTOCOL.json`. It uses a new fGn-only calibration sample,
freezes thresholds, and then generates an independent test sample. The original
thresholds remain as paired controls; no model is retrained. The protocol records
that this adjustment followed the adverse original-threshold transfer result.

```bash
python revision2/experiments/correct_thresholds.py calibrate --output revision2/experiments/results/E5_calibration_formal
python revision2/experiments/correct_thresholds.py test --calibration-dir revision2/experiments/results/E5_calibration_formal --output revision2/experiments/results/E5_test_formal
python revision2/experiments/verify_threshold_correction.py
```

The verification command uses these fixed formal paths and creates
`results/E5_verification`. It independently recomputes quantiles and counts,
checks sample separation, and replays neural predictions. Exact-replay
tolerances were set for the original CUDA environment; a failure on another
backend must be investigated, not bypassed by silently relaxing the tolerance.

The scripts refuse to overwrite output directories. For another run, preserve
the previous run and use a fresh working copy or consistently update the input
locations of fixed-path aggregate helpers. Do not pool smoke and formal runs.
Optional `plot_results.py` reads the E2/E3 and E4 formal tables and creates a new
`figures/` directory; it plots original-threshold transfer, not the E5 adjustment.

## Metrics and Interpretation

- `lambda2_signed`: unconstrained two-coordinate least-squares curvature.
- `lambda2_proj`: nonnegative coordinate with the retained 1e-6 numerical floor.
- `spectrum_mae`: mean absolute error over the six positive q values.
- `signed_bias`: signed estimate minus true curvature, averaged over sequences.
- `signed_mae` / `clipped_mae`: coordinate MAE before / after the floor.
- `signed_corr` / `clipped_corr`: Pearson correlation with true curvature.
- `p_mrw`: supervised diagnostic score, not a mechanism probability or test p-value.
- `FPR` / `TPR`: thresholded fractions on the stated null and alternative groups.

Undefined correlations for constant targets remain missing, rather than being
replaced by zero. Threshold adjustment leaves spectra, scores and AUROC
unchanged. Improvements against specified Gaussian controls do not establish
rejection of heavy-tailed or heteroskedastic alternatives. Signed boundary bias,
weak-curvature sensitivity and stress-process acceptance remain separate results.
