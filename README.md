# CMIN-SR

Finite-sample spectrum estimation and validity-aware spectral diagnostics for
multifractal random walk (MRW) inference.

The implementation separates estimating a scaling function from interpreting its
geometry. Neural finite-sample correction uses the existing spectrum encoder;
monofractal/MRW fits and a small geometry MLP provide complementary diagnostic
scores. Spectral compatibility is not identification of a generating mechanism.

## Corrected Release

**Use the entry points in `revision/` for corrected-data experiments.** Two issues
in the historical raw-signal evaluations were identified and corrected:

1. The Gaussian FFT sampler was missing the `sqrt(2*T)` normalization, reducing
   covariance by `2*T` and suppressing log-volatility fluctuations.
2. One structure-function baseline treated increments as paths. The corrected
   adapter requires an explicit input convention and integrates increments once.

The revised canonical MRW experiments at `H=0.5` show substantial curvature
recovery. The original attribution to a general finite-sample identifiability
barrier is therefore **not supported by the corrected experiments**. Estimation
accuracy and identification of an MRW mechanism remain different questions.

Historical `src/`, `experiments/`, and `scripts/` implementations remain available
for compatibility and paired comparisons. In particular, the original sampler
in `src/mrw_dl/generation.py` is retained intentionally; do not use it for new MRW
recovery experiments. Use `revision.generation.generate_pair` or
`revision.retraining.simulate` instead. Old raw-data training/evaluation scripts
are not the reproduction path for the corrected results.

## Method

```text
finite increments
  -> structure functions and learned spectrum correction
  -> empirical zeta(q)
  -> monofractal and MRW fits
  -> residual / curvature / boundary features
  -> geometry MLP diagnostic scores
```

`p_scaling`, `p_curved`, `p_mono`, `p_mrw`, and `p_boundary` are supervised
diagnostic scores for scaling, curvature, linear-spectrum compatibility,
MRW-spectrum compatibility, and boundary behavior. They are not verified
posterior probabilities of physical mechanisms. `lambda2_proj` is a spectral
projection coordinate; signed coefficients are also reported to expose clipping.

The corrected matched comparisons use training-consistent RMSE geometry features
and **tail input fixed to zero**, not process labels or an oracle tail target.
Stability and moment reliability cannot be established from one zeta curve alone.

## Installation

Python 3.10 or later is recommended. From the repository root:

```bash
python -m venv .venv
# Activate .venv with the command appropriate to your shell.
python -m pip install -r requirements.txt
```

Install a suitable PyTorch build for your CPU/CUDA environment. Commands below
use CPU for portability; full neural training is substantially faster on CUDA.
`requirements-tested.txt` records the directly tested versions (Python 3.10.14).
To constrain an installation to those versions, use
`python -m pip install -r requirements.txt -c requirements-tested.txt`.

## Quick Checks

These checks require neither private data nor a pretrained checkpoint:

```bash
python scripts/smoke_test_public_release.py
python revision/scripts/validate_generator_correction.py --output-dir revision/results/generator_check
python revision/scripts/train_corrected_neural_comparison.py --quick --device cpu --output-dir revision/results/neural_smoke
python revision/scripts/train_backend_ablation.py --quick --output-dir revision/results/backend_smoke
```

Output directories must be new: experiment scripts refuse to overwrite a run.
Quick runs test the pipeline, not the published numerical conclusions.

## Reproduction

The additional validation suite is documented in
[`revision2/experiments/README.md`](revision2/experiments/README.md). It includes
independent simulator checks, signed versus clipped curvature, a common-sample
estimator comparison, and nonwhite-fGn threshold tests. Its numerical scripts
retain the original experiment definitions; protocol metadata is sanitized for
public use. Generated results and checkpoints are not uploaded.

The command order, prerequisites, and fixed protocols are in
[`revision/README.md`](revision/README.md). Main entry points are:

| Experiment | Script in `revision/scripts/` |
| --- | --- |
| Generator/input correction and deterministic recovery | `run_corrected_recovery.py` |
| Scale, integral-scale and q-grid sensitivity | `run_scope_sensitivity.py` |
| Matched neural, prior, OLS and linear controls | `train_corrected_neural_comparison.py`, `evaluate_corrected_neural_comparison.py` |
| Paired estimation statistics and intervals | `summarize_neural_retraining.py` |
| MLP versus affine readouts and fixed geometry rules | `train_backend_ablation.py`, `evaluate_backend_ablation.py`, `summarize_backend_ablation.py` |
| Controlled spectral noise | `run_corrected_noise_bridge.py` |
| Stress-process controls | `evaluate_stress_controls.py`, `evaluate_retrained_stress.py` |
| Local-data surrogate sanity check | `run_real_surrogate_inference.py` |

Checkpoints, generated outputs and third-party data are not bundled. Training
scripts regenerate checkpoints; evaluations requiring a missing checkpoint stop
with an error, never silently substitute random weights. Reproducing exact
archived values requires matching checkpoint hashes and numerical environment,
not merely a successful quick run. Public protocol hashes differ from internal
historical protocol records.

## Interpretation and Scope

- Analytic MRW targets in the corrected confirmatory comparison apply to `H=0.5`.
  Other-H modulated processes are sensitivity controls, not asserted exact MRW
  scaling laws at arbitrary H or finite discretization.
- Neural correction improves spectrum estimation in the matched benchmark, but
  is not uniformly best for every coordinate or statistical baseline.
- The MLP/affine comparison tests fitting of five designed diagnostic targets
  and held-out transfer. It does not establish that an MLP is necessary for
  curvature ranking or that its outputs are mechanism probabilities.
- Low-curvature boundaries and out-of-distribution processes require separate
  scrutiny. Surrogate differences on real data are conditional diagnostics,
  not proof of multifractality or an MRW generating process.

## Repository Layout

```text
revision/             corrected simulation, matched experiments and protocols
revision2/experiments/ independent validation and threshold-transfer tests
src/mrw_inverse/      spectrum estimation, projections, models and losses
src/mrw_dl/           historical generation / neural modules
experiments/          original analytic calibration and historical experiments
scripts/              smoke tests, including the corrected public-release gate
```

This public package contains implementation and reproduction instructions only.
It excludes manuscript drafts, responses, private notes, raw data and generated
paper assets. The repository name is retained for continuity with existing links.
