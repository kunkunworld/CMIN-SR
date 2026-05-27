# CMIN-SR Version Comparison

## Comparison Target

The central comparison is:

```text
v2 reduces fGn p_MRW but also reduces MRW p_MRW.
v3 should reduce fGn / Gaussian p_MRW while restoring significant-MRW p_MRW.
```

This is why v3 adds `p_curved`. It removes curvature detection from the already
overloaded `p_MRW` variable.

## Versions

- `CMIN-SR v1`: first trainable spectral representation learner with
  `p_scaling`, MRW projection, and `p_MRW`.
- `CMIN-SR v2`: adds monofractal projection, `p_mono`,
  `mrw_vs_mono_gain`, and `curvature_significance`.
- `CMIN-SR v3`: adds explicit `p_curved`, curvature diagnostics, and
  boundary-MRW calibration.
- `CMIN-SR-Calibrated`: same v3 architecture, fine-tuned with same-H
  monofractal-vs-MRW boundary calibration.

## Process-Level Expectations

- Significant MRW: high `p_scaling`, high `p_curved`, high `p_MRW`, low
  `p_mono`, and MRW residual below monofractal residual.
- Low-`lambda2` MRW: high `p_scaling`, low-to-medium `p_curved`, medium
  `p_MRW`, medium `p_mono`, and high boundary score.
- `fGn`: high `p_scaling`, low `p_curved`, high `p_mono`, reduced `p_MRW`,
  and monofractal residual no worse than MRW residual.
- iid Gaussian: similar to `fGn`, with slightly less stable scaling.
- iid Student-t: low `p_MRW`; apparent curvature should be treated as
  tail-instability risk rather than MRW evidence.
- GARCH: cautious / ambiguous; moderate scaling or curvature should not be
  overclaimed as MRW.
- Regime-switching Gaussian: lower stability and low `p_MRW`.

## Outputs

The version comparison script reads existing v1/v2/v3/calibrated evaluation
tables and writes:

- `outputs/tables/cmin_sr_final_version_comparison/version_process_comparison.csv`
- `outputs/reports/cmin_sr_final_version_comparison/cmin_sr_final_version_comparison_summary.md`
- `outputs/figures/cmin_sr_final_version_comparison/pmrw_versions.png`

Run:

```bash
conda run -n for_codex python experiments/compare_cmin_sr_versions.py
```

## Interpretation

v3 can be considered the paper-facing calibration version if the main table
shows:

- `fGn p_scaling` remains high,
- `fGn p_curved` is low,
- `fGn p_MRW` is lower than v2 or clearly lower than significant MRW,
- significant MRW `p_curved` and `p_MRW` are restored,
- low-`lambda2` MRW shows boundary behavior rather than a forced binary label.
