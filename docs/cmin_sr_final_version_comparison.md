# CMIN-SR Final Version Comparison

## Compared Models

The current comparison includes:

- `CMIN-SR v1`
- `CMIN-SR v2`
- `CMIN-SR v3`
- `CMIN-SR-Calibrated`

The calibrated model is not a v4. It is an ablation that keeps the v3
architecture and adds same-H boundary calibration fine-tuning.

## Desired Pattern

The desired paper-table result is:

- `fGn p_scaling` remains high;
- `fGn p_curved` decreases;
- `fGn p_MRW` decreases;
- significant MRW `p_curved` increases;
- significant MRW `p_MRW` recovers;
- low-`lambda2` MRW shows medium `p_MRW` and high boundary score;
- Student-t and regime-switching controls remain low `p_MRW`;
- GARCH remains cautious / ambiguous.

## Current Reading

The calibrated run is a partial diagnostic success, not a final main-table
success. It keeps Gaussian, Student-t, and regime-switching controls under
control, but `fGn` and significant MRW remain too close.

At `T=1024`, calibrated scores remain approximately:

- `fGn`: high `p_scaling`, `p_curved` around 0.53, `p_MRW` around 0.64;
- MRW: high `p_scaling`, `p_curved` around 0.52, `p_MRW` around 0.63;
- Gaussian: low `p_curved`, low-to-medium `p_MRW`, high `p_mono`;
- Student-t / RegimeSwitch: low `p_MRW` with instability warnings.

The same-H boundary sweep also remains non-monotonic for `p_curved`.

## Spectrum-Space Follow-up

Spectrum-space calibration shows that the desired geometry is learnable on
analytic `zeta(q)` curves. The remaining problem is applying that interpretation
to raw CMIN-SR outputs: raw `zeta_emp` still makes fGn/Gaussian look too close
to boundary/noisy-MRW geometry.

Raw zeta alignment reduces that false monofractal curvature, but the first
aligned model also suppresses MRW curvature. The result is an important
ablation, not yet a final main-table version.

## Output Paths

Run:

```bash
conda run -n for_codex python experiments/compare_cmin_sr_versions.py
```

Outputs:

- `outputs/tables/cmin_sr_final_version_comparison/version_process_comparison.csv`
- `outputs/reports/cmin_sr_final_version_comparison/cmin_sr_final_version_comparison_summary.md`
- `outputs/figures/cmin_sr_final_version_comparison/pmrw_versions.png`
- `outputs/figures/cmin_sr_final_version_comparison/fgn_mrw_pmrw_versions.png`
