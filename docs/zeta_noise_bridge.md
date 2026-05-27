# Zeta Noise Bridge

## Motivation

The analytic spectral geometry calibrator succeeds on clean spectra but fails
when fed noisy raw CMIN-SR spectra. The zeta noise bridge asks how much
q-space noise is needed to destroy MRW/fGn separability.

## Entry Point

```bash
conda run -n for_codex python experiments/run_zeta_noise_bridge.py --quick
```

Outputs:

- `outputs/reports/zeta_noise_bridge/`
- `outputs/tables/zeta_noise_bridge/`
- `outputs/figures/zeta_noise_bridge/`

## Procedure

Analytic spectra are generated for:

- `linear_mono`;
- `boundary_mrw`;
- `curved_mrw`.

Controlled perturbations are added:

- smooth q-space noise;
- high-q localized noise;
- jagged noise;
- high-q bending bias.

The pretrained `spectral_geometry_calibrator.pt` then evaluates the noisy
spectra.

## Current Result

At zero noise the calibrator behaves as expected:

- `linear_mono`: low `p_curved`, low `p_MRW`, high `p_mono`;
- `boundary_mrw`: high `p_boundary`, medium `p_MRW`;
- `curved_mrw`: high `p_curved`, high `p_MRW`, low `p_mono`.

As q-space noise grows, linear spectra become increasingly MRW-compatible.
In the quick run, `linear_mono p_MRW` rises from about `0.13` at zero noise to
about `0.48` at noise level `0.10`. Boundary spectra also lose clean boundary
behavior as noise increases.

## Interpretation

The bridge explains why raw CMIN-SR outputs can fool the calibrator: the
calibrator is not the weak link in clean spectrum space. It is sensitive to
the geometry it receives, and noisy finite-sample `zeta_emp` can create
apparent curvature that is large enough to change the interpretation.
