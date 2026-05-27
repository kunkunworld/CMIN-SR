# CMIN-SR Spectrum-Calibrated Comparison

## Analytic Space

The standalone spectral geometry calibrator succeeds on analytic spectra:

- `linear_mono`: low `p_curved`, low `p_MRW`, high `p_mono`
- `boundary_mrw`: high `p_boundary`, medium `p_MRW`
- `curved_mrw`: high `p_curved`, high `p_MRW`, low `p_mono`
- `heavy_tail_distorted` / `regime_apparent`: low `p_MRW`

The controlled lambda2 sweep is monotonic.

## Raw CMIN-SR Application

Applying the calibrator to raw CMIN-SR v3 outputs currently does not separate
`fGn` and MRW. The calibrated scores interpret raw `fGn` and Gaussian spectra as
too MRW-compatible, which means the bottleneck is upstream:

```text
raw x(t) -> zeta_emp(q)
```

The next stage should improve `zeta_emp` estimation through stronger spectrum
supervision, smoothing / bootstrap structure-function estimates, or analytic
zeta pretraining for the raw encoder.

The raw zeta alignment stage implements this direction. It improves fGn and
Gaussian false-positive behavior but currently collapses MRW curvature too much,
so it should be read as a partial success rather than the final main result.
