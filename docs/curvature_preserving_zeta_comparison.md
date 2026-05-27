# Curvature-Preserving Zeta Comparison

## Compared Stages

- CMIN-SR + spectrum-space calibrator before zeta alignment
- raw zeta aligned
- curvature-preserving zeta aligned

## Summary

The comparison shows a trade-off:

- before zeta alignment, MRW scores were high but fGn/Gaussian false positives
  were also high;
- raw zeta alignment reduced fGn/Gaussian false positives but flattened MRW;
- curvature-preserving alignment can move either side depending on weights, but
  the final conservative run still under-recovers MRW curvature.

At `T=1024`, final conservative curvature-preserving alignment gives:

- fGn: low-to-medium `p_MRW_cal`, high `p_mono_cal`;
- Gaussian: low `p_MRW_cal`, high `p_mono_cal`;
- MRW medium/high: still low-to-medium `p_MRW_cal`, not recovered enough;
- Student-t / Regime: remain low or cautionary.

This should be reported as an ablation and limitation, not as the final closed
main result.
