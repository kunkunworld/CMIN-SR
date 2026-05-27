# Reviewer and Hallucination Audit Update, 2026-05-21

This note records the internal audit used while expanding the current CSF-style
draft.

## Hallucination-audit outcome

- Core framing is safe: CMIN-SR is a validity-aware spectral diagnostic
  framework, not a proof of an MRW mechanism.
- `lambda2_proj` must remain a projection coordinate.
- Analytic spectrum-space separation is strongly supported by
  `outputs/tables/spectral_geometry_calibrator_eval/summary_by_spectrum_type.csv`
  and the three-seed supplement.
- Finite-sample `lambda2` recovery remains weak in the tested short-window
  grid; this should be written as a limitation/identifiability result, not as a
  model failure hidden from the reader.
- Zeta-noise bridge numbers should be reported consistently. The three-seed
  supplement gives an MRW-linear `p_MRW` gap from about 0.804 at zero noise to
  about 0.365 at noise level 0.1. A single refreshed run gives about
  0.805 to 0.338.
- Raw zeta alignment numbers such as 0.73 to 0.36 refer to the documented
  T=1024 comparison and should not be generalized to all T.
- Fama-French results are exploratory sanity checks only.

## CSF-reviewer guidance integrated into the draft

- The method section now defines the finite-sample structure function and the
  MRW simulation procedure.
- The draft now includes the analytic `zeta(q)`, `alpha(q)`, and `f(alpha)`
  figure as analytic geometry, not empirical reconstruction accuracy.
- The experiments section now acknowledges MFDFA evidence while keeping
  WTMM/wavelet-leader as a limitation.
- Results now include the three-seed stability supplement and Fama-French
  sanity check with cautious wording.
- Discussion now explicitly warns against interpreting `p_MRW` as a mechanism
  probability or `lambda2_proj` as a guaranteed true parameter estimate.

## Remaining risks

- WTMM/wavelet-leader baselines remain absent.
- Current seed aggregation is a lightweight supplement, not a large full-grid
  statistical study.
- Tables generated earlier are compact and may need formatting polish for final
  submission.
