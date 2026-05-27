# Generator Alignment Audit

## Purpose

This audit checks whether apparent spectrum instability could come from:

- mismatched process generators,
- inconsistent normalization,
- different sequence lengths,
- or path / return representation inconsistencies

rather than from the process mechanisms themselves.

## Findings Before Refactor

Before introducing the shared generator module:

- `anti_confounded_dataset.py` used one internal MRW / Gaussian / GARCH implementation,
- `run_negative_controls.py` used another,
- normalization conventions were similar but not guaranteed identical,
- external negative controls included `fGn` while robust training data did not,
- robust training used `T=512`, while some external controls used `T=1024`,
- this made it hard to disentangle mechanism OOD from generator OOD.

## Current Refactor Action

New shared utilities:

- `src/mrw_inverse/data/process_generators.py`
- `src/mrw_inverse/data/normalization.py`

These now provide unified:

- MRW generation
- shuffled MRW generation
- fGn generation
- Gaussian generation
- Student-t generation
- GARCH generation
- regime-switching Gaussian generation
- standardization to zero-mean / unit-scale increments

## What Is Aligned Now

- `anti_confounded_dataset.py` now uses the shared generator entry
- `run_negative_controls.py` now uses the shared generator entry
- spectral-representation diagnostics use the same generator entry
- all these paths now use the same return/increment standardization

## Remaining Mismatches

Some mismatches are still scientifically real and should not be hidden:

1. Training process mix vs diagnostic process mix
   - robust training currently mixes MRW, shuffled MRW, Gaussian, Student-t, GARCH, regime-switching
   - some diagnostics also include `fGn`

2. Length distribution mismatch
   - older checkpoints were trained at `T=512` only
   - multi-length checkpoints train on `T in {512, 1024}`
   - external diagnostics may still test other lengths

3. Real-data mismatch
   - synthetic training uses standardized increments
   - real data can contain aggregation artifacts, structural breaks, missing-data effects, and market microstructure effects

## Interim Conclusion

The refactor reduces avoidable train/eval generator mismatch.

If future diagnostics still show:

- high MRW projection residual,
- unstable `p_scaling`,
- or inconsistent surrogate gaps,

these outcomes are more likely to reflect true process mismatch rather than accidental implementation drift.

