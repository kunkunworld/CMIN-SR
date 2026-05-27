# Top-Journal Writing Audit

## Critical Issues

1. **Figure 1 is not yet journal-quality.**
   - Risk: A reviewer may read the paper as an engineering workflow rather than a principled diagnostic framework.
   - Action: Redraw Figure 1 as a three-stage diagnostic pipeline with explicit conservative interpretation.

2. **Finite-sample lambda2 recovery must be framed as an identifiability result, not as model failure.**
   - Current status: The manuscript mostly does this correctly.
   - Action: Keep the language "finite-sample identifiability boundary", "estimator-level bottleneck", and "diagnostic framework".
   - Avoid: "failed to recover true MRW mechanism".

3. **Real-data sanity check should not be overclaimed.**
   - Risk: A single factor/sanity proxy cannot establish real-world MRW validity.
   - Action: Keep it as a warning-safe sanity check or appendix figure.

## Major Issues

1. **Table 3 was too broad for main text.**
   - Fix applied: compact main table plus full CSV retained.
   - Recommendation: Use Figure 5 as the main visual result and Table 3 as numerical support.

2. **Table 2 previously did not show the residual geometry it claimed to summarize.**
   - Fix applied: compact projection/residual table now includes lambda2 projection, residuals, gain, and instability.

3. **Historical CMIN-SR version progression should remain secondary.**
   - Risk: Too much version history makes the paper look like a development log.
   - Fix applied: compact Table 4 now reports only the most relevant p_MRW comparisons.
   - Recommendation: Full history belongs in appendix/repository documentation.

4. **Terminology around "calibrated probability" needs caution.**
   - Suggested wording: "calibrated diagnostic score" or "calibrated compatibility score".
   - Avoid unless statistically calibrated: "probability that the process is MRW."

## Minor Issues

- Prefer "generated MRW samples" over "true MRW" when discussing synthetic mechanisms.
- Prefer "checks" or "tests" over "validates" unless the evidence is comprehensive.
- Keep captions self-contained: each should state what is plotted and what conclusion is supported.
- In black-and-white print, scatter figures need marker shapes or line styles in addition to color.

## Suggested Direct Edits

### `sections/04_experiments.tex`
- Original: "It first validates the analytic spectral geometry..."
- Revised: "It first checks the analytic spectral geometry..."
- Reason: avoids overclaiming validation.

### `sections/05_results.tex`
- Original: "suppressed true MRW curvature"
- Revised: "suppressed curvature in generated MRW samples"
- Reason: avoids suggesting that lambda2 projection proves a real mechanism.

### `sections/05_results.tex`
- Original table input: `\input{latex_tables/table4_identifiability}`
- Revised table input: `\input{latex_tables/table3_finite_sample_lambda2_recovery}`
- Reason: uses compact main-text Table 3.

### `sections/05_results.tex`
- Original table input: `\input{latex_tables/table3_ablation}`
- Revised table input: `\input{latex_tables/table4_ablation}`
- Reason: uses compact version-comparison Table 4.

## Citation Audit

- Existing BibTeX keys for the heavy-tail/GARCH/regime-switching discussion are present in `references.bib`.
- If the PDF still shows `[?]`, rerun `pdflatex -> bibtex -> pdflatex -> pdflatex` from the LaTeX workspace.
- No new unverified references were added during this audit.

## Overall Recommendation

The manuscript is strongest when presented as a validity-aware spectral diagnostic framework with an explicit finite-sample identifiability limitation. It should not be presented as a guaranteed short-window lambda2 estimator or as proof of an MRW mechanism in empirical data.
