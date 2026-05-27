# Figure Style Unification Review

Date: 2026-05-26

## Main diagnosis

The two handmade figures carried useful scientific content, but their black
background, slide-like typography, and saturated colors did not match the rest
of the manuscript, which mostly uses white-background matplotlib figures. The
paper now uses white-background journal-style replacements for:

- `fig1_journal_cmin_sr_pipeline.pdf`
- `fig5_journal_finite_sample_identifiability.pdf`

The original handmade figures are retained in the figures directory as legacy
assets and are not deleted.

## Figure-by-figure status

### Fig. 1: CMIN-SR framework

Status: replaced in the manuscript.

The new figure keeps the three-stage logic:

1. finite-sample empirical spectrum estimation;
2. monofractal and MRW projection geometry;
3. calibrated diagnostics and instability warnings.

It also explicitly states that the MRW curvature coordinate is a projection
coordinate and not proof of mechanism.

### Fig. 5: finite-sample identifiability

Status: replaced in the manuscript.

The new figure uses a two-panel white-background line plot showing lambda2
correlation and lambda2 projection MAE across window lengths. This is more
consistent with the surrounding figures and better supports the finite-sample
identifiability argument.

### Fig. 8 / real-data sanity check

Status: readable but still the weakest publication figure.

Recommendation: keep as a sanity-check figure or move to appendix unless it is
redrawn with larger labels, fewer metrics per panel, and clearer distinction
between original and shuffled factor returns.

### Remaining figures

Most remaining figures are visually consistent enough for a draft. If more time
is available, the next useful redraw targets are:

1. real-data sanity check;
2. failure attribution;
3. projection residual geometry.

## LaTeX check

The manuscript was compiled after replacing the figures. No undefined
references, undefined citations, or overfull warnings were detected in
`main.log`.
