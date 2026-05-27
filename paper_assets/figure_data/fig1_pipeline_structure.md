# Fig.1 Pipeline Structure

## Nodes

| id | label | function |
|---|---|---|
| A | Raw finite increments | Finite observed returns/increments; not assumed to reveal a mechanism directly. |
| B | Empirical zeta estimation | Estimate structure-function slopes on a finite q-grid and scale set. |
| C | Monofractal projection | Fit linear spectrum zeta(q)=qH and compute residual. |
| D | MRW projection | Fit parabolic MRW-family spectrum and compute lambda2_proj and residual. |
| E | Residual and geometry features | Compare MRW-vs-mono residuals, curvature, linearity, boundary, and instability. |
| F | Calibrated diagnostic scores | Report p_scaling, p_curved, p_mono, p_MRW, p_boundary as diagnostic scores. |
| G | Conservative interpretation | Organize evidence; do not claim proof of an MRW data-generating mechanism. |

## Arrows

A -> B -> {C,D} -> E -> F -> G.

## Recommended layout

Use a three-stage horizontal pipeline with subtle vertical grouping:
Stage 1: finite increments to empirical zeta estimation.
Stage 2: parallel monofractal and MRW projections feeding residual geometry.
Stage 3: calibrated diagnostics feeding a warning/interpretable decision box.

## Caption suggestion

CMIN-SR diagnostic pipeline. Finite increments are first converted into an empirical scaling spectrum, then projected onto competing monofractal and MRW spectral families. Projection residuals, curvature, boundary behavior, and instability features are converted into diagnostic scores. The output organizes MRW-compatible spectral evidence but does not prove an MRW data-generating mechanism.