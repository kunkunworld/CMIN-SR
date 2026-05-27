# Fig.1 Redesign Prompt

Core message: CMIN-SR is a three-stage validity-aware diagnostic pipeline, not an MRW mechanism detector.

Recommended layout: wide three-stage horizontal graphic.

Stage 1: `Finite-sample spectrum estimation`
- Raw finite increments
- Structure-function / empirical zeta(q)
- Note: finite T, q-grid, scale range

Stage 2: `Competing spectral projections`
- Monofractal linear projection
- MRW-family parabolic projection
- Residual geometry and lambda2_proj

Stage 3: `Validity-aware diagnostics`
- p_scaling, p_curved, p_mono, p_MRW, p_boundary
- Tail / instability warning
- Final note: organizes evidence; does not prove MRW mechanism

Color guidance: neutral gray for data, blue for empirical spectrum, green for monofractal, orange for MRW, muted red for warnings. Avoid gradients and decorative icons.

PowerPoint/Figma: use aligned rounded rectangles with thin strokes, grouped by stage bands, and one small warning callout at the end.
Python matplotlib: use `FancyBboxPatch`, arrows, and stage background rectangles.
TikZ: use three grouped nodes with parallel projection branches; keep text short.

Caption: CMIN-SR converts finite increments into empirical scaling spectra, compares monofractal and MRW-family projections, and reports validity-aware diagnostic scores with instability warnings. The output is an evidence organization tool rather than proof of an MRW data-generating mechanism.