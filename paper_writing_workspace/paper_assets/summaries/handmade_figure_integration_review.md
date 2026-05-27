# Handmade Figure Integration Review

Two handmade figures were added from:

```text
D:/graduate_first/research/MS/paper_assets/handmade_figure/
```

They were copied into:

```text
D:/graduate_first/research/MS/paper_writing_workspace/figures/
```

## Integrated Figures

### `overall.png`

Copied as:

```text
figures/fig1_handmade_cmin_sr_pipeline.png
```

Used in:

```text
sections/02_framework.tex
```

Role:

- replaces the old temporary Fig.1 pipeline;
- much better communicates the three-stage CMIN-SR story:
  1. finite-sample spectrum;
  2. competing projections;
  3. validity diagnostics.

Assessment:

- This is a clear improvement over the old figure.
- The conservative interpretation box is good and supports the paper's core claim.

Minor improvement suggestions:

- Change `lambda²proj` to `lambda²_proj` or `\lambda^2_{\mathrm{proj}}`.
- Consider replacing `zeta-hat(q)` with `\hat{\zeta}(q)` if redrawing in vector software.
- The black background/title works visually, but for journal print a white or transparent background may be safer.

### `fig3.png`

Copied as:

```text
figures/fig5_handmade_finite_sample_identifiability.png
```

Used in:

```text
sections/05_results.tex
```

Role:

- replaces the old finite-sample identifiability figure;
- supports the central negative/diagnostic result:
  short-window projected `lambda2` remains weakly and inconsistently correlated
  with the known synthetic MRW curvature coordinate.

Assessment:

- This is visually stronger than the previous matplotlib figure.
- It is appropriate for the current finite-sample identifiability subsection.

Important limitation:

- The figure shows deterministic structure-function estimator variants
  (OLS, bootstrap, smoothed, trimmed), not the newly added MFDFA/Haar
  wavelet-style classical baselines.
- The caption was written to reflect this scope.

Minor improvement suggestions:

- Replace `true lambda2` with `known synthetic \lambda^2` or `known simulation \lambda^2`.
- The legend includes Bootstrap, but the orange series is hard to see or overlaps other curves; separate it visually if possible.
- Panel B has a very narrow y-axis range, which is acceptable for showing flatness but may exaggerate small differences. A note or wider range may be safer.
- If this figure becomes a main paper figure, consider adding a third panel or companion figure for the new MFDFA / Haar wavelet baselines.

## LaTeX Changes

Updated:

```text
sections/02_framework.tex
sections/05_results.tex
```

No experimental values were changed.
