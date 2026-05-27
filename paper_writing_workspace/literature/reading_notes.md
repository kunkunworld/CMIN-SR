# Reading Notes

## How to Use This Literature Set

The bibliography is organized to support the paper's conservative framing:

1. MRW-like spectra are theoretically meaningful.
2. Monofractal controls can show stable scaling without multifractality.
3. Empirical multifractal estimation is fragile under finite samples.
4. Financial time series contain confounders such as heavy tails, volatility clustering, and regime changes.
5. Therefore CMIN-SR should be framed as a validity-aware diagnostic framework.

## Core Reading Order

1. `bacry2001multifractal`  
   Read for the MRW spectrum and model motivation.

2. `mandelbrot1968fractional`  
   Read for fBm/fGn monofractal baselines.

3. `kantelhardt2002mfdfa`  
   Read for empirical multifractal estimation and why alternatives to simple structure functions matter.

4. `grech2013spurious` and `ludescher2011finite`  
   Read for why apparent multifractality can arise from finite-size or monofractal artifacts.

5. `cont2001stylized`, `bollerslev1986garch`, `hamilton1989regime`  
   Read for negative controls and financial confounders.

6. `bouchaud2000apparent`, `zhou2009components`, `barunik2012source`  
   Read for apparent multifractality, surrogate/shuffle logic, and sources of empirical multifractality in financial returns.

7. `halsey1986fractal`, `chhabra1989direct`  
   Read only if you need a short classical paragraph about multifractal singularity spectra and \(f(\alpha)\).

## Paper-Specific Framing

The strongest literature-backed paragraph is:

> MRW models provide a theoretically motivated parabolic scaling spectrum, but empirical multifractal estimation from finite samples is known to be sensitive to finite-size effects, moment instability, and monofractal artifacts. CMIN-SR therefore treats MRW inference as a calibrated diagnostic problem rather than a direct parameter recovery problem.

Useful citations:

```tex
\cite{bacry2001multifractal,kantelhardt2002mfdfa,bouchaud2000apparent,ludescher2011finite,grech2013spurious}
```

## Current Reference Count

The curated bibliography currently contains 22 entries, which is a good target
for a focused methods/diagnostics paper.

Two additional scientific machine-learning references have been added for the
Introduction/Methods framing:

- `raissi2019pinn`
- `karniadakis2021piml`

Use them only to justify constrained scientific ML / inverse-problem framing,
not to claim that CMIN-SR is a PDE-based PINN.
