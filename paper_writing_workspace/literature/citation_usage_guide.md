# Citation Usage Guide

This file records how each reference should be used in the CMIN-SR paper.

## MRW and Multifractal Process Background

| Cite key | Where to cite | Suggested use |
|---|---|---|
| `bacry2001multifractal` | Introduction, Framework | Original MRW formulation and the quadratic MRW scaling spectrum. Use near the equation \(\zeta(q)=qH-\frac{1}{2}\lambda^2 q(q-2)\). |
| `bacry2003log` | Framework / Related Work | Broader log-infinitely divisible multifractal process background. Useful if you discuss MRW-like families beyond the simplest two-parameter spectrum. |
| `halsey1986fractal` | Related Work / Framework | Classical multifractal formalism and singularity spectra. |
| `chhabra1989direct` | Related Work / Framework | Direct estimation of the \(f(\alpha)\) singularity spectrum. |
| `calvet2002multifractality` | Introduction / Related Work | Multifractality in asset returns and motivation from financial time series. |
| `mandelbrot1997multifractal` | Related Work | Early multifractal asset-return model background. |
| `calvet2001forecasting` | Related Work / Finance | Multifractal volatility forecasting; useful for finance motivation but not central validation. |

Suggested sentence:

> Multifractal random walk models encode intermittency through a nonlinear scaling spectrum, commonly written in the parabolic MRW form \(\zeta(q)=qH-\frac{1}{2}\lambda^2q(q-2)\) \cite{bacry2001multifractal}.

## Monofractal / fGn Baselines

| Cite key | Where to cite | Suggested use |
|---|---|---|
| `mandelbrot1968fractional` | Introduction / Experiments | Foundational reference for fractional Brownian motion and fractional Gaussian noise controls. |

Suggested sentence:

> Fractional Brownian motion and fractional Gaussian noise provide monofractal controls with stable scaling but linear spectra \cite{mandelbrot1968fractional}.

## Empirical Multifractal Estimation

| Cite key | Where to cite | Suggested use |
|---|---|---|
| `kantelhardt2002mfdfa` | Methods / Discussion | Canonical MFDFA method; cite when discussing alternatives to structure-function estimation. |
| `jaffard2007wavelet` | Discussion / Future Work | Wavelet leader multifractal formalism; cite as a robust future estimator direction. |
| `wendt2007bootstrap` | Methods / Discussion | Bootstrap uncertainty for empirical multifractal analysis. |
| `olivares2022mfdfa` | Discussion / Supplementary | Practical MFDFA implementation and modern reproducible tooling. |

Suggested sentence:

> More robust empirical spectrum estimators, including MFDFA \cite{kantelhardt2002mfdfa} and wavelet-leader methods \cite{jaffard2007wavelet}, are natural future directions.

## Physics-Informed / Constrained Scientific Machine Learning

| Cite key | Where to cite | Suggested use |
|---|---|---|
| `raissi2019pinn` | Introduction / Methods | General precedent for neural networks in constrained inverse problems. Do not imply CMIN-SR is a PDE PINN. |
| `karniadakis2021piml` | Introduction / Discussion | Broader scientific ML framing: combine data with mathematical/physical constraints. |

Suggested sentence:

> The role of the neural component is closer to constrained scientific machine learning \cite{raissi2019pinn,karniadakis2021piml} than to unconstrained black-box spectrum regression.

## Finite-Sample and Spurious Multifractality

| Cite key | Where to cite | Suggested use |
|---|---|---|
| `ludescher2011finite` | Failure Analysis / Discussion | Finite-size effects in multifractal detrended fluctuation analysis. |
| `grech2013spurious` | Introduction / Failure Analysis | Monofractal signals can generate apparent multifractal effects. |
| `bouchaud2000apparent` | Introduction / Failure Analysis | Apparent multifractality in financial time series; supports caution against overinterpreting raw spectra. |
| `zhou2009components` | Failure Analysis | Separates sources/components of empirical multifractality in returns. |
| `barunik2012source` | Failure Analysis / Finance | Source of multifractality in financial markets; useful for surrogate/shuffle discussion. |

Suggested sentence:

> Apparent multifractality can be induced by finite-size effects, monofractal finite-sample artifacts, and financial-return confounders \cite{ludescher2011finite,grech2013spurious,bouchaud2000apparent,barunik2012source}.

## Financial Controls and Negative Controls

| Cite key | Where to cite | Suggested use |
|---|---|---|
| `cont2001stylized` | Introduction / Experiments | Financial return stylized facts: heavy tails, volatility clustering, non-Gaussianity. |
| `bollerslev1986garch` | Experiments | GARCH negative-control / ambiguity baseline. |
| `hamilton1989regime` | Experiments | Regime-switching negative-control process. |
| `dimatteo2005long` | Related Work / Finance | Scaling analysis and generalized Hurst exponent in developed/emerging markets. |
| `buonocore2016measuring` | Methods / Finance | Multiscaling measurement in financial time series; useful when discussing empirical q-scaling. |

Suggested sentence:

> Heavy tails, volatility clustering, and regime changes are important confounders in financial time series \cite{cont2001stylized,bollerslev1986garch,hamilton1989regime}.

## Recommended Citation Placement by Section

### Introduction

Use:

```tex
\cite{bacry2001multifractal,mandelbrot1997multifractal,calvet2002multifractality,cont2001stylized}
```

When introducing finite-sample ambiguity:

```tex
\cite{bouchaud2000apparent,grech2013spurious,ludescher2011finite}
```

### Framework

Use:

```tex
\cite{bacry2001multifractal,mandelbrot1968fractional,halsey1986fractal,chhabra1989direct}
```

### Methods

Use:

```tex
\cite{kantelhardt2002mfdfa,jaffard2007wavelet,wendt2007bootstrap,buonocore2016measuring,raissi2019pinn,karniadakis2021piml}
```

### Experiments

Use:

```tex
\cite{bollerslev1986garch,hamilton1989regime}
```

### Discussion

Use:

```tex
\cite{kantelhardt2002mfdfa,jaffard2007wavelet,olivares2022mfdfa}
```
