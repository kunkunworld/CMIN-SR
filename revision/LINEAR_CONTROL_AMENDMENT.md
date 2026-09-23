# Matched Log-Volatility Control

The encoder observes log-volatility covariance in addition to zeta. The
`ridge_logvol` control therefore uses twelve OLS/prior spectral values plus seven
log-volatility covariances and their log-lag slope: twenty features.
StandardScaler and Ridge(alpha=.001) are fit on each training set only. No test
set selects features or regularization. Report this alongside the twelve-feature
Ridge baseline, not as a replacement.

This control does not match every graph/raw input. It cannot by itself attribute
all improvements specifically to the TCN architecture.
