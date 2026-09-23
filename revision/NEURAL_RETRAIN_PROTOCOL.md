# Corrected-Data Neural Comparison

Public implementation protocol preserving the numerical settings of the corrected
comparison; this is not the original dated internal record.

## Scope and Training

Test the existing CMINSRZetaAlignedModel against overlapping OLS, its
nonoverlapping structure-function prior, MFDFA and quadratic MFDFA, and matched
Ridge controls. Canonical MRW has H=0.5. The analytic zeta curve is an asymptotic
reference, not an exact finite-discretization expectation. fGn spans H.

- Existing architecture, hidden_dim=128; no new backbone or diagnostic heads.
- Initialize the final zeta correction layer to zero. Initial zeta equals the
  nonoverlapping prior. Freeze validity/stability heads; do not interpret their
  untrained scores.
- Seeds 51001/51002/51003; 6000 training and 1200 validation sequences per seed,
  equally split between T=512 and 1024. Independent parameter/innovation streams.
- MRW low/medium/high lambda2 bands each contribute 20%; fGn H in [.1,.9]
  contributes 20%, fGn H=.5 10%, independent Gaussian 10%.
- Eight epochs, batch 64, AdamW lr=.0003, weight decay=.0001, gradient clip=1.
- Matched configurations: existing full loss with correction coefficient
  .125/.25/.5, plus 2*L_zeta alone at .25. Primary configuration is full_c025.
- Select checkpoints by validation mean zeta MAE, never by test scores.
- Ridge(alpha=.001) uses train-standardized concatenated OLS/prior spectra;
  the additional log-volatility control is specified separately.
- fGn uses the full covariance circulant embedding, including lag T at the
  midpoint, sqrt(2T) normalization, and strict rejection of non-PSD embeddings.

## Independent Evaluation

Test seeds 81001..81005; threshold-validation seeds 82001..82003; T=512/1024/2048.
MRW lambda2 levels are 0/.005/.01/.03/.06/.1/.15/.2. There are 20 MRW samples per
test seed/cell and 10 per validation seed/cell. Same-H fGn, other-H fGn and
Gaussian are reported separately. T=2048 tests length extrapolation.

Use a frozen analytic calibrator with RMSE geometry features and tail input zero
for all methods. Null thresholds use validation same-H fGn at an empirical FPR
no greater than 5%. No stress/noise examples train the encoder.

Report all configurations/seeds, signed and clipped lambda2, zeta/H/lambda2
errors, band behavior, and paired bootstrap intervals. Repeated training seeds
do not multiply independent test N. Three training seeds limit interval precision.

## Gates

Check covariance against independent Cholesky samples, split/stream disjointness,
finite outputs and gradients, zero-head prior equivalence, frozen head parameters,
strict checkpoint reload, and unchanged protected code. Save seeds, hashes,
generated inputs and histories. Refuse existing output directories. Quick runs
use seed 51999 and reduced sizes solely for software verification.
