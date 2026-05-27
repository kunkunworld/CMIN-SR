# Estimator Comparison

## Three Estimators

The project now has four conceptually distinct estimators:

1. `proxy`
2. `cmin_tiny`
3. `cmin_robust`
4. `cmin_robust_multilength`

## Interpretation Roles

### Proxy

- transparent statistical baseline
- useful for sanity checks
- easy to understand
- still confounded by heavy tails and some nonstationarity

### Tiny CMIN

- first trainable inverse model checkpoint
- good enough to prove the model can learn synthetic MRW recovery
- but trained only on positive MRW samples
- develops an `always-explain-as-MRW` bias

### Robust CMIN

- same inverse architecture
- anti-confounded mixed-process training
- should be judged by false-positive reduction and meaningful `p_MRW`, not only by MRW MAE

### Multi-length Robust CMIN

- extends robust training across `T=512` and `T=1024`
- targets length OOD directly
- should be judged separately on:
  - internal by-length evaluation
  - external negative controls

## Updated Comparison Principle

Under the spectral-representation framing, estimator comparison is no longer only about driving non-MRW `lambda2` to zero.

It is also about:

- keeping empirical spectrum estimation stable,
- separating `p_scaling` from `p_MRW`,
- making MRW projection residual interpretable,
- and avoiding the collapse of all non-MRW processes into a trivial null label.
  - factor conservatism
  - eventual raw-market transfer

## Current Scientific Takeaway

The main lesson is now sharper:

- architecture was not the key bottleneck,
- training distribution and supervision were the key bottleneck,
- explicit negative controls in training materially improve selectivity,
- but stronger OOD robustness and raw-market transfer are still open questions.
