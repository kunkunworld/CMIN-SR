# Matched Geometry-Readout Comparison

Public specification of the existing experiment, not a new architecture.

## Inputs and Comparators

Freeze the upstream spectrum. Compare the existing two-hidden-layer, 128-wide
SpectralGeometryCalibrator with sigmoid-affine output on standardized features,
sigmoid-affine output after per-row LayerNorm, and fixed geometric ordering rules.
Learned methods use the same 26 features and original loss. Tail input is zero,
never a class-derived oracle. The legacy pretrained MLP is a descriptive reference,
not a matched contender.

The rules are the signed quadratic coefficient and the existing product of
curvature score, positive normalized gain and .1/(.1+MRW_RMSE). Rules are ranking
statistics, not five-output regressors or posterior probabilities.

## Training

Seeds 121001/121002/121003; independently generated 20000 analytic training and
4000 validation spectra per seed from the existing eight families. Thirty epochs,
batch 128, AdamW lr=.001 and weight decay=.0001. Identical minibatch schedules
across variants within a seed. Two unused outputs preserve the original module;
evaluate only the five supervised scores.

Select each checkpoint by validation five-target MSE. Select the primary affine
comparator between the two variants using validation MSE only; report both.
Equal budgets are not proof that either model class is globally optimized.

## Holdouts and Inference

- Analytic: seeds 123001..123005, 1600 spectra each. Main discrimination is curved
  MRW versus linear/noisy mono; retain other families separately.
- Raw: test seeds 124001..124005, null-validation seeds 125001..125003,
  T=512/1024/2048. H=.5 MRW, eight lambda2 levels, same-H/other-H fGn and Gaussian;
  1400 test and 420 validation sequences per T.
- Freeze corrected full_c025 frontend seed51001; also use OLS on identical
  sequences. The comparison is conditional on that frontend.
- Stress: seeds 126001..126005, T=1024, 50/family/seed: Student-t, GARCH, iid
  Gaussian variance mixture and shuffled corrected MRW. No clean zeta truth is
  assigned. Report compatibility decisions, not a universal FPR.
- Unseen algebraic curves: seeds 127001..127003, 100/family of quartic, kink,
  ripple and convex shapes. No stochastic realizability is asserted.

Use strict score > threshold with tie handling. Null thresholds use validation
scores only at empirical FPR <=5%. T1024 thresholds transfer unchanged to stress
data. Soft-target MSE measures agreement with designed labels, not physical
validity. One zeta curve does not establish scaling stability or moment reliability.

Report paired MLP-minus-validation-selected-affine differences with 2000 resamples
of data and training seeds. Stratify by evaluation seed/family or process/H/lambda.
Three seeds limit interval precision; hold frontend and validation thresholds
fixed and make no multiplicity guarantee. Save hashes, manifests, features,
predictions, split and strict reload checks. Quick runs verify execution only.
