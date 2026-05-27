# Citation Hallucination Audit

Date: 2026-05-26

This audit records the independent citation check for the references added in
the May 26 citation update. The purpose is to avoid invented references,
incorrect titles, and unsupported claims.

## Agent H: Bibliographic Verifier

Checked whether each added work has stable bibliographic metadata and a DOI or
publisher page.

| BibTeX key | Verified source type | Status |
|---|---|---|
| `muzy1991wavelets` | APS DOI page | Verified |
| `muzy1993multifractal` | APS DOI page | Verified |
| `arneodo1995thermodynamics` | Elsevier DOI page | Verified |
| `mallat1992singularity` | IEEE DOI page | Verified |
| `bacry2010mixed` | Project Euclid / DOI metadata | Verified |
| `arridge2019inverse` | Cambridge University Press DOI page | Verified |
| `brunton2020mlfluid` | Annual Reviews DOI page | Verified |
| `cranmer2020sbi` | PNAS DOI page / PubMed metadata | Verified |

## Agent M: Mathematical Relevance Checker

The added references are mathematically aligned with the manuscript:

- `muzy1991wavelets`, `muzy1993multifractal`, and
  `arneodo1995thermodynamics` support the WTMM and wavelet multifractal
  formalism background.
- `mallat1992singularity` supports the general wavelet singularity-detection
  basis behind wavelet-based multifractal estimators.
- `bacry2010mixed` supports the statement that multifractal estimation depends
  on asymptotic regime and scale support.
- `arridge2019inverse` supports the framing of finite-sample parameter recovery
  as a data-driven inverse problem.
- `cranmer2020sbi` supports the use of simulator-generated data for amortized
  or simulation-based inference, without claiming exact posterior inference in
  this paper.
- `brunton2020mlfluid` supports the broader scientific-machine-learning
  framing that neural networks should be used with physical/model structure.

No added reference is used to claim that the present CMIN-SR experiments
outperform a production WTMM or wavelet-leader package.

## Agent E: Hallucination Auditor

High-risk wording checked:

- The manuscript still describes Haar wavelet-leader and WTMM-style rows as
  compact controls, not full production implementations.
- The manuscript does not claim true MRW mechanism validation on Fama-French
  data.
- The manuscript does not claim guaranteed short-window lambda2 recovery.
- The added deep-learning references are used only for inverse-problem and
  scientific-machine-learning framing. They are not used to claim that CMIN-SR
  is a neural operator, a full Bayesian simulation-based inference method, or a
  fluid-mechanics model.

Conclusion: the added citations are real, relevant, and used conservatively.
