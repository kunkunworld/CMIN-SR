# GitHub Upload Notes

This folder is a cleaned upload package for the CMIN-SR project.

## Suggested repository name

`CMIN-SR`

## Suggested repository description

Validity-aware spectral diagnostics for multifractal random walk inference under finite samples.

## What is included

- `src/`: core Python modules for empirical spectra, projections, calibrators,
  datasets, losses, and finite-sample identifiability analysis.
- `experiments/`: training, evaluation, ablation, identifiability, and paper
  asset generation scripts.
- `scripts/`: smoke tests and utility scripts.
- `docs/`: project documentation, final manifest, implementation guide, and
  experiment interpretation notes.
- `paper_assets/`: paper-ready figures, tables, summaries, and figure data.
- `paper_writing_workspace/`: LaTeX manuscript source, references, figures,
  and compiled `main.pdf`.

## What is intentionally excluded

- `checkpoints/`
- full `outputs/`
- raw `data/`
- Python caches and LaTeX build caches

These are excluded to keep the GitHub repository lightweight. The manuscript
and paper assets include the key tables and figures needed to understand the
reported results.

## Quick smoke checks

From the repository root:

```powershell
python scripts/smoke_test_curvature_identifiability.py
python scripts/smoke_test_classical_multifractal_baselines.py
python scripts/smoke_test_paper_pipeline.py
```

If using the local project environment:

```powershell
D:\anaconda\envs\for_codex\python.exe scripts\smoke_test_paper_pipeline.py
```

## Paper

The current paper draft is in:

`paper_writing_workspace/main.tex`

The compiled PDF is:

`paper_writing_workspace/main.pdf`
