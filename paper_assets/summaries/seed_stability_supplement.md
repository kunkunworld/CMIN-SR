# Multi-Seed Stability Supplement

Seeds: 2024, 2025, 2026

No new model was trained. Existing evaluation scripts were rerun under multiple random seeds and aggregated.

## Key output tables

- `paper_assets/tables/seed_stability_spectral_geometry_summary.csv`
- `paper_assets/tables/seed_stability_identifiability_summary.csv`
- `paper_assets/tables/seed_stability_zeta_noise_bridge_summary.csv`
- `paper_assets/tables/seed_stability_failure_attribution_summary.csv`

## Interpretation note

Use these tables to report mean/std over random seeds. The finite-sample identifiability result should still be framed cautiously because quick-mode sample sizes are intentionally small.