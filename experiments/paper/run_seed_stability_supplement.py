from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_TABLE = ROOT / "paper_assets" / "tables"
OUT_SUMMARY = ROOT / "paper_assets" / "summaries"


def _run(script: str, args: list[str]) -> None:
    cmd = [sys.executable, str(ROOT / script), *args]
    completed = subprocess.run(cmd, cwd=ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def _read_csv(rel: str) -> pd.DataFrame:
    path = ROOT / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _summarize_numeric(df: pd.DataFrame, group_cols: list[str], source: str, seed: int) -> pd.DataFrame:
    numeric = [c for c in df.columns if c not in group_cols and pd.api.types.is_numeric_dtype(df[c])]
    out = df[group_cols + numeric].copy()
    out["seed"] = seed
    out["source_table"] = source
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small multi-seed stability supplement for paper evidence.")
    parser.add_argument("--seeds", default="2024,2025,2026")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--num-samples-ident", type=int, default=8)
    parser.add_argument("--num-samples-spectral", type=int, default=600)
    parser.add_argument("--num-samples-noise", type=int, default=30)
    parser.add_argument("--num-samples-attribution", type=int, default=16)
    args = parser.parse_args()

    OUT_TABLE.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    spectral_rows: list[pd.DataFrame] = []
    ident_rows: list[pd.DataFrame] = []
    noise_rows: list[pd.DataFrame] = []
    attribution_rows: list[pd.DataFrame] = []

    for seed in seeds:
        _run(
            "experiments/evaluate_spectral_geometry_calibrator.py",
            ["--num-samples", str(args.num_samples_spectral), "--seed", str(seed)],
        )
        sg = _read_csv("outputs/tables/spectral_geometry_calibrator_eval/summary_by_spectrum_type.csv")
        spectral_rows.append(_summarize_numeric(sg, ["spectrum_type"], "spectral_geometry_summary", seed))

        ident_args = ["--num-samples", str(args.num_samples_ident), "--seed", str(seed)]
        if args.quick:
            ident_args.insert(0, "--quick")
        _run("experiments/run_finite_sample_curvature_identifiability.py", ident_args)
        ident = _read_csv("outputs/tables/finite_sample_identifiability/lambda2_recovery_by_T.csv")
        ident_rows.append(_summarize_numeric(ident, ["T", "estimator"], "lambda2_recovery_by_T", seed))

        noise_args = ["--num-samples", str(args.num_samples_noise), "--seed", str(seed)]
        if args.quick:
            noise_args.insert(0, "--quick")
        _run("experiments/run_zeta_noise_bridge.py", noise_args)
        noise = _read_csv("outputs/tables/zeta_noise_bridge/separation_margin_vs_noise.csv")
        noise = noise.rename(columns={noise.columns[0]: "noise_level", noise.columns[1]: "pmrw_gap_curved_minus_linear"})
        noise["seed"] = seed
        noise["source_table"] = "separation_margin_vs_noise"
        noise_rows.append(noise)

        attr_args = ["--num-samples", str(args.num_samples_attribution), "--seed", str(seed)]
        if args.quick:
            attr_args.insert(0, "--quick")
        _run("experiments/run_cmin_sr_failure_attribution.py", attr_args)
        attr = _read_csv("outputs/tables/cmin_sr_failure_attribution/failure_attribution_summary_table.csv")
        attribution_rows.append(_summarize_numeric(attr, ["process_type", "level"], "failure_attribution_summary", seed))

    spectral_all = pd.concat(spectral_rows, ignore_index=True)
    ident_all = pd.concat(ident_rows, ignore_index=True)
    noise_all = pd.concat(noise_rows, ignore_index=True)
    attribution_all = pd.concat(attribution_rows, ignore_index=True)

    spectral_all.to_csv(OUT_TABLE / "seed_stability_spectral_geometry_by_seed.csv", index=False)
    ident_all.to_csv(OUT_TABLE / "seed_stability_identifiability_by_seed.csv", index=False)
    noise_all.to_csv(OUT_TABLE / "seed_stability_zeta_noise_bridge_by_seed.csv", index=False)
    attribution_all.to_csv(OUT_TABLE / "seed_stability_failure_attribution_by_seed.csv", index=False)

    def agg(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        numeric = [c for c in df.columns if c not in group_cols + ["seed", "source_table"] and pd.api.types.is_numeric_dtype(df[c])]
        parts = []
        for col in numeric:
            tmp = df.groupby(group_cols)[col].agg(["mean", "std", "count"]).reset_index()
            tmp["metric"] = col
            tmp = tmp.rename(columns={"mean": "mean_value", "std": "std_value", "count": "n_seeds"})
            parts.append(tmp)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    spectral_agg = agg(spectral_all, ["spectrum_type"])
    ident_agg = agg(ident_all, ["T", "estimator"])
    noise_agg = agg(noise_all, ["noise_level"])
    attribution_agg = agg(attribution_all, ["process_type", "level"])

    spectral_agg.to_csv(OUT_TABLE / "seed_stability_spectral_geometry_summary.csv", index=False)
    ident_agg.to_csv(OUT_TABLE / "seed_stability_identifiability_summary.csv", index=False)
    noise_agg.to_csv(OUT_TABLE / "seed_stability_zeta_noise_bridge_summary.csv", index=False)
    attribution_agg.to_csv(OUT_TABLE / "seed_stability_failure_attribution_summary.csv", index=False)

    key_lines = [
        "# Multi-Seed Stability Supplement",
        "",
        f"Seeds: {', '.join(str(s) for s in seeds)}",
        "",
        "No new model was trained. Existing evaluation scripts were rerun under multiple random seeds and aggregated.",
        "",
        "## Key output tables",
        "",
        "- `paper_assets/tables/seed_stability_spectral_geometry_summary.csv`",
        "- `paper_assets/tables/seed_stability_identifiability_summary.csv`",
        "- `paper_assets/tables/seed_stability_zeta_noise_bridge_summary.csv`",
        "- `paper_assets/tables/seed_stability_failure_attribution_summary.csv`",
        "",
        "## Interpretation note",
        "",
        "Use these tables to report mean/std over random seeds. The finite-sample identifiability result should still be framed cautiously because quick-mode sample sizes are intentionally small.",
    ]
    (OUT_SUMMARY / "seed_stability_supplement.md").write_text("\n".join(key_lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "seeds": seeds,
                "spectral_summary": str((OUT_TABLE / "seed_stability_spectral_geometry_summary.csv").relative_to(ROOT)),
                "identifiability_summary": str((OUT_TABLE / "seed_stability_identifiability_summary.csv").relative_to(ROOT)),
                "zeta_noise_summary": str((OUT_TABLE / "seed_stability_zeta_noise_bridge_summary.csv").relative_to(ROOT)),
                "failure_attribution_summary": str((OUT_TABLE / "seed_stability_failure_attribution_summary.csv").relative_to(ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
