from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SURROGATE_SUMMARY = ROOT / "outputs" / "tables" / "raw_market_surrogate_validation" / "raw_market_surrogate_gap_summary.csv"
RETURNS_PATH = ROOT / "data" / "market_processed" / "all_market_returns.csv"
OUT_DIR = ROOT / "outputs" / "reports" / "raw_market_dynamics"
FIG_DIR = OUT_DIR / "figures"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze raw market latent dynamics.")
    parser.add_argument("--surrogate-summary", default=str(SURROGATE_SUMMARY))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sur_path = Path(args.surrogate_summary)
    if not sur_path.exists() or not RETURNS_PATH.exists():
        out = {
            "status": "missing_raw_market_inputs",
            "needed_surrogate_summary": str(sur_path.relative_to(ROOT) if sur_path.is_absolute() else sur_path),
            "needed_returns": str(RETURNS_PATH.relative_to(ROOT)),
            "note": "Run preprocess_market_price_csvs.py and run_raw_market_surrogate_validation.py after providing SPY/QQQ/BTC/ETH CSVs.",
        }
        (OUT_DIR / "raw_market_dynamics_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return

    summary = pd.read_csv(sur_path)
    if summary.empty:
        out = {"status": "empty_surrogate_summary", "path": str(sur_path.relative_to(ROOT))}
        (OUT_DIR / "raw_market_dynamics_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return

    fig_paths = []
    if "pred_lambda2_gap_original_minus_shuffled" in summary.columns:
        fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
        summary.boxplot(column="pred_lambda2_gap_original_minus_shuffled", by="asset", ax=ax, grid=False, rot=35)
        ax.set_title("Raw market original - shuffled lambda2 gaps")
        ax.set_xlabel("")
        ax.set_ylabel("gap")
        fig.suptitle("")
        path = FIG_DIR / "asset_comparison_boxplots.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        fig_paths.append(path)

    report_lines = [
        "# Raw Market Dynamics Summary",
        "",
        f"- Surrogate summary: `{sur_path.relative_to(ROOT)}`",
        "",
        "## Aggregated Gaps",
        "",
        summary.to_csv(index=False),
    ]
    if fig_paths:
        report_lines.extend(["", "## Figures", ""])
        report_lines.extend([f"- `{p.relative_to(ROOT)}`" for p in fig_paths])
    report_path = OUT_DIR / "raw_market_dynamics_summary.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    meta = {
        "report": str(report_path.relative_to(ROOT)),
        "figures": [str(p.relative_to(ROOT)) for p in fig_paths],
    }
    (OUT_DIR / "raw_market_dynamics_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
