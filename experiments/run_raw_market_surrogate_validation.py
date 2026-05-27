from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_inverse.proxy import estimate_to_dict, estimate_window
from mrw_inverse.surrogates import block_shuffled_surrogate, phase_randomized_surrogate, shuffled_surrogate


INPUT_PATH = ROOT / "data" / "market_processed" / "all_market_returns.csv"
REPORT_DIR = ROOT / "outputs" / "reports" / "raw_market_surrogate_validation"
TABLE_DIR = ROOT / "outputs" / "tables" / "raw_market_surrogate_validation"
FIG_DIR = ROOT / "outputs" / "figures" / "raw_market_surrogate_validation"


def _iter_windows(series: np.ndarray, dates: np.ndarray, window: int, step: int):
    for end in range(window, len(series) + 1, step):
        yield dates[end - window], dates[end - 1], series[end - window : end]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run raw market rolling surrogate validation.")
    parser.add_argument("--input", default=str(INPUT_PATH))
    parser.add_argument("--assets", nargs="*", default=["SPY", "QQQ", "BTC", "ETH"])
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--also-window", type=int, default=1024)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--mode", choices=["proxy", "auto", "model"], default="auto")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--include-phase-randomized", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if not input_path.exists():
        out = {
            "status": "missing_market_returns",
            "needed_input": str(input_path.relative_to(ROOT) if input_path.is_absolute() else input_path),
            "expected_columns": ["asset", "date", "price", "log_return", "source_file"],
            "next_step": "Run conda run -n for_codex python scripts/preprocess_market_price_csvs.py after placing SPY/QQQ/BTC/ETH CSVs.",
        }
        (REPORT_DIR / "raw_market_surrogate_validation_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return

    df = pd.read_csv(input_path, parse_dates=["date"])
    asset_col = "asset" if "asset" in df.columns else "source_name"
    available = sorted(df[asset_col].dropna().astype(str).unique().tolist())
    selected = [a for a in available if any(tok.upper() in a.upper() for tok in args.assets)]
    if not selected:
        out = {
            "status": "no_target_assets_found",
            "available_assets": available,
            "requested_assets": args.assets,
            "note": "Expected raw market assets like SPY/QQQ/BTC/ETH were not found in processed market returns.",
        }
        (REPORT_DIR / "raw_market_surrogate_validation_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return

    rng = np.random.default_rng(args.seed)
    windows = [args.window] + ([args.also_window] if args.also_window and args.also_window != args.window else [])
    rows = []
    for asset in selected:
        sub = df[df[asset_col] == asset].sort_values("date")
        x = sub["log_return"].to_numpy(dtype=np.float64) if "log_return" in sub.columns else sub["return"].to_numpy(dtype=np.float64)
        dates = sub["date"].to_numpy()
        for window in windows:
            if len(x) < window:
                continue
            for idx, (d0, d1, arr) in enumerate(_iter_windows(x, dates, window, args.step)):
                variants = {
                    "original": arr.copy(),
                    "shuffled": shuffled_surrogate(arr, rng),
                    "block_shuffled": block_shuffled_surrogate(arr, args.block_size, rng),
                }
                if args.include_phase_randomized:
                    variants["phase_randomized"] = phase_randomized_surrogate(arr, rng)
                for surrogate_type, vals in variants.items():
                    est = estimate_window(vals, checkpoint_path=args.checkpoint, mode=args.mode)
                    row = {
                        "asset": asset,
                        "window_start": pd.Timestamp(d0),
                        "window_end": pd.Timestamp(d1),
                        "window_index": idx,
                        "T": window,
                        "surrogate_type": surrogate_type,
                        **estimate_to_dict(est),
                    }
                    if args.model_name:
                        row["model_name"] = args.model_name
                    rows.append(row)

    metrics = pd.DataFrame(rows)
    detail_path = TABLE_DIR / "raw_market_surrogate_window_metrics.csv"
    metrics.to_csv(detail_path, index=False)

    wide = metrics.pivot_table(
        index=["asset", "window_start", "window_end", "window_index", "T"],
        columns="surrogate_type",
        values=["pred_lambda2", "p_MRW", "f_alpha_width", "logvol_cov_slope", "pred_H"],
        aggfunc="first",
    )
    wide.columns = ["_".join(col) for col in wide.columns]
    wide = wide.reset_index()
    for metric in ["pred_lambda2", "p_MRW", "f_alpha_width", "logvol_cov_slope"]:
        if f"{metric}_original" in wide.columns and f"{metric}_shuffled" in wide.columns:
            wide[f"{metric}_gap_original_minus_shuffled"] = wide[f"{metric}_original"] - wide[f"{metric}_shuffled"]
        if f"{metric}_original" in wide.columns and f"{metric}_block_shuffled" in wide.columns:
            wide[f"{metric}_gap_original_minus_block_shuffled"] = wide[f"{metric}_original"] - wide[f"{metric}_block_shuffled"]
        if f"{metric}_original" in wide.columns and f"{metric}_phase_randomized" in wide.columns:
            wide[f"{metric}_gap_original_minus_phase_randomized"] = wide[f"{metric}_original"] - wide[f"{metric}_phase_randomized"]
    gap_path = TABLE_DIR / "raw_market_surrogate_gap_table.csv"
    wide.to_csv(gap_path, index=False)
    summary = wide.groupby(["asset", "T"]).mean(numeric_only=True).reset_index()
    summary_path = TABLE_DIR / "raw_market_surrogate_gap_summary.csv"
    summary.to_csv(summary_path, index=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    plot_col = "pred_lambda2_gap_original_minus_shuffled"
    if plot_col in wide.columns:
        wide.boxplot(column=plot_col, by="asset", ax=ax, grid=False, rot=35)
        ax.set_title("Raw market lambda2 original - shuffled")
        ax.set_xlabel("")
        ax.set_ylabel("gap")
        fig.suptitle("")
    fig_path = FIG_DIR / "raw_market_lambda2_gap_boxplots.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    report_lines = [
        "# Raw Market Surrogate Validation Summary",
        "",
        f"- Input: `{input_path.relative_to(ROOT) if input_path.is_absolute() else input_path}`",
        f"- Selected assets: `{', '.join(selected)}`",
        f"- Windows: `{windows}`",
        f"- Mode: `{metrics['mode'].iloc[0] if not metrics.empty else 'none'}`",
        f"- Detail CSV: `{detail_path.relative_to(ROOT)}`",
        f"- Gap table: `{gap_path.relative_to(ROOT)}`",
        f"- Gap summary: `{summary_path.relative_to(ROOT)}`",
        f"- Figure: `{fig_path.relative_to(ROOT)}`",
    ]
    report_path = REPORT_DIR / "raw_market_surrogate_validation_summary.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    meta = {
        "detail_csv": str(detail_path.relative_to(ROOT)),
        "gap_table": str(gap_path.relative_to(ROOT)),
        "gap_summary": str(summary_path.relative_to(ROOT)),
        "figure": str(fig_path.relative_to(ROOT)),
        "report": str(report_path.relative_to(ROOT)),
        "selected_assets": selected,
    }
    (REPORT_DIR / "raw_market_surrogate_validation_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
