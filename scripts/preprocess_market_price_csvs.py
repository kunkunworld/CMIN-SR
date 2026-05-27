from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIRS = [ROOT / "data" / "real", ROOT / "data" / "market", ROOT / "data" / "real_market", ROOT / "data" / "raw_market"]
OUTPUT_DIR = ROOT / "data" / "market_processed"
FIG_DIR = ROOT / "outputs" / "reports" / "market_processed_figures"
REPORT_PATH = ROOT / "outputs" / "reports" / "market_price_preprocessing_summary.md"
SUMMARY_JSON = ROOT / "outputs" / "reports" / "market_price_preprocessing_summary.json"

DATE_CANDIDATES = ["datetime", "date", "time", "timestamp", "open time", "close time"]
PRICE_CANDIDATES = [
    "adj close",
    "adj_close",
    "adjusted close",
    "close",
    "last",
    "price",
    "settle",
    "value",
]
ASSET_HINTS = ["SPY", "QQQ", "BTC", "BTC-USD", "BTCUSD", "ETH", "ETH-USD", "ETHUSD"]


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _canonical_col_map(columns: list[str]) -> dict[str, str]:
    return {str(col).strip().lower(): col for col in columns}


def _looks_like_factor_file(path: Path) -> bool:
    name = path.name.lower()
    if "f-f_" in name or "fama" in name or "factor" in name:
        return True
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:500].lower()
    except Exception:
        return False
    return "fama" in head or "crsp" in head or "portfolio" in head


def _infer_asset_name(path: Path, df: pd.DataFrame) -> str:
    lower_cols = _canonical_col_map(list(df.columns))
    symbol_col = next((lower_cols[c] for c in ["symbol", "ticker", "asset"] if c in lower_cols), None)
    if symbol_col is not None and df[symbol_col].dropna().nunique() == 1:
        return str(df[symbol_col].dropna().iloc[0])
    upper_name = path.stem.upper()
    for hint in ASSET_HINTS:
        if hint.replace("-", "") in upper_name.replace("-", "_").replace("_", "") or hint in upper_name:
            return hint
    return path.stem


def _infer_frequency(dates: pd.Series) -> str:
    dates = pd.Series(pd.to_datetime(dates).dropna().sort_values().unique())
    if len(dates) < 3:
        return "unknown"
    deltas = dates.diff().dropna().dt.total_seconds()
    median = float(deltas.median())
    if median <= 2 * 3600:
        return "intraday_hourly_or_faster"
    if median <= 8 * 3600:
        return "intraday_4h_or_session"
    if median <= 2 * 86400:
        return "daily"
    return "lower_than_daily_or_irregular"


def _read_market_csv(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("empty CSV")
    lower_cols = _canonical_col_map(list(df.columns))
    date_col = next((lower_cols[c] for c in DATE_CANDIDATES if c in lower_cols), None)
    price_col = next((lower_cols[c] for c in PRICE_CANDIDATES if c in lower_cols), None)
    if date_col is None:
        raise ValueError(f"could not infer datetime column from columns={list(df.columns)}")
    if price_col is None:
        raise ValueError(f"could not infer close/price column from columns={list(df.columns)}")

    out = pd.DataFrame()
    raw_dates = df[date_col]
    if pd.api.types.is_numeric_dtype(raw_dates):
        # Heuristic for UNIX timestamps in seconds or milliseconds.
        vals = pd.to_numeric(raw_dates, errors="coerce")
        unit = "ms" if vals.dropna().median() > 1e11 else "s"
        out["date"] = pd.to_datetime(vals, unit=unit, errors="coerce")
    else:
        out["date"] = pd.to_datetime(raw_dates, errors="coerce")
    out["price"] = pd.to_numeric(df[price_col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "price"])
    out = out[out["price"] > 0].sort_values("date").drop_duplicates(subset=["date"], keep="last")
    out["log_return"] = np.log(out["price"]).diff()
    out["simple_return"] = out["price"].pct_change()
    out = out.dropna(subset=["log_return"])
    asset = _infer_asset_name(path, df)
    frequency = _infer_frequency(out["date"])
    out["asset"] = asset
    out["source_name"] = asset
    out["series_name"] = "log_return"
    out["source_file"] = path.name
    out["data_type"] = "price_history"
    out["unit_conversion"] = "computed_log_returns"
    out["frequency"] = frequency
    meta = {
        "input_file": str(path.relative_to(ROOT)),
        "asset": asset,
        "date_column": str(date_col),
        "price_column": str(price_col),
        "frequency": frequency,
        "date_start": str(out["date"].min()),
        "date_end": str(out["date"].max()),
        "valid_observations": int(len(out)),
        "cleaning": "Parsed datetime and price columns; removed invalid rows; sorted by time; computed log returns.",
    }
    out["return"] = out["log_return"]
    return out[
        [
            "asset",
            "date",
            "price",
            "log_return",
            "simple_return",
            "return",
            "source_name",
            "series_name",
            "source_file",
            "data_type",
            "unit_conversion",
            "frequency",
        ]
    ], meta


def _plot_market_series(df: pd.DataFrame, path: Path) -> None:
    df = df.sort_values("date")
    ann_factor = np.sqrt(365 * 24) if str(df["frequency"].iloc[0]).startswith("intraday") else np.sqrt(252)
    rolling = df["return"].rolling(63).std() * ann_factor
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].plot(df["date"], df["price"], color="#111827", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("price")
    axes[0].set_title(f"{df['source_name'].iloc[0]} market CSV sanity check")
    axes[1].plot(df["date"], df["return"], color="#374151", linewidth=0.55)
    axes[1].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1].set_ylabel("log return")
    axes[2].plot(df["date"], rolling, color="#dc2626", linewidth=1.0)
    axes[2].set_ylabel("rolling vol")
    axes[2].set_xlabel("time")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw SPY/QQQ/BTC/ETH price-history CSVs.")
    parser.add_argument("--input-dirs", nargs="*", default=[str(p) for p in DEFAULT_INPUT_DIRS])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    input_dirs = [Path(p) for p in args.input_dirs]
    csv_paths = []
    for input_dir in input_dirs:
        if input_dir.exists():
            csv_paths.extend(sorted(input_dir.glob("*.csv")))
    csv_paths = [p for p in csv_paths if not _looks_like_factor_file(p)]

    processed = []
    summaries = []
    skipped = []
    for path in csv_paths:
        try:
            df, meta = _read_market_csv(path)
        except Exception as exc:
            skipped.append({"input_file": str(path.relative_to(ROOT)), "reason": str(exc)})
            continue
        out_path = OUTPUT_DIR / f"{_safe_name(path.stem)}_market_returns.csv"
        df.to_csv(out_path, index=False)
        fig_path = FIG_DIR / f"{_safe_name(meta['asset'])}_market_sanity.png"
        _plot_market_series(df, fig_path)
        meta["processed_file"] = str(out_path.relative_to(ROOT))
        meta["figure"] = str(fig_path.relative_to(ROOT))
        summaries.append(meta)
        processed.append(df)

    if processed:
        combined = pd.concat(processed, ignore_index=True).sort_values(["source_name", "date"])
    else:
        combined = pd.DataFrame(
            columns=["asset", "date", "price", "log_return", "simple_return", "return", "source_name", "series_name", "source_file", "data_type", "unit_conversion", "frequency"]
        )
    combined_path = OUTPUT_DIR / "all_market_returns.csv"
    combined.to_csv(combined_path, index=False)
    report = {
        "input_dirs": [str(p.relative_to(ROOT)) if p.is_absolute() and ROOT in p.parents else str(p) for p in input_dirs],
        "candidate_csvs": [str(p.relative_to(ROOT)) for p in csv_paths],
        "processed_count": len(summaries),
        "combined_file": str(combined_path.relative_to(ROOT)),
        "processed_files": summaries,
        "skipped_files": skipped,
        "note": "Only non-Fama-French price-history CSVs are processed here. Put SPY/QQQ/BTC/ETH CSVs under data/real/, data/market/, or data/real_market/.",
    }
    SUMMARY_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Market Price CSV Preprocessing Summary",
        "",
        f"Processed count: `{len(summaries)}`",
        f"Combined file: `{report['combined_file']}`",
        "",
    ]
    if summaries:
        lines.append("## Processed Files")
        lines.append("")
        for item in summaries:
            lines.extend(
                [
                    f"### {item['asset']}",
                    "",
                    f"- Input: `{item['input_file']}`",
                    f"- Processed: `{item['processed_file']}`",
                    f"- Figure: `{item['figure']}`",
                    f"- Date range: `{item['date_start']}` to `{item['date_end']}`",
                    f"- Frequency: `{item['frequency']}`",
                    f"- Valid observations: `{item['valid_observations']}`",
                    f"- Date column: `{item['date_column']}`",
                    f"- Price column: `{item['price_column']}`",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "## No Raw Market Price CSVs Detected",
                "",
                "Only Fama-French factor CSVs were present in the scanned folders. To run Phase 2, place SPY/QQQ/BTC/ETH price-history CSVs under `data/real/`, `data/market/`, or `data/real_market/` and rerun:",
                "",
                "```bash",
                "conda run -n for_codex python scripts/preprocess_market_price_csvs.py",
                "```",
                "",
            ]
        )
    if skipped:
        lines.append("## Skipped Files")
        lines.append("")
        for item in skipped:
            lines.append(f"- `{item['input_file']}`: {item['reason']}")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
