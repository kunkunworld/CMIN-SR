from __future__ import annotations

import csv
import json
import re
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "real"
OUTPUT_DIR = ROOT / "data" / "real_processed"
FIG_DIR = ROOT / "outputs" / "reports" / "real_processed_figures"
REPORT_PATH = ROOT / "outputs" / "reports" / "real_data_preprocessing_summary.md"
SUMMARY_JSON = ROOT / "outputs" / "reports" / "real_data_preprocessing_summary.json"

MISSING_SENTINELS = {-99.99, -999.0, -999.99}
DATE_RE = re.compile(r"^\s*\d{8}\s*,")


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _find_fama_french_table(lines: list[str]) -> tuple[int, int] | None:
    first_data = None
    for i, line in enumerate(lines):
        if DATE_RE.match(line):
            first_data = i
            break
    if first_data is None:
        return None

    header = first_data - 1
    while header >= 0 and "," not in lines[header]:
        header -= 1
    if header < 0:
        return None

    end = first_data
    while end < len(lines) and DATE_RE.match(lines[end]):
        end += 1
    return header, end


def _parse_fama_french(path: Path) -> tuple[pd.DataFrame, dict[str, object]] | None:
    lines = _read_lines(path)
    bounds = _find_fama_french_table(lines)
    if bounds is None:
        return None
    header, end = bounds
    table_text = "\n".join(lines[header:end])
    df = pd.read_csv(StringIO(table_text))
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "date"})
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    value_cols = [c for c in df.columns if c != "date"]
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col].round(2).isin(MISSING_SENTINELS), col] = np.nan

    # Fama-French daily factor files are in percent units. Keep this explicit
    # rather than relying only on heuristics.
    df[value_cols] = df[value_cols] / 100.0
    meta = {
        "data_type": "factor_return",
        "parser": "fama_french_daily",
        "header_line": header + 1,
        "first_data_line": header + 2,
        "last_data_line": end,
        "return_columns": value_cols,
        "unit_conversion": "percent_to_decimal",
        "cleaning": "Skipped text header/footer; parsed YYYYMMDD dates; converted -99.99/-999 to NaN; divided returns by 100.",
    }
    return df, meta


def _parse_price_history(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.read_csv(path)
    lower_cols = {str(c).strip().lower(): c for c in df.columns}
    date_col = next((lower_cols[c] for c in ["date", "datetime", "time", "timestamp"] if c in lower_cols), None)
    if date_col is None:
        raise ValueError(f"Could not infer date column for {path}")
    price_col = next(
        (lower_cols[c] for c in ["adj close", "adj_close", "adjusted close", "close", "price", "value"] if c in lower_cols),
        None,
    )
    if price_col is None:
        raise ValueError(f"Could not infer price/value column for {path}")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    price = pd.to_numeric(df[price_col], errors="coerce")
    out["return"] = np.log(price).diff()
    out = out.dropna(subset=["date", "return"]).sort_values("date")
    meta = {
        "data_type": "price_history",
        "parser": "generic_price_history",
        "date_column": str(date_col),
        "price_column": str(price_col),
        "return_columns": ["return"],
        "unit_conversion": "computed_log_returns",
        "cleaning": "Parsed date/price columns, sorted by date, computed log returns, removed non-finite rows.",
    }
    return out, meta


def _to_long_returns(df: pd.DataFrame, path: Path, meta: dict[str, object]) -> pd.DataFrame:
    rows = []
    if meta["data_type"] == "factor_return":
        for col in meta["return_columns"]:
            series = df[["date", col]].rename(columns={col: "return"}).copy()
            series["series_name"] = str(col).strip()
            rows.append(series)
    else:
        series = df[["date", "return"]].copy()
        series["series_name"] = "return"
        rows.append(series)
    long_df = pd.concat(rows, axis=0, ignore_index=True)
    long_df["return"] = pd.to_numeric(long_df["return"], errors="coerce")
    long_df = long_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "return"])
    long_df["source_file"] = path.name
    long_df["source_name"] = long_df["source_file"].map(lambda x: Path(x).stem) + ":" + long_df["series_name"]
    long_df["data_type"] = str(meta["data_type"])
    long_df["unit_conversion"] = str(meta["unit_conversion"])
    return long_df[["date", "return", "source_name", "series_name", "source_file", "data_type", "unit_conversion"]]


def _plot_series(series: pd.DataFrame, output_path: Path) -> None:
    series = series.sort_values("date")
    rolling_vol = series["return"].rolling(63).std() * np.sqrt(252)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, constrained_layout=True)
    axes[0].plot(series["date"], series["return"], color="#1f2937", linewidth=0.6)
    axes[0].axhline(0.0, color="0.5", linewidth=0.8)
    axes[0].set_title(series["source_name"].iloc[0])
    axes[0].set_ylabel("daily return")
    axes[1].plot(series["date"], rolling_vol, color="#dc2626", linewidth=1.0)
    axes[1].set_ylabel("63d ann. vol")
    axes[1].set_xlabel("date")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(path for path in INPUT_DIR.glob("*.csv") if path.is_file())
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

    all_long = []
    file_summaries = []
    for path in csv_paths:
        parsed = _parse_fama_french(path)
        if parsed is None:
            df, meta = _parse_price_history(path)
        else:
            df, meta = parsed
        long_df = _to_long_returns(df, path, meta)
        out_path = OUTPUT_DIR / f"{_safe_name(path.stem)}_processed_returns.csv"
        long_df.to_csv(out_path, index=False)
        all_long.append(long_df)

        for source_name, source_df in long_df.groupby("source_name"):
            figure_path = FIG_DIR / f"{_safe_name(source_name)}_sanity.png"
            _plot_series(source_df, figure_path)

        file_summaries.append(
            {
                "input_file": str(path.relative_to(ROOT)),
                "processed_file": str(out_path.relative_to(ROOT)),
                "data_type": meta["data_type"],
                "parser": meta["parser"],
                "date_start": str(long_df["date"].min().date()),
                "date_end": str(long_df["date"].max().date()),
                "valid_observations": int(len(long_df)),
                "series": sorted(long_df["source_name"].unique().tolist()),
                "unit_conversion": meta["unit_conversion"],
                "cleaning": meta["cleaning"],
            }
        )

    combined = pd.concat(all_long, axis=0, ignore_index=True).sort_values(["source_name", "date"])
    combined_path = OUTPUT_DIR / "all_processed_returns.csv"
    combined.to_csv(combined_path, index=False)
    wide = combined.pivot_table(index="date", columns="source_name", values="return", aggfunc="first").sort_index()
    wide_path = OUTPUT_DIR / "aligned_returns_wide.csv"
    wide.to_csv(wide_path, index_label="date")

    summary = {
        "input_dir": str(INPUT_DIR.relative_to(ROOT)),
        "output_dir": str(OUTPUT_DIR.relative_to(ROOT)),
        "figure_dir": str(FIG_DIR.relative_to(ROOT)),
        "input_files_found": [str(p.relative_to(ROOT)) for p in csv_paths],
        "combined_processed_file": str(combined_path.relative_to(ROOT)),
        "aligned_wide_file": str(wide_path.relative_to(ROOT)),
        "files": file_summaries,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "# Real Data Preprocessing Summary",
        "",
        f"Input directory: `{summary['input_dir']}`",
        f"Output directory: `{summary['output_dir']}`",
        f"Combined long file: `{summary['combined_processed_file']}`",
        f"Aligned wide file: `{summary['aligned_wide_file']}`",
        "",
        "## Files",
        "",
    ]
    for item in file_summaries:
        report_lines.extend(
            [
                f"### {item['input_file']}",
                "",
                f"- Inferred type: `{item['data_type']}`",
                f"- Parser: `{item['parser']}`",
                f"- Date range: `{item['date_start']}` to `{item['date_end']}`",
                f"- Valid observations: `{item['valid_observations']}`",
                f"- Processed file: `{item['processed_file']}`",
                f"- Unit handling: `{item['unit_conversion']}`",
                f"- Cleaning: {item['cleaning']}",
                f"- Series: {', '.join(f'`{s}`' for s in item['series'])}",
                "",
            ]
        )
    report_lines.extend(
        [
            "## Assumptions",
            "",
            "- Fama-French daily factor values are percent returns and were divided by 100.",
            "- `RF` is retained as a return series for completeness, but downstream MRW rolling inference should usually focus on risky factor series such as `Mkt-RF`, `SMB`, `HML`, and `Mom`.",
            "- Missing sentinels `-99.99`, `-999`, and `-999.99` are treated as missing.",
            "",
            "## Ready For Rolling Inference",
            "",
            "The processed long-format file contains `date`, `return`, and `source_name`, so it is ready for rolling-window model inference.",
        ]
    )
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
