from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "real"
DEFAULT_ASSETS = ["SPY", "QQQ", "BTC-USD", "ETH-USD", "GLD", "^VIX"]


def _safe_asset_name(asset: str) -> str:
    return asset.replace("^", "").replace("-", "_")


def download_prices(assets: list[str], start: str, end: str | None) -> pd.DataFrame:
    data = yf.download(
        tickers=assets,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"].copy()
        elif "Adj Close" in data.columns.get_level_values(0):
            close = data["Adj Close"].copy()
        else:
            raise KeyError(f"Could not find Close/Adj Close in columns: {data.columns}")
    else:
        close = data[["Close"]].copy()
        close.columns = assets[:1]
    close = close.sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def _download_one_chart(asset: str, start: str, end: str | None) -> pd.Series:
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int((pd.Timestamp(end) if end else pd.Timestamp.utcnow()).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{asset}"
    params = {
        "period1": start_ts,
        "period2": end_ts,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    payload = response.json()
    result = payload["chart"]["result"][0]
    timestamps = pd.to_datetime(result["timestamp"], unit="s").tz_localize(None)
    quote = result["indicators"]["quote"][0]
    adjclose = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    values = adjclose if adjclose is not None else quote["close"]
    return pd.Series(values, index=timestamps, name=asset, dtype="float64").dropna()


def download_prices_direct_chart(assets: list[str], start: str, end: str | None) -> pd.DataFrame:
    series = []
    for asset in assets:
        try:
            series.append(_download_one_chart(asset, start, end))
            time.sleep(0.3)
        except Exception as exc:
            print(f"warning: failed direct Yahoo chart download for {asset}: {exc}")
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1).sort_index()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices).diff()
    returns = returns.replace([np.inf, -np.inf], np.nan)
    return returns.dropna(how="all")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Yahoo Finance prices and log returns.")
    parser.add_argument("--assets", nargs="*", default=DEFAULT_ASSETS)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = OUT_DIR / "yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    prices = download_prices(args.assets, args.start, args.end)
    if prices.empty:
        print("warning: yfinance returned no rows; trying direct Yahoo chart API fallback")
        prices = download_prices_direct_chart(args.assets, args.start, args.end)
    returns = compute_returns(prices)

    prices.to_csv(OUT_DIR / "yahoo_prices.csv", index_label="date")
    returns.to_csv(OUT_DIR / "yahoo_log_returns.csv", index_label="date")

    for asset in prices.columns:
        name = _safe_asset_name(asset)
        asset_df = pd.DataFrame({"price": prices[asset], "log_return": returns.get(asset)})
        asset_df.dropna().to_csv(OUT_DIR / f"{name}_prices_returns.csv", index_label="date")

    np.savez_compressed(
        OUT_DIR / "yahoo_log_returns.npz",
        dates=returns.index.astype(str).to_numpy(),
        assets=np.array(list(returns.columns), dtype=str),
        returns=returns.to_numpy(dtype=np.float32),
        prices=prices.reindex(returns.index).to_numpy(dtype=np.float32),
    )
    meta = {
        "assets": list(prices.columns),
        "start": args.start,
        "end": args.end,
        "num_rows_prices": int(len(prices)),
        "num_rows_returns": int(len(returns)),
        "outputs": [
            "data/real/yahoo_prices.csv",
            "data/real/yahoo_log_returns.csv",
            "data/real/yahoo_log_returns.npz",
        ],
    }
    (OUT_DIR / "yahoo_data_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
