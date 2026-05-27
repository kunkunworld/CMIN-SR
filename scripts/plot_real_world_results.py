from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "outputs" / "real_world" / "rolling_mrw_estimates.csv"
OUT_DIR = ROOT / "outputs" / "real_world" / "figures"


EVENTS = {
    "COVID crash": "2020-03-16",
    "2022 tightening": "2022-06-15",
    "FTX/BTC stress": "2022-11-09",
}


def _mark_events(ax) -> None:
    for label, date in EVENTS.items():
        ts = pd.Timestamp(date)
        ax.axvline(ts, color="0.4", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(ts, 0.98, label, transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right", fontsize=8)


def _plot_asset(df: pd.DataFrame, asset: str, window: int) -> Path:
    sub = df[(df["asset"] == asset) & (df["window"] == window)].sort_values("date")
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    axes[0].plot(sub["date"], sub["price"], color="#1f2937", linewidth=1.2)
    axes[0].set_ylabel("Price")
    axes[0].set_title(f"{asset}: price and rolling MRW latent estimates (window={window})")
    axes[0].set_yscale("log")
    _mark_events(axes[0])

    axes[1].plot(sub["date"], sub["H_final"], label="Final hybrid H", color="#2563eb", linewidth=1.2)
    axes[1].plot(sub["date"], sub["H_pc_smin"], label="PC-SMIN H", color="#93c5fd", linewidth=0.9, alpha=0.8)
    axes[1].set_ylabel("H")
    axes[1].legend(loc="upper right", fontsize=8)
    _mark_events(axes[1])

    axes[2].plot(sub["date"], sub["lambda2_final"], label="Final hybrid lambda2", color="#dc2626", linewidth=1.2)
    axes[2].plot(sub["date"], sub["lambda2_lmmi"], label="LMMI lambda2", color="#fca5a5", linewidth=0.9, alpha=0.8)
    axes[2].set_ylabel("lambda2")
    axes[2].legend(loc="upper right", fontsize=8)
    _mark_events(axes[2])

    axes[2].xaxis.set_major_locator(mdates.YearLocator(1))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    path = OUT_DIR / f"{asset.replace('^', '').replace('-', '_')}_rolling_params_w{window}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_compare(df: pd.DataFrame, assets: list[str], window: int) -> Path:
    sub = df[(df["asset"].isin(assets)) & (df["window"] == window)].copy()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    for asset in assets:
        asset_df = sub[sub["asset"] == asset].sort_values("date")
        axes[0].plot(asset_df["date"], asset_df["H_final"], label=asset, linewidth=1.1)
        axes[1].plot(asset_df["date"], asset_df["lambda2_final"], label=asset, linewidth=1.1)
    axes[0].set_title(f"Cross-asset rolling latent comparison (window={window})")
    axes[0].set_ylabel("H_final")
    axes[1].set_ylabel("lambda2_final")
    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)
        _mark_events(ax)
    axes[1].xaxis.set_major_locator(mdates.YearLocator(1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    path = OUT_DIR / f"cross_asset_SPY_BTC_w{window}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot real-world rolling MRW estimates.")
    parser.add_argument("--window", type=int, default=256)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_PATH, parse_dates=["date"])
    paths = []
    for asset in ["SPY", "QQQ", "BTC-USD", "ETH-USD", "GLD", "^VIX"]:
        if ((df["asset"] == asset) & (df["window"] == args.window)).any():
            paths.append(_plot_asset(df, asset, args.window))
    if {"SPY", "BTC-USD"}.issubset(set(df["asset"].unique())):
        paths.append(_plot_compare(df, ["SPY", "BTC-USD"], args.window))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
