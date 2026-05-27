from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROLLING = ROOT / "outputs" / "real_world_market" / "rolling_mrw_estimates.csv"
RETURNS_PATH = ROOT / "data" / "market_processed" / "all_market_returns.csv"
OUT_DIR = ROOT / "outputs" / "reports" / "market_real_world"
FIG_DIR = OUT_DIR / "figures"

EVENTS = {
    "COVID crash": "2020-03-16",
    "2022 tightening": "2022-06-15",
    "BTC/FTX stress": "2022-11-09",
    "Spot BTC ETF": "2024-01-11",
}

FEATURE_SETS = {
    "hist_vol_only": ["hist_vol_20", "hist_vol_window"],
    "pc_smin": ["hist_vol_20", "hist_vol_window", "lambda2_pc_smin", "H_pc_smin"],
    "lmmi_net": ["hist_vol_20", "hist_vol_window", "lambda2_lmmi", "H_lmmi"],
    "final_hybrid": ["hist_vol_20", "hist_vol_window", "lambda2_final", "H_final"],
}
MODELS = {"linear": LinearRegression(), "ridge": Ridge(alpha=1.0), "lasso": Lasso(alpha=1e-4, max_iter=10000)}


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    spearman = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(spearman) if np.isfinite(spearman) else float("nan"),
    }


def _mark_events(ax) -> None:
    for label, date in EVENTS.items():
        ts = pd.Timestamp(date)
        ax.axvline(ts, color="0.3", linestyle="--", linewidth=0.8, alpha=0.65)
        ax.text(ts, 0.97, label, transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right", fontsize=8)


def _future_rv(returns: pd.DataFrame, estimates: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frames = []
    for asset, est in estimates.groupby("asset"):
        ret = returns[returns["source_name"] == asset].sort_values("date").reset_index(drop=True)
        pos = {date: i for i, date in enumerate(ret["date"])}
        vals = ret["return"].to_numpy(dtype=np.float64)
        targets = []
        for date in est["date"]:
            i = pos.get(date)
            if i is None or i + horizon >= len(vals):
                targets.append(np.nan)
            else:
                targets.append(float(np.std(vals[i + 1 : i + horizon + 1], ddof=1) * np.sqrt(252)))
        tmp = est.copy()
        tmp[f"future_rv_{horizon}"] = targets
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True)


def _plot_asset(df: pd.DataFrame, asset: str) -> list[Path]:
    sub = df[df["asset"] == asset].sort_values("date")
    paths = []
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, constrained_layout=True)
    axes[0].plot(sub["date"], sub["price"], color="#111827", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Price")
    axes[0].set_title(f"{asset}: price and final-hybrid lambda2")
    axes[1].plot(sub["date"], sub["lambda2_final"], color="#b91c1c", linewidth=1.1)
    axes[1].set_ylabel("lambda2_final")
    for ax in axes:
        _mark_events(ax)
    axes[1].xaxis.set_major_locator(mdates.YearLocator(1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    path = FIG_DIR / f"{_safe_name(asset)}_price_lambda2.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, constrained_layout=True)
    axes[0].plot(sub["date"], sub["price"], color="#111827", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Price")
    axes[0].set_title(f"{asset}: price and final-hybrid H")
    axes[1].plot(sub["date"], sub["H_final"], color="#1d4ed8", linewidth=1.1)
    axes[1].set_ylabel("H_final")
    for ax in axes:
        _mark_events(ax)
    axes[1].xaxis.set_major_locator(mdates.YearLocator(1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    path = FIG_DIR / f"{_safe_name(asset)}_price_H.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)
    return paths


def _plot_compare(df: pd.DataFrame) -> Path | None:
    assets = sorted(df["asset"].unique())
    btc = next((a for a in assets if "BTC" in a.upper()), None)
    spy = next((a for a in assets if "SPY" in a.upper()), None)
    if btc is None or spy is None:
        return None
    sub = df[df["asset"].isin([spy, btc])].copy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    for asset in [spy, btc]:
        tmp = sub[sub["asset"] == asset].sort_values("date")
        axes[0].plot(tmp["date"], tmp["H_final"], label=asset, linewidth=1.1)
        axes[1].plot(tmp["date"], tmp["lambda2_final"], label=asset, linewidth=1.1)
    axes[0].set_title("SPY vs BTC latent MRW mechanisms")
    axes[0].set_ylabel("H_final")
    axes[1].set_ylabel("lambda2_final")
    for ax in axes:
        ax.legend(fontsize=8)
        _mark_events(ax)
    path = FIG_DIR / "SPY_vs_BTC_latents.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _forecast(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows = []
    target = f"future_rv_{horizon}"
    for asset, asset_df in df.groupby("asset"):
        asset_df = asset_df.sort_values("date")
        for feature_set, features in FEATURE_SETS.items():
            usable = asset_df.dropna(subset=features + [target])
            if len(usable) < 80:
                continue
            split = int(len(usable) * 0.7)
            train, test = usable.iloc[:split], usable.iloc[split:]
            x_train = train[features].to_numpy(dtype=np.float64)
            y_train = train[target].to_numpy(dtype=np.float64)
            x_test = test[features].to_numpy(dtype=np.float64)
            y_test = test[target].to_numpy(dtype=np.float64)
            for model_name, model in MODELS.items():
                estimator = make_pipeline(StandardScaler(), model)
                estimator.fit(x_train, y_train)
                pred = estimator.predict(x_test)
                rows.append(
                    {
                        "asset": asset,
                        "horizon": horizon,
                        "feature_set": feature_set,
                        "model": model_name,
                        "n_train": len(train),
                        "n_test": len(test),
                        **_metrics(y_test, pred),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze raw market rolling MRW outputs.")
    parser.add_argument("--rolling", default=str(DEFAULT_ROLLING))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rolling_path = Path(args.rolling)
    if not rolling_path.exists() or not RETURNS_PATH.exists():
        report = {
            "status": "missing_market_inputs",
            "needed_rolling": str(rolling_path.relative_to(ROOT)) if rolling_path.is_absolute() and ROOT in rolling_path.parents else str(rolling_path),
            "needed_returns": str(RETURNS_PATH.relative_to(ROOT)),
            "next_steps": [
                "Place SPY/QQQ/BTC/ETH price-history CSVs under data/real/, data/market/, or data/real_market/.",
                "Run: conda run -n for_codex python scripts/preprocess_market_price_csvs.py",
                "Run: conda run -n for_codex python scripts/rolling_real_estimation.py --input-long data/market_processed/all_market_returns.csv --output-dir outputs/real_world_market --window 256 --also-window 512 --step 5",
                "Run: conda run -n for_codex python scripts/analyze_market_real_world.py",
            ],
        }
        out = OUT_DIR / "market_real_world_summary.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    rolling = pd.read_csv(rolling_path, parse_dates=["date"])
    returns = pd.read_csv(RETURNS_PATH, parse_dates=["date"])
    rolling = _future_rv(returns, rolling, horizon=5)
    rolling = _future_rv(returns, rolling, horizon=20)
    enriched_path = OUT_DIR / "market_rolling_with_future_rv.csv"
    rolling.to_csv(enriched_path, index=False)

    stats = (
        rolling.groupby("asset")
        .agg(
            n_obs=("H_final", "count"),
            date_start=("date", "min"),
            date_end=("date", "max"),
            H_mean=("H_final", "mean"),
            H_std=("H_final", "std"),
            H_range=("H_final", lambda x: float(x.max() - x.min())),
            lambda2_mean=("lambda2_final", "mean"),
            lambda2_std=("lambda2_final", "std"),
            lambda2_range=("lambda2_final", lambda x: float(x.max() - x.min())),
            lambda2_histvol_corr=("lambda2_final", lambda x: float(pd.Series(x).corr(rolling.loc[x.index, "hist_vol_20"]))),
            lambda2_future20_corr=("lambda2_final", lambda x: float(pd.Series(x).corr(rolling.loc[x.index, "future_rv_20"]))),
        )
        .reset_index()
    )
    stats_path = OUT_DIR / "market_parameter_statistics.csv"
    stats.to_csv(stats_path, index=False)

    metrics = pd.concat([_forecast(rolling, 5), _forecast(rolling, 20)], ignore_index=True)
    metrics_path = OUT_DIR / "market_vol_forecast_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    fig_paths = []
    for asset in sorted(rolling["asset"].unique()):
        fig_paths.extend(_plot_asset(rolling, asset))
    compare = _plot_compare(rolling)
    if compare is not None:
        fig_paths.append(compare)

    report_path = OUT_DIR / "market_real_world_summary.md"
    lines = [
        "# Raw Market Real-World MRW Validation Summary",
        "",
        f"Rolling input: `{rolling_path.relative_to(ROOT)}`",
        f"Processed returns: `{RETURNS_PATH.relative_to(ROOT)}`",
        "",
        "## Parameter Statistics",
        "",
        stats.to_csv(index=False),
        "",
        "## Forecast Metrics",
        "",
        metrics.groupby(["horizon", "feature_set", "model"], as_index=False)[["mae", "rmse", "r2", "spearman"]].mean().to_csv(index=False),
        "",
        "## Figures",
        "",
    ]
    lines.extend([f"- `{p.relative_to(ROOT)}`" for p in fig_paths])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "status": "completed",
        "rolling_with_targets": str(enriched_path.relative_to(ROOT)),
        "parameter_statistics": str(stats_path.relative_to(ROOT)),
        "forecast_metrics": str(metrics_path.relative_to(ROOT)),
        "figures": [str(p.relative_to(ROOT)) for p in fig_paths],
        "report": str(report_path.relative_to(ROOT)),
    }
    (OUT_DIR / "market_real_world_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
