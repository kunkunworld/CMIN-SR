from __future__ import annotations

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
ROLLING_PATH = ROOT / "outputs" / "real_world" / "rolling_mrw_estimates.csv"
RETURNS_PATH = ROOT / "data" / "real_processed" / "all_processed_returns.csv"
OUT_DIR = ROOT / "outputs" / "reports" / "factor_real_world"
FIG_DIR = OUT_DIR / "figures"

MKT = "F-F_Research_Data_Factors_daily:Mkt-RF"
SMB = "F-F_Research_Data_Factors_daily:SMB"
MOM = "F-F_Momentum_Factor_daily:Mom"
ASSETS = [MKT, SMB, MOM]

CRISES = {
    "1929 Crash": "1929-10-24",
    "1987 Crash": "1987-10-19",
    "Dot-com Peak": "2000-03-10",
    "Lehman/GFC": "2008-09-15",
    "COVID Crash": "2020-03-16",
}

FEATURE_SETS = {
    "hist_vol_only": ["hist_vol_20", "hist_vol_window"],
    "pc_smin": ["hist_vol_20", "hist_vol_window", "lambda2_pc_smin", "H_pc_smin"],
    "lmmi_net": ["hist_vol_20", "hist_vol_window", "lambda2_lmmi", "H_lmmi"],
    "final_hybrid": ["hist_vol_20", "hist_vol_window", "lambda2_final", "H_final"],
}

MODELS = {
    "linear": LinearRegression(),
    "ridge": Ridge(alpha=1.0),
    "lasso": Lasso(alpha=1e-4, max_iter=10000),
}


def _short_name(source_name: str) -> str:
    return source_name.split(":")[-1]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    spearman = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(spearman) if np.isfinite(spearman) else float("nan"),
    }


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                vals.append(format(float(value), floatfmt))
            elif isinstance(value, pd.Timestamp):
                vals.append(str(value.date()))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _future_realized_vol(
    returns: pd.DataFrame,
    estimates: pd.DataFrame,
    horizon: int = 20,
) -> pd.DataFrame:
    frames = []
    for asset, est in estimates.groupby("asset"):
        ret = returns[returns["source_name"] == asset].sort_values("date").reset_index(drop=True)
        ret_dates = pd.Series(ret["date"].to_numpy())
        ret_values = ret["return"].to_numpy(dtype=np.float64)
        pos_map = {date: idx for idx, date in enumerate(ret_dates)}
        targets = []
        for date in est["date"]:
            pos = pos_map.get(date)
            if pos is None or pos + horizon >= len(ret_values):
                targets.append(np.nan)
                continue
            future = ret_values[pos + 1 : pos + horizon + 1]
            targets.append(float(np.std(future, ddof=1) * np.sqrt(252)) if np.isfinite(future).all() else np.nan)
        tmp = est.copy()
        tmp[f"future_rv_{horizon}"] = targets
        frames.append(tmp)
    return pd.concat(frames, axis=0, ignore_index=True)


def _mark_crises(ax) -> None:
    for label, date in CRISES.items():
        ts = pd.Timestamp(date)
        ax.axvline(ts, color="0.25", linestyle="--", linewidth=0.9, alpha=0.65)
        ax.text(ts, 0.98, label, transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right", fontsize=8)


def plot_mktrf_params(df: pd.DataFrame) -> Path:
    sub = df[df["asset"] == MKT].sort_values("date")
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.8), sharex=True, constrained_layout=True)
    axes[0].plot(sub["date"], sub["H_final"], color="#1d4ed8", linewidth=1.2, label="Final hybrid H")
    axes[0].plot(sub["date"], sub["H_pc_smin"], color="#93c5fd", linewidth=0.8, alpha=0.75, label="PC-SMIN H")
    axes[0].set_title("Mkt-RF rolling MRW latent roughness and intermittency, 1926-2026")
    axes[0].set_ylabel("H(t)")
    axes[0].legend(loc="upper right", fontsize=8)
    _mark_crises(axes[0])

    axes[1].plot(sub["date"], sub["lambda2_final"], color="#b91c1c", linewidth=1.2, label="Final hybrid lambda2")
    axes[1].plot(sub["date"], sub["lambda2_lmmi"], color="#fca5a5", linewidth=0.8, alpha=0.75, label="LMMI lambda2")
    axes[1].set_ylabel("lambda2(t)")
    axes[1].legend(loc="upper right", fontsize=8)
    _mark_crises(axes[1])
    axes[1].xaxis.set_major_locator(mdates.YearLocator(10))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    path = FIG_DIR / "mktrf_H_lambda2_1926_2026.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_factor_comparison(df: pd.DataFrame) -> Path:
    sub = df[df["asset"].isin(ASSETS)].copy()
    sub["factor"] = sub["asset"].map(_short_name)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    sub.boxplot(column="H_final", by="factor", ax=axes[0], grid=False, color="#1d4ed8")
    axes[0].set_title("H distribution")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("H_final")
    sub.boxplot(column="lambda2_final", by="factor", ax=axes[1], grid=False, color="#b91c1c")
    axes[1].set_title("lambda2 distribution")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("lambda2_final")
    fig.suptitle("MRW latent parameter comparison across factor returns")
    path = FIG_DIR / "factor_parameter_boxplots.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def run_forecast(df: pd.DataFrame, target_col: str = "future_rv_20") -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    pred_rows = []
    for asset, asset_df in df[df["asset"].isin(ASSETS)].groupby("asset"):
        asset_df = asset_df.sort_values("date")
        for feature_set, features in FEATURE_SETS.items():
            usable = asset_df.dropna(subset=features + [target_col]).copy()
            if len(usable) < 100:
                continue
            split = int(len(usable) * 0.7)
            train = usable.iloc[:split]
            test = usable.iloc[split:]
            x_train = train[features].to_numpy(dtype=np.float64)
            y_train = train[target_col].to_numpy(dtype=np.float64)
            x_test = test[features].to_numpy(dtype=np.float64)
            y_test = test[target_col].to_numpy(dtype=np.float64)

            mean_pred = np.full_like(y_test, fill_value=float(np.mean(y_train)))
            if feature_set == "hist_vol_only":
                metric_rows.append(
                    {
                        "asset": asset,
                        "feature_set": "train_mean_naive",
                        "model": "mean",
                        "n_train": len(train),
                        "n_test": len(test),
                        **_metrics(y_test, mean_pred),
                    }
                )

            for model_name, model in MODELS.items():
                estimator = make_pipeline(StandardScaler(), model)
                estimator.fit(x_train, y_train)
                pred = estimator.predict(x_test)
                metric_rows.append(
                    {
                        "asset": asset,
                        "feature_set": feature_set,
                        "model": model_name,
                        "n_train": len(train),
                        "n_test": len(test),
                        **_metrics(y_test, pred),
                    }
                )
                for date, truth, yhat in zip(test["date"], y_test, pred):
                    pred_rows.append(
                        {
                            "date": date,
                            "asset": asset,
                            "feature_set": feature_set,
                            "model": model_name,
                            "true": float(truth),
                            "pred": float(yhat),
                        }
                    )
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


def plot_forecast(metrics: pd.DataFrame) -> Path:
    agg = metrics.groupby("feature_set", as_index=False)[["rmse", "r2", "spearman"]].mean()
    order = ["train_mean_naive", "hist_vol_only", "pc_smin", "lmmi_net", "final_hybrid"]
    agg["feature_set"] = pd.Categorical(agg["feature_set"], categories=order, ordered=True)
    agg = agg.sort_values("feature_set")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    colors = ["#6b7280", "#64748b", "#2563eb", "#059669", "#b91c1c"]
    for ax, metric, title in zip(axes, ["rmse", "r2", "spearman"], ["RMSE", "R2", "Spearman"]):
        ax.bar(agg["feature_set"].astype(str), agg[metric], color=colors[: len(agg)])
        ax.set_title(f"Next-20d RV forecast {title}")
        ax.tick_params(axis="x", labelrotation=35)
    path = FIG_DIR / "next20_rv_forecast_summary.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rolling = pd.read_csv(ROLLING_PATH, parse_dates=["date"])
    returns = pd.read_csv(RETURNS_PATH, parse_dates=["date"])
    rolling = rolling[(rolling["window"] == 256) & (rolling["asset"].isin(ASSETS))].copy()
    rolling = _future_realized_vol(returns, rolling, horizon=20)
    rolling_path = OUT_DIR / "factor_rolling_with_future_rv20.csv"
    rolling.to_csv(rolling_path, index=False)

    stats = (
        rolling.groupby("asset")
        .agg(
            n_obs=("H_final", "count"),
            date_start=("date", "min"),
            date_end=("date", "max"),
            H_mean=("H_final", "mean"),
            H_std=("H_final", "std"),
            H_median=("H_final", "median"),
            lambda2_mean=("lambda2_final", "mean"),
            lambda2_std=("lambda2_final", "std"),
            lambda2_median=("lambda2_final", "median"),
            hist_vol_mean=("hist_vol_20", "mean"),
            future_rv20_mean=("future_rv_20", "mean"),
        )
        .reset_index()
    )
    stats_path = OUT_DIR / "factor_parameter_statistics.csv"
    stats.to_csv(stats_path, index=False)

    forecast_metrics, forecast_predictions = run_forecast(rolling)
    metrics_path = OUT_DIR / "factor_next20_rv_forecast_metrics.csv"
    pred_path = OUT_DIR / "factor_next20_rv_forecast_predictions.csv"
    forecast_metrics.to_csv(metrics_path, index=False)
    forecast_predictions.to_csv(pred_path, index=False)

    fig_paths = [
        plot_mktrf_params(rolling),
        plot_factor_comparison(rolling),
        plot_forecast(forecast_metrics),
    ]

    best = forecast_metrics.sort_values(["asset", "rmse"]).groupby("asset").head(3)
    avg = forecast_metrics.groupby(["feature_set", "model"], as_index=False)[["mae", "rmse", "r2", "spearman"]].mean()
    report_path = OUT_DIR / "factor_real_world_summary.md"
    lines = [
        "# Factor Real-World MRW Validation Summary",
        "",
        "## Inputs",
        "",
        f"- Rolling estimates: `{ROLLING_PATH.relative_to(ROOT)}`",
        f"- Processed returns: `{RETURNS_PATH.relative_to(ROOT)}`",
        "- Assets analyzed: `Mkt-RF`, `SMB`, `Mom`",
        "- Rolling window: `256`, step inherited from rolling estimates (`20` trading days in this run)",
        "",
        "## Figures",
        "",
    ]
    lines.extend([f"- `{p.relative_to(ROOT)}`" for p in fig_paths])
    lines.extend(
        [
            "",
            "## Parameter Statistics",
            "",
            _markdown_table(stats, floatfmt=".4f"),
            "",
            "## Forecast Metrics: Average Across Factors",
            "",
            _markdown_table(avg.sort_values(["rmse"]), floatfmt=".4f"),
            "",
            "## Best Models By Asset",
            "",
            _markdown_table(best[["asset", "feature_set", "model", "mae", "rmse", "r2", "spearman"]], floatfmt=".4f"),
            "",
            "## Main Findings",
            "",
            "- The current real-data files are Fama-French factor-return series, not Yahoo price histories.",
            "- `lambda2_final` is inherited from PC-SMIN. On these factor windows it often sits close to the lower synthetic support, so interpret absolute lambda2 levels cautiously.",
            "- `H_final` varies more visibly over the century and is the cleaner transferred latent signal in this first factor-data benchmark.",
            "- Forecasting should be interpreted as a first time-series split benchmark, not a final market claim.",
            "",
            "## Outputs",
            "",
            f"- `{rolling_path.relative_to(ROOT)}`",
            f"- `{stats_path.relative_to(ROOT)}`",
            f"- `{metrics_path.relative_to(ROOT)}`",
            f"- `{pred_path.relative_to(ROOT)}`",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "rolling_with_target": str(rolling_path.relative_to(ROOT)),
        "parameter_statistics": str(stats_path.relative_to(ROOT)),
        "forecast_metrics": str(metrics_path.relative_to(ROOT)),
        "forecast_predictions": str(pred_path.relative_to(ROOT)),
        "figures": [str(p.relative_to(ROOT)) for p in fig_paths],
        "report": str(report_path.relative_to(ROOT)),
    }
    (OUT_DIR / "factor_real_world_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
