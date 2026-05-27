from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "outputs" / "real_world" / "rolling_mrw_estimates.csv"
OUT_DIR = ROOT / "outputs" / "real_world"


FEATURE_SETS = {
    "hist_vol_only": ["hist_vol_20", "hist_vol_window"],
    "pc_smin_features": ["hist_vol_20", "hist_vol_window", "lambda2_pc_smin", "H_pc_smin"],
    "lmmi_features": ["hist_vol_20", "hist_vol_window", "lambda2_lmmi", "H_lmmi"],
    "final_hybrid_features": ["hist_vol_20", "hist_vol_window", "lambda2_final", "H_final"],
}


MODELS = {
    "linear": LinearRegression(),
    "ridge": Ridge(alpha=1.0),
    "lasso": Lasso(alpha=1e-4, max_iter=10000),
}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rho = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(rho) if np.isfinite(rho) else float("nan"),
    }


def _time_split(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date")
    split = int(len(df) * train_frac)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast future realized volatility from rolling MRW features.")
    parser.add_argument("--target", default="future_rv_5", choices=["future_rv_5", "future_rv_10"])
    parser.add_argument("--window", type=int, default=256)
    args = parser.parse_args()

    df = pd.read_csv(IN_PATH, parse_dates=["date"])
    df = df[df["window"] == args.window].copy()
    rows = []
    predictions = []

    for asset, asset_df in df.groupby("asset"):
        asset_df = asset_df.dropna(subset=[args.target, "hist_vol_20", "hist_vol_window"]).sort_values("date")
        if len(asset_df) < 100:
            continue
        train_df, test_df = _time_split(asset_df)
        for feature_name, features in FEATURE_SETS.items():
            usable_train = train_df.dropna(subset=features + [args.target])
            usable_test = test_df.dropna(subset=features + [args.target])
            if len(usable_train) < 50 or len(usable_test) < 20:
                continue
            x_train = usable_train[features].to_numpy(dtype=np.float32)
            y_train = usable_train[args.target].to_numpy(dtype=np.float32)
            x_test = usable_test[features].to_numpy(dtype=np.float32)
            y_test = usable_test[args.target].to_numpy(dtype=np.float32)

            naive_pred = np.full_like(y_test, fill_value=float(np.mean(y_train)))
            naive_metrics = _metrics(y_test, naive_pred)
            if feature_name == "hist_vol_only":
                rows.append(
                    {
                        "asset": asset,
                        "window": args.window,
                        "target": args.target,
                        "feature_set": "train_mean_naive",
                        "model": "mean",
                        **naive_metrics,
                    }
                )

            for model_name, model in MODELS.items():
                estimator = make_pipeline(StandardScaler(), model)
                estimator.fit(x_train, y_train)
                pred = estimator.predict(x_test)
                rows.append(
                    {
                        "asset": asset,
                        "window": args.window,
                        "target": args.target,
                        "feature_set": feature_name,
                        "model": model_name,
                        **_metrics(y_test, pred),
                    }
                )
                for date, truth, yhat in zip(usable_test["date"], y_test, pred):
                    predictions.append(
                        {
                            "date": date,
                            "asset": asset,
                            "window": args.window,
                            "target": args.target,
                            "feature_set": feature_name,
                            "model": model_name,
                            "true": float(truth),
                            "pred": float(yhat),
                        }
                    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(rows).sort_values(["target", "asset", "feature_set", "model"])
    pred_df = pd.DataFrame(predictions)
    metrics_path = OUT_DIR / f"forecast_metrics_{args.target}_w{args.window}.csv"
    pred_path = OUT_DIR / f"forecast_predictions_{args.target}_w{args.window}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    summary = {
        "input": str(IN_PATH),
        "target": args.target,
        "window": args.window,
        "metrics": str(metrics_path),
        "predictions": str(pred_path),
        "rows": int(len(metrics_df)),
    }
    (OUT_DIR / f"forecast_summary_{args.target}_w{args.window}.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(metrics_df.groupby(["feature_set", "model"])[["mae", "rmse", "r2", "spearman"]].mean().round(4))


if __name__ == "__main__":
    main()
