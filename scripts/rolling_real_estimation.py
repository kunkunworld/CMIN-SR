from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mrw_dl.data import RAW_DATA_PATH, load_dataset, load_splits, select_features, standardize_from_train  # noqa: E402
from src.mrw_dl.models import LMMINetRegressor, PCSMINRegressor  # noqa: E402


REAL_DIR = ROOT / "data" / "real"
PROCESSED_LONG_PATH = ROOT / "data" / "real_processed" / "all_processed_returns.csv"
OUT_DIR = ROOT / "outputs" / "real_world"
SPLIT_PATH = ROOT / "data" / "processed" / "splits_robust_fgn_4800.npz"
MODEL_LENGTH = 4096


def _load_target_stats(output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    target_mean = np.asarray(metrics["target_mean"], dtype=np.float32).reshape(1, -1)
    target_std = np.asarray(metrics["target_std"], dtype=np.float32).reshape(1, -1)
    return target_mean, target_std


def _load_train_feature_stats() -> tuple[np.ndarray, np.ndarray]:
    bundle = load_dataset(RAW_DATA_PATH)
    splits = load_splits(SPLIT_PATH)
    features = select_features(bundle, "dx")
    _, mean, std = standardize_from_train(features, splits["train_idx"], mode="pointwise")
    return mean.reshape(-1), std.reshape(-1)


def _resample_window(window: np.ndarray, target_length: int = MODEL_LENGTH) -> np.ndarray:
    x = np.asarray(window, dtype=np.float32)
    x = x - np.nanmean(x)
    scale = np.nanstd(x)
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    x = x / scale
    old_grid = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
    new_grid = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    return np.interp(new_grid, old_grid, x).astype(np.float32)


def _window_features(returns: np.ndarray, window: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(returns, dtype=np.float32)
    valid = np.isfinite(values)
    if valid.sum() < window:
        return np.empty((0, MODEL_LENGTH), dtype=np.float32), np.empty((0,), dtype=np.int64)
    rows = []
    end_positions = []
    for end in range(window, len(values) + 1, step):
        segment = values[end - window : end]
        if np.isfinite(segment).all():
            rows.append(_resample_window(segment))
            end_positions.append(end - 1)
    if not rows:
        return np.empty((0, MODEL_LENGTH), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(rows, axis=0), np.asarray(end_positions, dtype=np.int64)


def _predict_params(
    model: torch.nn.Module,
    features: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    batch_size: int = 128,
) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            xb = torch.from_numpy(features[start : start + batch_size])
            pred_norm = model(xb).cpu().numpy()
            preds.append(pred_norm * target_std + target_mean)
    return np.concatenate(preds, axis=0)


def _load_model(model_cls: type[torch.nn.Module], output_dir: Path) -> tuple[torch.nn.Module, np.ndarray, np.ndarray]:
    model = model_cls(output_dim=2, dropout=0.1)
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location="cpu"))
    model.eval()
    target_mean, target_std = _load_target_stats(output_dir)
    return model, target_mean, target_std


def _clip_to_train_support(params: np.ndarray) -> np.ndarray:
    bundle = load_dataset(RAW_DATA_PATH)
    support = bundle.params[:, [0, 3]].astype(np.float32)
    lo = np.percentile(support, 0.5, axis=0)
    hi = np.percentile(support, 99.5, axis=0)
    return np.clip(params, lo, hi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling MRW parameter estimates on real returns.")
    parser.add_argument("--input-long", default=str(PROCESSED_LONG_PATH))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--also-window", type=int, default=512)
    parser.add_argument("--step", type=int, default=5)
    args = parser.parse_args()

    input_long = Path(args.input_long)
    if input_long.exists():
        long_returns = pd.read_csv(input_long, parse_dates=["date"])
        returns = long_returns.pivot_table(index="date", columns="source_name", values="return", aggfunc="first").sort_index()
        if "price" in long_returns.columns:
            prices = long_returns.pivot_table(index="date", columns="source_name", values="price", aggfunc="first").sort_index()
        else:
            prices = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=np.float32)
        input_description = str(input_long.relative_to(ROOT))
    else:
        returns = pd.read_csv(REAL_DIR / "yahoo_log_returns.csv", parse_dates=["date"]).set_index("date")
        prices = pd.read_csv(REAL_DIR / "yahoo_prices.csv", parse_dates=["date"]).set_index("date")
        input_description = "data/real/yahoo_log_returns.csv"

    feature_mean, feature_std = _load_train_feature_stats()
    pc_model, pc_target_mean, pc_target_std = _load_model(PCSMINRegressor, ROOT / "outputs" / "dl_spectrum_pc_smin")
    lmmi_model, lmmi_target_mean, lmmi_target_std = _load_model(LMMINetRegressor, ROOT / "outputs" / "dl_spectrum_lmmi_net")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    windows = [args.window]
    if args.also_window and args.also_window != args.window:
        windows.append(args.also_window)

    for window in windows:
        for asset in returns.columns:
            series = returns[asset].dropna()
            if len(series) < window:
                continue
            raw_features, end_positions = _window_features(series.to_numpy(), window=window, step=args.step)
            if len(raw_features) == 0:
                continue
            features = ((raw_features - feature_mean.reshape(1, -1)) / feature_std.reshape(1, -1)).astype(np.float32)
            pc_pred = _clip_to_train_support(_predict_params(pc_model, features, pc_target_mean, pc_target_std))
            lmmi_pred = _clip_to_train_support(_predict_params(lmmi_model, features, lmmi_target_mean, lmmi_target_std))
            hybrid_pred = np.column_stack([pc_pred[:, 0], lmmi_pred[:, 1]]).astype(np.float32)

            dates = series.index[end_positions]
            price_series = prices[asset].reindex(dates)
            realized_vol_5 = series.rolling(5).std().shift(-5).reindex(dates) * np.sqrt(252)
            realized_vol_10 = series.rolling(10).std().shift(-10).reindex(dates) * np.sqrt(252)
            hist_vol_20 = series.rolling(20).std().reindex(dates) * np.sqrt(252)
            hist_vol_window = series.rolling(window).std().reindex(dates) * np.sqrt(252)

            for i, date in enumerate(dates):
                all_rows.append(
                    {
                        "date": date,
                        "asset": asset,
                        "window": window,
                        "price": float(price_series.iloc[i]) if pd.notna(price_series.iloc[i]) else np.nan,
                        "lambda2_pc_smin": float(pc_pred[i, 0]),
                        "H_pc_smin": float(pc_pred[i, 1]),
                        "lambda2_lmmi": float(lmmi_pred[i, 0]),
                        "H_lmmi": float(lmmi_pred[i, 1]),
                        "lambda2_final": float(hybrid_pred[i, 0]),
                        "H_final": float(hybrid_pred[i, 1]),
                        "hist_vol_20": float(hist_vol_20.iloc[i]) if pd.notna(hist_vol_20.iloc[i]) else np.nan,
                        "hist_vol_window": float(hist_vol_window.iloc[i]) if pd.notna(hist_vol_window.iloc[i]) else np.nan,
                        "future_rv_5": float(realized_vol_5.iloc[i]) if pd.notna(realized_vol_5.iloc[i]) else np.nan,
                        "future_rv_10": float(realized_vol_10.iloc[i]) if pd.notna(realized_vol_10.iloc[i]) else np.nan,
                    }
                )

    out = pd.DataFrame(all_rows).sort_values(["asset", "window", "date"])
    out.to_csv(output_dir / "rolling_mrw_estimates.csv", index=False)
    np.savez_compressed(
        output_dir / "rolling_mrw_estimates.npz",
        values=out.drop(columns=["date", "asset"]).to_numpy(dtype=np.float32),
        dates=out["date"].astype(str).to_numpy(),
        assets=out["asset"].astype(str).to_numpy(),
        columns=np.array([c for c in out.columns if c not in {"date", "asset"}], dtype=str),
    )
    summary = {
        "input_returns": input_description,
        "output_csv": str((output_dir / "rolling_mrw_estimates.csv").relative_to(ROOT)),
        "rows": int(len(out)),
        "assets": sorted(out["asset"].dropna().unique().tolist()),
        "windows": sorted(out["window"].dropna().unique().astype(int).tolist()),
        "step": args.step,
        "model_length": MODEL_LENGTH,
        "note": "Real windows are z-scored, resampled to 4096, then standardized with synthetic training pointwise stats.",
    }
    (output_dir / "rolling_mrw_estimates_meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
