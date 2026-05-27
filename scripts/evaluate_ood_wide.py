from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mrw_dl.data import load_dataset, load_splits, select_features
from src.mrw_dl.models import LMMINetRegressor, PCSMINRegressor, PCSMINV2Regressor
from src.mrw_dl.spectrum_baseline import (
    SpectrumTrainConfig,
    _parameter_metrics,
    _set_seed,
    _spectrum_metrics,
)


TRAIN_DATA = ROOT / "data" / "raw" / "mrw_dataset_robust_fgn.npz"
OOD_DATA = ROOT / "data" / "raw" / "mrw_dataset_wide_fgn.npz"
SPLITS = ROOT / "data" / "processed" / "splits_robust_fgn_4800.npz"
OUT_PATH = ROOT / "outputs" / "reports" / "ood_wide_fgn_metrics.csv"


MODELS = {
    "pc_smin": (
        PCSMINRegressor,
        ROOT / "outputs" / "dl_spectrum_pc_smin" / "best_model.pt",
        {},
    ),
    "pc_smin_v2_no_gate": (
        PCSMINV2Regressor,
        ROOT / "outputs" / "dl_spectrum_pc_smin_v2_no_gate" / "best_model.pt",
        {"use_raw_gates": False},
    ),
    "lmmi_net": (
        LMMINetRegressor,
        ROOT / "outputs" / "dl_spectrum_lmmi_net" / "best_model.pt",
        {},
    ),
}


def _pointwise_standardize_with_train_stats(x: np.ndarray) -> np.ndarray:
    train_bundle = load_dataset(TRAIN_DATA)
    train_splits = load_splits(SPLITS)
    train_x = select_features(train_bundle, "dx")
    mean = train_x[train_splits["train_idx"]].mean(axis=0, keepdims=True)
    std = np.maximum(train_x[train_splits["train_idx"]].std(axis=0, keepdims=True), 1e-6)
    return ((x - mean) / std).astype(np.float32)


def _load_target_stats(output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    target_mean = np.asarray(metrics["target_mean"], dtype=np.float32).reshape(1, -1)
    target_std = np.asarray(metrics["target_std"], dtype=np.float32).reshape(1, -1)
    return target_mean, target_std


def evaluate_model(
    model_name: str,
    model_cls: type[torch.nn.Module],
    checkpoint: Path,
    model_kwargs: dict[str, object],
    x: np.ndarray,
    y_true: np.ndarray,
    q_vals: np.ndarray,
) -> dict[str, float | str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_cls(output_dim=2, dropout=0.1, **model_kwargs).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    target_mean, target_std = _load_target_stats(checkpoint.parent)
    preds = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 128):
            batch = torch.from_numpy(x[start : start + 128]).to(device)
            pred_norm = model(batch).cpu().numpy()
            preds.append(pred_norm * target_std + target_mean)
    y_pred = np.concatenate(preds, axis=0)

    param_metrics = _parameter_metrics(y_true, y_pred, ["lambda2", "H"])
    spectrum_metrics = _spectrum_metrics(y_true, y_pred, q_vals)
    return {
        "model": model_name,
        "lambda2_mae": param_metrics["lambda2"]["mae"],
        "H_mae": param_metrics["H"]["mae"],
        "zeta_mae": spectrum_metrics["zeta_mae"],
        "f_alpha_mae": spectrum_metrics["f_mae"],
        "alpha_mae": spectrum_metrics["alpha_mae"],
    }


def main() -> None:
    _set_seed(2026)
    config = SpectrumTrainConfig()
    q_vals = np.linspace(config.q_min, config.q_max, config.q_count, dtype=np.float32)

    ood_bundle = load_dataset(OOD_DATA)
    x = _pointwise_standardize_with_train_stats(select_features(ood_bundle, "dx"))
    y_true = ood_bundle.params[:, [0, 3]].astype(np.float32)

    rows = [
        evaluate_model(model_name, model_cls, checkpoint, kwargs, x, y_true, q_vals)
        for model_name, (model_cls, checkpoint, kwargs) in MODELS.items()
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
