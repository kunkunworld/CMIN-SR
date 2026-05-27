from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_dl.baselines import legendre_spectrum_from_zeta, true_mrw_zeta
from mrw_dl.data import RAW_DATA_PATH, load_dataset, load_splits
from mrw_dl.models import LMMINetRegressor, PCSMINRegressor, PCSMINV2Regressor


MODEL_SPECS = {
    "pc_smin": (PCSMINRegressor, ROOT / "outputs" / "dl_spectrum_pc_smin"),
    "pc_smin_v2_no_gate": (
        lambda output_dim: PCSMINV2Regressor(output_dim=output_dim, use_raw_gates=False),
        ROOT / "outputs" / "dl_spectrum_pc_smin_v2_no_gate",
    ),
    "lmmi_net": (LMMINetRegressor, ROOT / "outputs" / "dl_spectrum_lmmi_net"),
}


def _standardize_crop(dx: np.ndarray, train_idx: np.ndarray, length: int) -> np.ndarray:
    cropped = dx[:, :length].astype(np.float32)
    mean = cropped[train_idx].mean(axis=0, keepdims=True)
    std = np.maximum(cropped[train_idx].std(axis=0, keepdims=True), 1e-6)
    return ((cropped - mean) / std).astype(np.float32)


def _spectrum_metrics(true_params: np.ndarray, pred_params: np.ndarray, q_vals: np.ndarray) -> dict[str, float]:
    zeta_true = np.stack([true_mrw_zeta(q_vals, h=row[1], lambda2=row[0]) for row in true_params])
    zeta_pred = np.stack([true_mrw_zeta(q_vals, h=row[1], lambda2=row[0]) for row in pred_params])
    alpha_true, f_true, alpha_pred, f_pred = [], [], [], []
    for i in range(zeta_true.shape[0]):
        a_t, f_t = legendre_spectrum_from_zeta(q_vals, zeta_true[i])
        a_p, f_p = legendre_spectrum_from_zeta(q_vals, zeta_pred[i])
        alpha_true.append(a_t)
        f_true.append(f_t)
        alpha_pred.append(a_p)
        f_pred.append(f_p)
    alpha_true = np.stack(alpha_true)
    f_true = np.stack(f_true)
    alpha_pred = np.stack(alpha_pred)
    f_pred = np.stack(f_pred)
    return {
        "zeta_mae": float(np.mean(np.abs(zeta_true - zeta_pred))),
        "f_alpha_mae": float(np.mean(np.abs(f_true - f_pred))),
        "alpha_mae": float(np.mean(np.abs(alpha_true - alpha_pred))),
    }


def _evaluate_model(model_name: str, length: int) -> dict[str, float | str | int]:
    bundle = load_dataset(RAW_DATA_PATH)
    splits = load_splits(ROOT / "data" / "processed" / "splits_robust_fgn_4800.npz")
    features = _standardize_crop(bundle.dx, splits["train_idx"], length)
    params = bundle.params[:, [0, 3]].astype(np.float32)
    target_mean = params[splits["train_idx"]].mean(axis=0, keepdims=True)
    target_std = np.maximum(params[splits["train_idx"]].std(axis=0, keepdims=True), 1e-6)

    factory, output_dir = MODEL_SPECS[model_name]
    model = factory(output_dim=2)
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location="cpu"))
    model.eval()

    preds = []
    test_idx = splits["test_idx"]
    with torch.no_grad():
        for start in range(0, len(test_idx), 64):
            batch_idx = test_idx[start:start + 64]
            xb = torch.from_numpy(features[batch_idx])
            pred_norm = model(xb).cpu().numpy()
            preds.append(pred_norm * target_std + target_mean)
    pred = np.concatenate(preds, axis=0)
    true = params[test_idx]
    err = pred - true
    q_vals = np.linspace(0.5, 3.0, 11)
    spec = _spectrum_metrics(true, pred, q_vals)
    return {
        "model": model_name,
        "length": length,
        "lambda2_mae": float(np.mean(np.abs(err[:, 0]))),
        "H_mae": float(np.mean(np.abs(err[:, 1]))),
        **spec,
    }


def main() -> None:
    lengths = [4096, 2048, 1024]
    rows = [_evaluate_model(model_name, length) for model_name in MODEL_SPECS for length in lengths]
    out_path = ROOT / "outputs" / "reports" / "length_robustness.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out_path)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
