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

from src.mrw_dl.models import LMMINetRegressor, PCSMINRegressor  # noqa: E402
from src.mrw_dl.spectrum_baseline import SpectrumTrainConfig, _build_loaders, _predict_all  # noqa: E402


MODEL_SPECS = {
    "pc_smin": (PCSMINRegressor, ROOT / "outputs" / "dl_spectrum_pc_smin"),
    "lmmi_net": (LMMINetRegressor, ROOT / "outputs" / "dl_spectrum_lmmi_net"),
    "lmmi_ident": (LMMINetRegressor, ROOT / "outputs" / "dl_spectrum_lmmi_ident"),
    "lmmi_ident_h_control": (LMMINetRegressor, ROOT / "outputs" / "dl_spectrum_lmmi_ident_h_control"),
}
OUT_PATH = ROOT / "outputs" / "reports" / "identifiability_pair_diagnostics.csv"


def _pair_accuracy(
    true: np.ndarray,
    pred: np.ndarray,
    matched_index: int,
    varied_index: int,
    match_tol: float,
    varied_delta: float,
) -> tuple[int, float, float]:
    match_diff = true[:, matched_index, None] - true[:, matched_index][None, :]
    varied_diff = true[:, varied_index, None] - true[:, varied_index][None, :]
    upper = np.triu(np.ones_like(match_diff, dtype=bool), k=1)
    mask = upper & (np.abs(match_diff) <= match_tol) & (np.abs(varied_diff) >= varied_delta)
    n_pairs = int(mask.sum())
    if n_pairs == 0:
        return 0, float("nan"), float("nan")

    pred_diff = pred[:, varied_index, None] - pred[:, varied_index][None, :]
    signed = np.sign(varied_diff[mask]) * pred_diff[mask]
    accuracy = float(np.mean(signed > 0.0))
    mean_signed_margin = float(np.mean(signed))
    return n_pairs, accuracy, mean_signed_margin


def _load_model(model_cls: type[torch.nn.Module], output_dir: Path, dropout: float) -> torch.nn.Module:
    model = model_cls(output_dim=2, dropout=dropout)
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location="cpu"))
    model.eval()
    return model


def _evaluate(model_name: str, model_cls: type[torch.nn.Module], output_dir: Path) -> dict[str, float | int | str]:
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    cfg = SpectrumTrainConfig(**metrics["config"])
    _, loaders, _, _, target_mean, target_std = _build_loaders(cfg)
    model = _load_model(model_cls, output_dir, cfg.dropout)
    pred, true = _predict_all(model, loaders["test_idx"], torch.device("cpu"), target_mean, target_std)

    lambda_pairs, lambda_acc, lambda_margin = _pair_accuracy(
        true=true,
        pred=pred,
        matched_index=1,
        varied_index=0,
        match_tol=cfg.pair_h_tolerance,
        varied_delta=cfg.pair_lambda_delta,
    )
    h_pairs, h_acc, h_margin = _pair_accuracy(
        true=true,
        pred=pred,
        matched_index=0,
        varied_index=1,
        match_tol=cfg.pair_lambda_tolerance,
        varied_delta=cfg.pair_h_delta,
    )
    return {
        "model": model_name,
        "matched_H_lambda_pairs": lambda_pairs,
        "matched_H_lambda_order_acc": lambda_acc,
        "matched_H_lambda_signed_margin": lambda_margin,
        "matched_lambda_H_pairs": h_pairs,
        "matched_lambda_H_order_acc": h_acc,
        "matched_lambda_H_signed_margin": h_margin,
    }


def main() -> None:
    rows = [
        _evaluate(model_name, model_cls, output_dir)
        for model_name, (model_cls, output_dir) in MODEL_SPECS.items()
        if (output_dir / "best_model.pt").exists()
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
