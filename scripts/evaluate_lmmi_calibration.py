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

from src.mrw_dl.models import LMMICurvatureRegressor, LMMINetRegressor  # noqa: E402
from src.mrw_dl.spectrum_baseline import (  # noqa: E402
    SpectrumTrainConfig,
    _build_loaders,
    _parameter_metrics,
    _predict_all,
    _set_seed,
    _spectrum_metrics,
)


MODEL_SPECS = {
    "lmmi_net": (LMMINetRegressor, ROOT / "outputs" / "dl_spectrum_lmmi_net"),
    "lmmi_curvature": (LMMICurvatureRegressor, ROOT / "outputs" / "dl_spectrum_lmmi_curvature"),
    "lmmi_curvature_focus": (LMMICurvatureRegressor, ROOT / "outputs" / "dl_spectrum_lmmi_curvature_focus"),
}
OUT_PATH = ROOT / "outputs" / "reports" / "lmmi_calibrated_metrics.csv"


def _fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.stack([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


def _evaluate(model_name: str, model_cls: type[torch.nn.Module], output_dir: Path) -> dict[str, float | str]:
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    cfg = SpectrumTrainConfig(**metrics["config"])
    cfg.output_dir = str(output_dir)

    _set_seed(cfg.seed)
    _, loaders, _, _, target_mean, target_std = _build_loaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_cls(output_dim=2, dropout=cfg.dropout).to(device)
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    model.eval()

    val_pred, val_true = _predict_all(model, loaders["val_idx"], device, target_mean, target_std)
    test_pred, test_true = _predict_all(model, loaders["test_idx"], device, target_mean, target_std)

    lambda_a, lambda_b = _fit_affine(val_pred[:, 0], val_true[:, 0])
    h_a, h_b = _fit_affine(val_pred[:, 1], val_true[:, 1])
    calibrated = test_pred.copy()
    calibrated[:, 0] = lambda_a * calibrated[:, 0] + lambda_b
    calibrated[:, 1] = h_a * calibrated[:, 1] + h_b

    q_vals = np.linspace(cfg.q_min, cfg.q_max, cfg.q_count, dtype=np.float32)
    param_metrics = _parameter_metrics(test_true, calibrated, ["lambda2", "H"])
    spectrum_metrics = _spectrum_metrics(test_true, calibrated, q_vals)
    return {
        "model": model_name,
        "lambda_calibration_a": lambda_a,
        "lambda_calibration_b": lambda_b,
        "H_calibration_a": h_a,
        "H_calibration_b": h_b,
        "lambda2_mae": param_metrics["lambda2"]["mae"],
        "H_mae": param_metrics["H"]["mae"],
        "zeta_mae": spectrum_metrics["zeta_mae"],
        "f_alpha_mae": spectrum_metrics["f_mae"],
        "alpha_mae": spectrum_metrics["alpha_mae"],
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
