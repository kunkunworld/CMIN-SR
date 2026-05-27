from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mrw_dl.models import LMMINetRegressor, PCSMINRegressor  # noqa: E402
from src.mrw_dl.spectrum_baseline import (  # noqa: E402
    SpectrumTrainConfig,
    _build_loaders,
    _parameter_metrics,
    _predict_all,
    _set_seed,
    _spectrum_metrics,
)


PC_DIR = ROOT / "outputs" / "dl_spectrum_pc_smin"
LMMI_DIR = ROOT / "outputs" / "dl_spectrum_lmmi_net"
OUT_DIR = ROOT / "outputs" / "dl_spectrum_final_hybrid"
REPORT_PATH = ROOT / "outputs" / "reports" / "final_hybrid_metrics.json"


def _load_config(output_dir: Path) -> SpectrumTrainConfig:
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    return SpectrumTrainConfig(**metrics["config"])


def _load_model(model_cls: type[torch.nn.Module], output_dir: Path, dropout: float) -> torch.nn.Module:
    model = model_cls(output_dim=2, dropout=dropout)
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location="cpu"))
    model.eval()
    return model


def _pair_accuracy(
    true: np.ndarray,
    pred: np.ndarray,
    matched_index: int,
    varied_index: int,
    match_tol: float,
    varied_delta: float,
) -> dict[str, float | int]:
    match_diff = true[:, matched_index, None] - true[:, matched_index][None, :]
    varied_diff = true[:, varied_index, None] - true[:, varied_index][None, :]
    upper = np.triu(np.ones_like(match_diff, dtype=bool), k=1)
    mask = upper & (np.abs(match_diff) <= match_tol) & (np.abs(varied_diff) >= varied_delta)
    n_pairs = int(mask.sum())
    if n_pairs == 0:
        return {"pairs": 0, "order_acc": float("nan"), "signed_margin": float("nan")}
    pred_diff = pred[:, varied_index, None] - pred[:, varied_index][None, :]
    signed = np.sign(varied_diff[mask]) * pred_diff[mask]
    return {
        "pairs": n_pairs,
        "order_acc": float(np.mean(signed > 0.0)),
        "signed_margin": float(np.mean(signed)),
    }


def main() -> None:
    _set_seed(2026)
    cfg = _load_config(PC_DIR)
    _, loaders, _, _, target_mean, target_std = _build_loaders(cfg)

    pc_model = _load_model(PCSMINRegressor, PC_DIR, cfg.dropout)
    lmmi_model = _load_model(LMMINetRegressor, LMMI_DIR, cfg.dropout)

    pc_pred, true = _predict_all(pc_model, loaders["test_idx"], torch.device("cpu"), target_mean, target_std)
    lmmi_pred, _ = _predict_all(lmmi_model, loaders["test_idx"], torch.device("cpu"), target_mean, target_std)

    # Final mechanism split: PC-SMIN is the intermittency/lambda2 expert;
    # LMMI-Net is the slope/H expert.
    hybrid_pred = np.column_stack([pc_pred[:, 0], lmmi_pred[:, 1]]).astype(np.float32)

    q_vals = np.linspace(cfg.q_min, cfg.q_max, cfg.q_count, dtype=np.float32)
    parameter_metrics = _parameter_metrics(true, hybrid_pred, ["lambda2", "H"])
    spectrum_metrics = _spectrum_metrics(true, hybrid_pred, q_vals)
    lambda_diag = _pair_accuracy(
        true=true,
        pred=hybrid_pred,
        matched_index=1,
        varied_index=0,
        match_tol=cfg.pair_h_tolerance,
        varied_delta=cfg.pair_lambda_delta,
    )
    h_diag = _pair_accuracy(
        true=true,
        pred=hybrid_pred,
        matched_index=0,
        varied_index=1,
        match_tol=cfg.pair_lambda_tolerance,
        varied_delta=cfg.pair_h_delta,
    )

    summary = {
        "model_name": "final_hybrid_pc_lambda_lmmi_h",
        "description": "Structured expert fusion: lambda2 from PC-SMIN, H from LMMI-Net, analytic MRW spectrum decoder.",
        "pc_smin_dir": str(PC_DIR),
        "lmmi_dir": str(LMMI_DIR),
        "parameter_metrics": parameter_metrics,
        "spectrum_metrics": spectrum_metrics,
        "controlled_identifiability": {
            "matched_H_lambda": lambda_diag,
            "matched_lambda_H": h_diag,
        },
        "q_vals": q_vals.tolist(),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        OUT_DIR / "test_predictions.npz",
        pred=hybrid_pred,
        true=true,
        q_vals=q_vals,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
