from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_dl.data import RAW_DATA_PATH, load_dataset, load_splits, select_features, standardize_from_train
from mrw_dl.models import LMMINetRegressor


def _latent_correlations(latent: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    latent = (latent - latent.mean(axis=0, keepdims=True)) / np.maximum(latent.std(axis=0, keepdims=True), 1e-8)
    lambda2 = (targets[:, 0] - targets[:, 0].mean()) / max(targets[:, 0].std(), 1e-8)
    h = (targets[:, 1] - targets[:, 1].mean()) / max(targets[:, 1].std(), 1e-8)
    corr_lambda = np.mean(latent * lambda2[:, None], axis=0)
    corr_h = np.mean(latent * h[:, None], axis=0)
    return {
        "max_abs_corr_lambda2": float(np.max(np.abs(corr_lambda))),
        "mean_abs_corr_lambda2_top5": float(np.mean(np.sort(np.abs(corr_lambda))[-5:])),
        "max_abs_corr_H": float(np.max(np.abs(corr_h))),
        "mean_abs_corr_H_top5": float(np.mean(np.sort(np.abs(corr_h))[-5:])),
    }


def main() -> None:
    output_dir = ROOT / "outputs" / "dl_spectrum_lmmi_net"
    if len(sys.argv) >= 2:
        output_dir = ROOT / "outputs" / sys.argv[1]

    bundle = load_dataset(RAW_DATA_PATH)
    splits = load_splits(ROOT / "data" / "processed" / "splits_robust_fgn_4800.npz")
    features = select_features(bundle, "dx")
    features, _, _ = standardize_from_train(features, splits["train_idx"], mode="pointwise")
    params = bundle.params[:, [0, 3]].astype(np.float32)
    test_idx = splits["test_idx"]

    model = LMMINetRegressor(output_dim=2)
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location="cpu"))
    model.eval()

    latents: dict[str, list[np.ndarray]] = {"z_slope": [], "z_curvature": [], "z_dependency": []}
    with torch.no_grad():
        for start in range(0, len(test_idx), 64):
            batch_idx = test_idx[start:start + 64]
            xb = torch.from_numpy(features[batch_idx])
            _ = model(xb)
            diagnostics = model.latent_diagnostics()
            for key in latents:
                latents[key].append(diagnostics[key].cpu().numpy())

    targets = params[test_idx]
    summary = {
        key: _latent_correlations(np.concatenate(parts, axis=0), targets)
        for key, parts in latents.items()
    }

    report_dir = ROOT / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"{output_dir.name}_latent_diagnostics.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
