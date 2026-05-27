from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_inverse.data import AnalyticSpectrumDatasetConfig, SPECTRUM_TYPES, generate_analytic_spectrum_dataset
from mrw_inverse.losses import spectrum_space_calibration_loss
from mrw_inverse.models import SpectralGeometryCalibrator

CHECKPOINT_PATH = ROOT / "checkpoints" / "cmin" / "spectral_geometry_calibrator.pt"
REPORT_DIR = ROOT / "outputs" / "reports" / "spectral_geometry_calibrator_training"
TABLE_DIR = ROOT / "outputs" / "tables" / "spectral_geometry_calibrator_training"
FIG_DIR = ROOT / "outputs" / "figures" / "spectral_geometry_calibrator_training"

ORDER = [
    "zeta_input", "zeta_mono", "zeta_mrw", "mono_residual_norm", "mrw_residual_norm",
    "mrw_vs_mono_gain", "normalized_mrw_gain", "curvature_score", "linearity_score",
    "boundary_score", "target_tail_instability", "spectrum_code", "lambda2_true",
    "target_p_scaling", "target_p_curved", "target_p_mono", "target_p_mrw", "target_p_boundary",
]


def _loader(ds, batch_size, shuffle):
    tensors = []
    for k in ORDER:
        dtype = torch.long if k == "spectrum_code" else torch.float32
        tensors.append(torch.tensor(ds[k], dtype=dtype))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def _forward(model, batch, device):
    vals = [x.to(device) for x in batch]
    return vals, model(*vals[:11])


def _eval(model, loader, device):
    model.eval()
    rows = []
    losses = []
    with torch.no_grad():
        for batch in loader:
            vals, out = _forward(model, batch, device)
            spectrum_code, lambda2_true = vals[11], vals[12]
            targets = vals[13:]
            loss = spectrum_space_calibration_loss(out, spectrum_code, lambda2_true, *targets)
            losses.append(float(loss.total.cpu()))
            for i in range(spectrum_code.shape[0]):
                rows.append({
                    "spectrum_type": SPECTRUM_TYPES[int(spectrum_code[i].cpu())],
                    "lambda2_true": float(lambda2_true[i, 0].cpu()),
                    "p_scaling": float(out["p_scaling"][i, 0].cpu()),
                    "p_curved": float(out["p_curved"][i, 0].cpu()),
                    "p_mono": float(out["p_mono"][i, 0].cpu()),
                    "p_mrw": float(out["p_mrw"][i, 0].cpu()),
                    "p_boundary": float(out["p_boundary"][i, 0].cpu()),
                })
    df = pd.DataFrame(rows)
    summary = df.groupby("spectrum_type").mean(numeric_only=True).reset_index()
    return float(np.mean(losses)), df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train", type=int, default=20000)
    parser.add_argument("--num-val", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True); TABLE_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True); CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    train = generate_analytic_spectrum_dataset(AnalyticSpectrumDatasetConfig(num_samples=args.num_train, seed=args.seed))
    val = generate_analytic_spectrum_dataset(AnalyticSpectrumDatasetConfig(num_samples=args.num_val, seed=args.seed + 1))
    train_loader = _loader(train, args.batch_size, True)
    val_loader = _loader(val, args.batch_size, False)
    model = SpectralGeometryCalibrator().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    hist = []
    best = None; best_loss = float("inf"); best_summary = None
    nan = False
    for ep in range(1, args.epochs + 1):
        model.train(); losses = []
        for batch in train_loader:
            vals, out = _forward(model, batch, device)
            loss = spectrum_space_calibration_loss(out, vals[11], vals[12], *vals[13:]).total
            if torch.isnan(loss):
                nan = True; break
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.detach().cpu()))
        val_loss, _pred, summary = _eval(model, val_loader, device)
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss; best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; best_summary = summary
        if nan:
            break
    torch.save({"model_state_dict": best, "model_name": "spectral_geometry_calibrator", "config": vars(args)}, CHECKPOINT_PATH)
    pd.DataFrame(hist).to_csv(TABLE_DIR / "train_history.csv", index=False)
    best_summary.to_csv(TABLE_DIR / "val_by_spectrum_type.csv", index=False)
    fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
    h = pd.DataFrame(hist); ax.plot(h["epoch"], h["train_loss"], label="train"); ax.plot(h["epoch"], h["val_loss"], label="val"); ax.legend(); ax.set_title("Spectral geometry calibrator")
    fig.savefig(FIG_DIR / "loss_curve.png", dpi=220); plt.close(fig)
    report = REPORT_DIR / "spectral_geometry_calibrator_training_summary.md"
    report.write_text("\n".join(["# Spectral Geometry Calibrator Training", "", f"- Checkpoint: `{CHECKPOINT_PATH.relative_to(ROOT)}`", f"- NaN occurred: `{nan}`", "", best_summary.to_csv(index=False)]), encoding="utf-8")
    meta = {"checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "report": str(report.relative_to(ROOT)), "nan_occurred": nan}
    (REPORT_DIR / "spectral_geometry_calibrator_training_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
