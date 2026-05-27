from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from mrw_inverse.data import AnalyticSpectrumDatasetConfig, generate_analytic_spectrum_dataset
from mrw_inverse.data.analytic_spectrum_dataset import _diagnostics, _linear_zeta, _mono_fit, _mrw_fit, _mrw_zeta
from mrw_inverse.models import SpectralGeometryCalibrator

CHECKPOINT_PATH = ROOT / "checkpoints" / "cmin" / "spectral_geometry_calibrator.pt"
REPORT_DIR = ROOT / "outputs" / "reports" / "spectral_geometry_calibrator_eval"
TABLE_DIR = ROOT / "outputs" / "tables" / "spectral_geometry_calibrator_eval"
FIG_DIR = ROOT / "outputs" / "figures" / "spectral_geometry_calibrator_eval"


def _predict(model, ds, device):
    rows = []
    with torch.no_grad():
        n = ds["zeta_input"].shape[0]
        for i in range(n):
            args = [torch.tensor(ds[k][i:i+1], dtype=torch.float32, device=device) for k in [
                "zeta_input","zeta_mono","zeta_mrw","mono_residual_norm","mrw_residual_norm","mrw_vs_mono_gain","normalized_mrw_gain","curvature_score","linearity_score","boundary_score","target_tail_instability"]]
            out = model(*args)
            rows.append({"spectrum_type": ds["spectrum_type"][i], "lambda2_true": float(ds["lambda2_true"][i,0]), **{k: float(v.item()) for k,v in out.items() if k.startswith("p_")}})
    return pd.DataFrame(rows)


def _mono_frac(vals, inc=True):
    d = np.diff(vals)
    return float((d >= -1e-6).mean() if inc else (d <= 1e-6).mean()) if len(d) else float("nan")


def _controlled_sweep(model, device):
    q = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32)
    rows = []
    for h in [0.2, 0.4, 0.6, 0.8]:
        for lam in [0.0, 0.005, 0.01, 0.03, 0.06, 0.10, 0.15, 0.20]:
            zeta = _linear_zeta(q, h) if lam == 0.0 else _mrw_zeta(q, h, lam)
            _hm, zmono, mono_r = _mono_fit(q, zeta)
            _h, lam_proj, zmrw, mrw_r = _mrw_fit(q, zeta)
            d = _diagnostics(q, zeta, zmono, zmrw, mono_r, mrw_r, lam_proj)
            args = [
                zeta[None, :], zmono[None, :], zmrw[None, :],
                np.array([[mono_r]], dtype=np.float32), np.array([[mrw_r]], dtype=np.float32),
                np.array([[d["mrw_vs_mono_gain"]]], dtype=np.float32),
                np.array([[d["normalized_mrw_gain"]]], dtype=np.float32),
                np.array([[d["curvature_score"]]], dtype=np.float32),
                np.array([[d["linearity_score"]]], dtype=np.float32),
                np.array([[d["boundary_score"]]], dtype=np.float32),
                np.array([[0.0]], dtype=np.float32),
            ]
            tensors = [torch.tensor(a, dtype=torch.float32, device=device) for a in args]
            with torch.no_grad():
                out = model(*tensors)
            rows.append({"H": h, "lambda2_true": lam, **{k: float(v.item()) for k, v in out.items() if k.startswith("p_")}})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", default=str(CHECKPOINT_PATH)); p.add_argument("--num-samples", type=int, default=6000); p.add_argument("--seed", type=int, default=8080); p.add_argument("--device", default="cpu")
    args = p.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True); TABLE_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        out = {"status":"missing_checkpoint","checkpoint":str(ckpt)}; print(json.dumps(out, indent=2)); return
    device = args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    state = torch.load(ckpt, map_location="cpu"); model = SpectralGeometryCalibrator(); model.load_state_dict(state["model_state_dict"], strict=False); model.to(device).eval()
    ds = generate_analytic_spectrum_dataset(AnalyticSpectrumDatasetConfig(num_samples=args.num_samples, seed=args.seed))
    df = _predict(model, ds, device); df.to_csv(TABLE_DIR / "predictions.csv", index=False)
    summary = df.groupby("spectrum_type").mean(numeric_only=True).reset_index(); summary.to_csv(TABLE_DIR / "summary_by_spectrum_type.csv", index=False)
    sweep = _controlled_sweep(model, device).groupby("lambda2_true").mean(numeric_only=True).reset_index().sort_values("lambda2_true")
    sweep.to_csv(TABLE_DIR / "lambda2_sweep.csv", index=False)
    monotonic = {"p_curved_monotonic_fraction": _mono_frac(sweep["p_curved"].to_numpy(), True), "p_mrw_monotonic_fraction": _mono_frac(sweep["p_mrw"].to_numpy(), True), "p_mono_decreasing_fraction": _mono_frac(sweep["p_mono"].to_numpy(), False)}
    fig, ax = plt.subplots(figsize=(7,4.8), constrained_layout=True)
    for col in ["p_curved","p_mrw","p_mono","p_boundary"]: ax.plot(sweep["lambda2_true"], sweep[col], marker="o", label=col)
    ax.set_xlabel("lambda2_true"); ax.set_ylabel("score"); ax.set_title("Analytic spectrum lambda2 sweep"); ax.legend(); fig.savefig(FIG_DIR / "lambda2_sweep_scores.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5,5), constrained_layout=True)
    for typ, sub in df.groupby("spectrum_type"): ax.scatter(sub["p_curved"], sub["p_mrw"], s=10, alpha=.45, label=typ)
    ax.set_xlabel("p_curved"); ax.set_ylabel("p_MRW"); ax.legend(fontsize=7, ncol=2); fig.savefig(FIG_DIR / "spectrum_type_map.png", dpi=220); plt.close(fig)
    report = REPORT_DIR / "spectral_geometry_calibrator_eval_summary.md"
    report.write_text("\n".join(["# Spectral Geometry Calibrator Evaluation","",summary.to_csv(index=False),"","## Monotonicity","",json.dumps(monotonic, indent=2)]), encoding="utf-8")
    meta = {"summary": str((TABLE_DIR/"summary_by_spectrum_type.csv").relative_to(ROOT)), "lambda2_sweep": str((TABLE_DIR/"lambda2_sweep.csv").relative_to(ROOT)), "report": str(report.relative_to(ROOT)), **monotonic}
    (REPORT_DIR / "spectral_geometry_calibrator_eval_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__": main()
