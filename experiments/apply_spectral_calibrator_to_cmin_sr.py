from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from mrw_inverse.data import PROCESS_CODE_TO_NAME, SpectralRepresentationDatasetConfig, generate_spectral_representation_dataset
from mrw_inverse.models import CMINSRv3Model, SpectralGeometryCalibrator

CMIN_CKPT = ROOT / "checkpoints" / "cmin" / "cmin_sr_v3_synthetic.pt"
CAL_CKPT = ROOT / "checkpoints" / "cmin" / "spectral_geometry_calibrator.pt"
REPORT_DIR = ROOT / "outputs" / "reports" / "cmin_sr_spectrum_calibrated_eval"
TABLE_DIR = ROOT / "outputs" / "tables" / "cmin_sr_spectrum_calibrated_eval"
FIG_DIR = ROOT / "outputs" / "figures" / "cmin_sr_spectrum_calibrated_eval"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--cmin-checkpoint", default=str(CMIN_CKPT)); p.add_argument("--calibrator", default=str(CAL_CKPT)); p.add_argument("--num-samples", type=int, default=600); p.add_argument("--t-eval", nargs="*", type=int, default=[512,1024]); p.add_argument("--seed", type=int, default=9090); p.add_argument("--device", default="cpu")
    args = p.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True); TABLE_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    cmin_path, cal_path = Path(args.cmin_checkpoint), Path(args.calibrator)
    if not cmin_path.exists() or not cal_path.exists():
        out = {"status":"missing_checkpoint","cmin_exists":cmin_path.exists(),"calibrator_exists":cal_path.exists()}
        (REPORT_DIR/"cmin_sr_spectrum_calibrated_eval_warning.json").write_text(json.dumps(out, indent=2), encoding="utf-8"); print(json.dumps(out, indent=2)); return
    device = args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    cstate = torch.load(cmin_path, map_location="cpu"); cmin = CMINSRv3Model(); cmin.load_state_dict(cstate["model_state_dict"] if isinstance(cstate, dict) and "model_state_dict" in cstate else cstate, strict=False); cmin.to(device).eval()
    sstate = torch.load(cal_path, map_location="cpu"); cal = SpectralGeometryCalibrator(); cal.load_state_dict(sstate["model_state_dict"], strict=False); cal.to(device).eval()
    rows = []
    with torch.no_grad():
        for ti, T in enumerate(args.t_eval):
            ds = generate_spectral_representation_dataset(SpectralRepresentationDatasetConfig(length=T, num_samples=args.num_samples, seed=args.seed + 17*ti, mrw_ratio=.20, low_lambda2_mrw_ratio=.10))
            for i in range(args.num_samples):
                x = torch.tensor(ds["x"][i:i+1], dtype=torch.float32, device=device)
                out = cmin(x)
                cout = cal(
                    out["zeta_emp"], out["zeta_mono"], out["zeta_mrw"], out["mono_residual_norm"], out["residual_norm"],
                    out["mrw_vs_mono_gain"], out["normalized_mrw_gain"], out["curvature_score"], out["linearity_score"], out["boundary_mrw_score"], out["tail_instability"],
                )
                proc = PROCESS_CODE_TO_NAME[int(ds["process_code"][i])]
                rows.append({
                    "T": T, "process_type": proc, "true_lambda2": float(ds["lambda2_true"][i,0]),
                    "p_scaling": float(out["p_scaling"].item()), "p_curved": float(out["p_curved"].item()), "p_mono": float(out["p_mono"].item()), "p_mrw": float(out["p_mrw"].item()), "p_boundary": float(out["boundary_mrw_score"].item()),
                    "p_scaling_cal": float(cout["p_scaling"].item()), "p_curved_cal": float(cout["p_curved"].item()), "p_mono_cal": float(cout["p_mono"].item()), "p_mrw_cal": float(cout["p_mrw"].item()), "p_boundary_cal": float(cout["p_boundary"].item()),
                    "tail_instability": float(out["tail_instability"].item()),
                })
    df = pd.DataFrame(rows); df.to_csv(TABLE_DIR / "predictions.csv", index=False)
    summary = df.groupby(["T","process_type"]).mean(numeric_only=True).reset_index(); summary.to_csv(TABLE_DIR / "process_by_T.csv", index=False)
    plot = summary[summary["T"] == 1024] if (summary["T"] == 1024).any() else summary
    fig, ax = plt.subplots(figsize=(8,4.8), constrained_layout=True)
    focus = plot[plot["process_type"].isin(["MRW","fGn","iid Gaussian","iid Student-t","Regime-switching Gaussian"])]
    x = np.arange(len(focus)); ax.bar(x-.18, focus["p_mrw"], width=.36, label="raw"); ax.bar(x+.18, focus["p_mrw_cal"], width=.36, label="spectrum-cal")
    ax.set_xticks(x); ax.set_xticklabels(focus["process_type"], rotation=25, ha="right"); ax.set_ylabel("p_MRW"); ax.legend(); ax.set_title("Raw CMIN-SR before/after spectrum calibration")
    fig.savefig(FIG_DIR / "pmrw_before_after.png", dpi=220); plt.close(fig)
    report = REPORT_DIR / "cmin_sr_spectrum_calibrated_eval_summary.md"
    report.write_text("\n".join(["# CMIN-SR + Spectrum-Space Calibrator", "", summary.to_csv(index=False)]), encoding="utf-8")
    meta = {"predictions": str((TABLE_DIR/"predictions.csv").relative_to(ROOT)), "process_by_T": str((TABLE_DIR/"process_by_T.csv").relative_to(ROOT)), "report": str(report.relative_to(ROOT))}
    (REPORT_DIR/"cmin_sr_spectrum_calibrated_eval_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8"); print(json.dumps(meta, indent=2))


if __name__ == "__main__": main()
