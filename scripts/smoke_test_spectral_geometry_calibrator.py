from __future__ import annotations
import json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from mrw_inverse.data import AnalyticSpectrumDatasetConfig, generate_analytic_spectrum_dataset
from mrw_inverse.models import SpectralGeometryCalibrator

def main():
    ds = generate_analytic_spectrum_dataset(AnalyticSpectrumDatasetConfig(num_samples=8, seed=2))
    model = SpectralGeometryCalibrator()
    args = [torch.tensor(ds[k], dtype=torch.float32) for k in ["zeta_input","zeta_mono","zeta_mrw","mono_residual_norm","mrw_residual_norm","mrw_vs_mono_gain","normalized_mrw_gain","curvature_score","linearity_score","boundary_score","target_tail_instability"]]
    with torch.no_grad(): out = model(*args)
    result = {"status":"ok","keys":list(out.keys()),"p_curved_range":[float(out["p_curved"].min()), float(out["p_curved"].max())]}
    p = ROOT/"outputs"/"reports"/"spectral_geometry_calibrator_smoke_test.json"; p.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))
if __name__ == "__main__": main()
