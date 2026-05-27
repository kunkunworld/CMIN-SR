from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from mrw_inverse.data import AnalyticSpectrumDatasetConfig, generate_analytic_spectrum_dataset

def main():
    ds = generate_analytic_spectrum_dataset(AnalyticSpectrumDatasetConfig(num_samples=80, seed=1))
    result = {"status":"ok","shape":list(ds["zeta_input"].shape),"types":sorted(set(ds["spectrum_type"].tolist())),"has_nan":bool(np.isnan(ds["zeta_input"]).any())}
    p = ROOT/"outputs"/"reports"/"analytic_spectrum_dataset_smoke_test.json"; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))
if __name__ == "__main__": main()
