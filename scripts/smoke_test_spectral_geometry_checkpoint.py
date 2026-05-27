from __future__ import annotations
import json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from mrw_inverse.models import SpectralGeometryCalibrator

def main():
    ckpt = ROOT/"checkpoints"/"cmin"/"spectral_geometry_calibrator.pt"
    if not ckpt.exists(): result = {"status":"missing_checkpoint"}
    else:
        state = torch.load(ckpt, map_location="cpu"); m = SpectralGeometryCalibrator(); m.load_state_dict(state["model_state_dict"], strict=False)
        result = {"status":"ok","checkpoint":str(ckpt.relative_to(ROOT))}
    p = ROOT/"outputs"/"reports"/"spectral_geometry_checkpoint_smoke_test.json"; p.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))
if __name__ == "__main__": main()
