from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_dl.spectrum_baseline import SpectrumTrainConfig, train_spectrum_baseline


def main() -> None:
    model_name = "resnet"
    if len(sys.argv) >= 2:
        model_name = sys.argv[1]

    scale_models = {"scale_cnn", "scale_invariant_cnn", "physics_scale_net", "psn"}
    standardization = "global" if model_name in scale_models else "pointwise"
    target_mode = "zeta_aux" if model_name in {"zeta_physics_hybrid"} else "params"
    identifiability_kwargs = {}
    if model_name == "lmmi_ident":
        identifiability_kwargs = {
            "identifiability_loss_weight": 0.08,
            "identifiability_mode": "matched_h_lambda",
        }
    elif model_name == "lmmi_ident_h_control":
        identifiability_kwargs = {
            "identifiability_loss_weight": 0.08,
            "identifiability_mode": "matched_lambda_h",
        }
    config = SpectrumTrainConfig(
        model_name=model_name,
        output_dir=f"outputs/dl_spectrum_{model_name}",
        standardization=standardization,
        target_mode=target_mode,
        **identifiability_kwargs,
    )
    summary = train_spectrum_baseline(config)
    print(json.dumps({
        "model_name": model_name,
        "best_val_loss": summary["best_val_loss"],
        "parameter_metrics": summary["parameter_metrics"],
        "spectrum_metrics": summary["spectrum_metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
