from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def check(path: str) -> bool:
    return (ROOT / path).exists()


def main() -> int:
    modules = [
        "mrw_inverse.models.robust_zeta_estimator",
        "mrw_inverse.models.mrw_projection",
        "mrw_inverse.models.monofractal_projection",
        "mrw_inverse.models.spectral_geometry_calibrator",
        "mrw_inverse.analysis.curvature_identifiability",
    ]
    imported = []
    for module in modules:
        importlib.import_module(module)
        imported.append(module)

    wrappers = [
        "experiments/paper/run_exp1_spectral_geometry_calibration.py",
        "experiments/paper/run_exp2_process_family_diagnostics.py",
        "experiments/paper/run_exp3_boundary_projection.py",
        "experiments/paper/run_exp4_finite_sample_identifiability.py",
        "experiments/paper/run_exp5_real_world_sanity_check.py",
        "experiments/paper/collect_paper_assets.py",
        "experiments/paper/generate_latex_tables.py",
        "experiments/paper/generate_paper_figures.py",
        "experiments/paper/generate_all_paper_assets.py",
    ]
    missing_wrappers = [p for p in wrappers if not check(p)]
    if missing_wrappers:
        raise FileNotFoundError(missing_wrappers)

    for name in ("paper_assets/figures", "paper_assets/tables", "paper_assets/summaries", "paper_assets/latex"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)

    commands = [
        [sys.executable, "experiments/paper/generate_latex_tables.py", "--quick"],
        [sys.executable, "experiments/paper/generate_paper_figures.py", "--quick"],
    ]
    results = []
    for cmd in commands:
        completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        results.append({"cmd": " ".join(cmd), "returncode": completed.returncode})
        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)
            raise SystemExit(completed.returncode)

    summary = {
        "status": "ok",
        "imported_modules": imported,
        "wrapper_count": len(wrappers),
        "paper_assets_exists": check("paper_assets"),
        "commands": results,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
