from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAPER_ASSETS = ROOT / "paper_assets"
TABLE_DIR = PAPER_ASSETS / "tables"
FIG_DIR = PAPER_ASSETS / "figures"
SUMMARY_DIR = PAPER_ASSETS / "summaries"


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save_fig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(FIG_DIR / f"{name}.png", dpi=240)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)


def _mrw_zeta(q: np.ndarray, h: float, lambda2: float) -> np.ndarray:
    return q * h - 0.5 * lambda2 * q * (q - 2.0)


def _legendre(q: np.ndarray, zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.gradient(zeta, q)
    f_alpha = q * alpha - zeta + 1.0
    return alpha, f_alpha


def generate_baseline_summary() -> pd.DataFrame:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    improved = _load_json(ROOT / "outputs" / "baselines" / "baseline_results_robust_improved_10.json")
    if isinstance(improved, list) and improved:
        metrics = [item.get("metrics", {}) for item in improved]
        for key in ["zeta_mae_sf", "zeta_mae_mfdfa", "spectrum_mae_sf", "spectrum_mae_mfdfa"]:
            vals = [float(m[key]) for m in metrics if key in m]
            if vals:
                rows.append(
                    {
                        "evidence_block": "single_sample_classical_baseline",
                        "method_metric": key,
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "n": len(vals),
                        "source": "outputs/baselines/baseline_results_robust_improved_10.json",
                    }
                )

    ensemble = _load_json(ROOT / "outputs" / "baselines" / "baseline_ensemble_result.json")
    if isinstance(ensemble, dict):
        for key, value in ensemble.get("metrics", {}).items():
            rows.append(
                {
                    "evidence_block": "ensemble_high_lambda_mrw_baseline",
                    "method_metric": key,
                    "mean": float(value),
                    "std": np.nan,
                    "n": int(ensemble.get("num_realizations", 0)),
                    "source": "outputs/baselines/baseline_ensemble_result.json",
                }
            )

    metrics_paths = {
        "unconstrained_mlp": ROOT / "outputs" / "dl_spectrum_mlp" / "metrics.json",
        "unconstrained_cnn": ROOT / "outputs" / "dl_spectrum_cnn" / "metrics.json",
        "pc_smin_constrained": ROOT / "outputs" / "dl_spectrum_pc_smin" / "metrics.json",
        "final_hybrid_constrained": ROOT / "outputs" / "dl_spectrum_final_hybrid" / "metrics.json",
    }
    for name, path in metrics_paths.items():
        data = _load_json(path)
        if not isinstance(data, dict):
            continue
        for family, metrics in [
            ("parameter", data.get("parameter_metrics", {})),
            ("spectrum", data.get("spectrum_metrics", {})),
        ]:
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    for sub_name, sub_value in metric_value.items():
                        rows.append(
                            {
                                "evidence_block": name,
                                "method_metric": f"{family}_{metric_name}_{sub_name}",
                                "mean": float(sub_value),
                                "std": np.nan,
                                "n": np.nan,
                                "source": str(path.relative_to(ROOT)).replace("\\", "/"),
                            }
                        )
                elif isinstance(metric_value, (int, float)):
                    rows.append(
                        {
                            "evidence_block": name,
                            "method_metric": f"{family}_{metric_name}",
                            "mean": float(metric_value),
                            "std": np.nan,
                            "n": np.nan,
                            "source": str(path.relative_to(ROOT)).replace("\\", "/"),
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "todo_supplemental_baseline_summary.csv", index=False)
    return df


def generate_spectrum_shape_figure() -> None:
    q = np.linspace(0.5, 3.0, 51)
    cases = [
        ("linear monofractal", 0.60, 0.00, "#386cb0"),
        ("boundary MRW", 0.60, 0.02, "#fdb462"),
        ("curved MRW", 0.60, 0.12, "#7fc97f"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    for label, h, lam, color in cases:
        zeta = _mrw_zeta(q, h, lam)
        alpha, f_alpha = _legendre(q, zeta)
        axes[0].plot(q, zeta, label=label, color=color, lw=2)
        axes[1].plot(q, alpha, label=label, color=color, lw=2)
        axes[2].plot(alpha, f_alpha, label=label, color=color, lw=2)

    axes[0].set_xlabel(r"$q$")
    axes[0].set_ylabel(r"$\zeta(q)$")
    axes[1].set_xlabel(r"$q$")
    axes[1].set_ylabel(r"$\alpha(q)$")
    axes[2].set_xlabel(r"$\alpha$")
    axes[2].set_ylabel(r"$f(\alpha)$")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Analytic Monofractal and MRW Spectrum Geometry", y=0.98, fontsize=15)
    _save_fig(fig, "fig7_multifractal_spectrum_shapes")


def write_status_files(df: pd.DataFrame) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# TODO Evidence Completion Status",
        "",
        "This file records which missing-evidence items were addressed using existing project scripts and outputs. No new CMIN-SR model, validity head, or backbone was trained.",
        "",
        "| TODO | Status | Evidence / action | Remaining risk |",
        "|---|---|---|---|",
        "| Classical MFDFA / structure-function baseline comparison | Partially completed | Ran `scripts/run_baselines_improved.py` and `scripts/run_baselines_ensemble.py`; see `paper_assets/tables/todo_supplemental_baseline_summary.csv`. | WTMM/wavelet-leader comparison is still missing. |",
        "| Direct neural spectrum regression baseline | Completed from historical outputs | Summarized `outputs/dl_spectrum_mlp`, `outputs/dl_spectrum_cnn`, and constrained PC-SMIN/final-hybrid metrics. | Historical split, not a newly standardized CMIN-SR final split. |",
        "| Unconstrained H/lambda2 regression baseline | Completed from historical outputs | Same summary table includes unconstrained MLP/CNN parameter metrics and constrained PC-SMIN/final-hybrid metrics. | Use as appendix/ablation evidence, not main claim. |",
        "| f(alpha) reconstruction visualization | Completed | Generated `paper_assets/figures/fig7_multifractal_spectrum_shapes.{png,pdf}`. | Analytic spectrum-shape figure, not empirical reconstruction accuracy. |",
        "| Multiple random seed stability | Partially completed | Key scripts have seed controls and quick supplemental runs were refreshed at seed 2026. | Full 3-5 seed aggregation remains recommended before final submission. |",
        "| Real-world complex-system validation | Assessed, not promoted to main | Existing Fama-French data are present, but current paper claim is diagnostic/identifiability; real data are useful as sanity check, not required to prove the core synthetic identifiability result. | CSF reviewers may still prefer one cautious real-world example. |",
        "| Scale-length sensitivity | Completed quick refresh | `outputs/tables/finite_sample_identifiability/lambda2_recovery_by_scale_range.csv`. | Full grid with larger T and more samples would strengthen it. |",
        "| q-grid sensitivity | Completed quick refresh | `outputs/tables/finite_sample_identifiability/lambda2_recovery_by_qgrid.csv`. | Full Q1-Q5 grid with more samples would strengthen it. |",
        "| Zeta-space noise bridge | Completed refresh | `outputs/tables/zeta_noise_bridge/separation_margin_vs_noise.csv`. | Raw signal noise perturbation remains optional. |",
        "| Statistical confidence intervals | Partially completed | Baseline table includes mean/std where sample-level data are available. | Bootstrap CIs for all main diagnostics remain optional polish. |",
        "| Abstract numerical source comments | Improved | Refreshed tables for spectral geometry, noise bridge, failure attribution, identifiability. | Raw zeta alignment before/after numbers should stay sourced to comparison CSV/docs. |",
        "| Diagnostic framing vs parameter recovery | Completed as writing decision | Evidence supports diagnostic framing; avoid guaranteed short-window `lambda2` recovery claim. | Must keep language consistent in the manuscript. |",
        "",
        "## Baseline Evidence Summary",
        "",
        df.to_csv(index=False) if not df.empty else "No baseline rows were generated.",
    ]
    (SUMMARY_DIR / "todo_evidence_completion_status.md").write_text("\n".join(lines), encoding="utf-8")

    data_lines = [
        "# Real-World Data Necessity Assessment",
        "",
        "## Short answer",
        "",
        "A real financial or other complex-system dataset is not strictly required to support the paper's central claim if the paper is framed as a finite-sample spectral identifiability and diagnostic-method study. The core evidence is synthetic/controlled because the paper asks whether MRW-like curvature is identifiable when ground-truth `H` and `lambda2` are known.",
        "",
        "## Why real data still help",
        "",
        "- CSF readers often expect connection to real complex systems.",
        "- A cautious real-world sanity check can show that the pipeline runs on empirical data and reports instability/surrogate warnings.",
        "- It should not be used to claim that a market, turbulence signal, or physiological signal is truly generated by an MRW.",
        "",
        "## Existing local data",
        "",
        "- `data/real/F-F_Research_Data_Factors_daily.csv`",
        "- `data/real/F-F_Momentum_Factor_daily.csv`",
        "- processed returns under `data/real_processed/`",
        "",
        "## Recommendation",
        "",
        "Use real data only as an optional sanity-check appendix: original-vs-shuffled/surrogate diagnostics, residual geometry, and instability warnings. Do not make it a main proof. If the target venue or reviewer asks for stronger application evidence, then add one real-data section with careful caveats.",
        "",
        "## Data still needed if we choose to strengthen real-world evidence",
        "",
        "- One clearly described complex-system dataset with stable provenance, such as a market index/asset series, turbulence velocity increments, or physiological heartbeat intervals.",
        "- A preprocessing note specifying whether input is returns/increments or paths.",
        "- A surrogate construction protocol, e.g. shuffled returns and/or phase-randomized surrogates.",
        "- A statement that real-data results are diagnostic, not mechanism proof.",
    ]
    (SUMMARY_DIR / "real_world_data_necessity_assessment.md").write_text("\n".join(data_lines), encoding="utf-8")


def main() -> int:
    df = generate_baseline_summary()
    generate_spectrum_shape_figure()
    write_status_files(df)
    print(
        json.dumps(
            {
                "baseline_summary": str((TABLE_DIR / "todo_supplemental_baseline_summary.csv").relative_to(ROOT)),
                "spectrum_shapes": str((FIG_DIR / "fig7_multifractal_spectrum_shapes.png").relative_to(ROOT)),
                "status": str((SUMMARY_DIR / "todo_evidence_completion_status.md").relative_to(ROOT)),
                "real_world_assessment": str((SUMMARY_DIR / "real_world_data_necessity_assessment.md").relative_to(ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
