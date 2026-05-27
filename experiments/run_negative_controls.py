from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_inverse.data.process_generators import generate_process_sample
from mrw_inverse.proxy import estimate_to_dict, estimate_window

BASE_REPORT_DIR = ROOT / "outputs" / "reports"
BASE_TABLE_DIR = ROOT / "outputs" / "tables"
BASE_FIG_DIR = ROOT / "outputs" / "figures"
def _interpret(summary: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    get = lambda name, col: float(summary.loc[summary["process_type"] == name, col].iloc[0]) if (summary["process_type"] == name).any() else float("nan")

    mrw_lambda = get("MRW", "mean_pred_lambda2")
    shuf_lambda = get("Shuffled MRW", "mean_pred_lambda2")
    student_lambda = get("iid Student-t", "mean_pred_lambda2")
    gauss_lambda = get("iid Gaussian", "mean_pred_lambda2")
    fgn_lambda = get("fGn", "mean_pred_lambda2")
    garch_lambda = get("GARCH(1,1)", "mean_pred_lambda2")

    if np.isfinite(mrw_lambda) and np.isfinite(shuf_lambda):
        gap = mrw_lambda - shuf_lambda
        if gap > 0.01:
            lines.append(f"- MRW vs shuffled MRW: lambda2 drops by about {gap:.4f} after shuffling, supporting the interpretation that temporal dependence contributes materially to inferred intermittency.")
        elif gap > 0.0:
            lines.append(f"- MRW vs shuffled MRW: lambda2 decreases slightly after shuffling (gap about {gap:.4f}), which is directionally consistent but not a large separation.")
        else:
            lines.append("- MRW vs shuffled MRW: lambda2 does not clearly decrease after shuffling in this run, so the temporal-dependence signal is weak under the current proxy setup.")

    if np.isfinite(student_lambda) and np.isfinite(gauss_lambda):
        if student_lambda <= gauss_lambda + 0.01:
            lines.append("- Heavy-tail stress test: iid Student-t does not generate dramatically larger lambda2 than iid Gaussian, which suggests the estimator is not simply equating heavy tails with multifractality.")
        else:
            lines.append("- Heavy-tail stress test: iid Student-t produces noticeably larger lambda2 than iid Gaussian, so some heavy-tail sensitivity remains in proxy mode and should not be over-interpreted.")

    if np.isfinite(fgn_lambda):
        if fgn_lambda < 0.03:
            lines.append("- fGn control: nontrivial H can be recovered while lambda2 stays relatively low, consistent with long memory without strong intermittency.")
        else:
            lines.append("- fGn control: lambda2 is not especially low, so the proxy still confounds roughness and intermittency in some long-memory settings.")

    if np.isfinite(garch_lambda):
        lines.append(f"- GARCH ambiguity: GARCH(1,1) mean lambda2 is {garch_lambda:.4f}. This should be interpreted as a stress test for volatility clustering rather than proof of MRW behavior.")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Run non-MRW negative controls with proxy/CMIN fallback.")
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", choices=["proxy", "auto", "model"], default="auto")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    output_tag = args.output_tag or ("negative_controls_cmin" if args.mode in {"auto", "model"} and args.checkpoint else "negative_controls_proxy")
    REPORT_DIR = BASE_REPORT_DIR / output_tag
    TABLE_DIR = BASE_TABLE_DIR / output_tag
    FIG_DIR = BASE_FIG_DIR / output_tag
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "model" and (args.checkpoint is None or not Path(args.checkpoint).exists()):
        out = {
            "status": "missing_checkpoint",
            "mode": args.mode,
            "checkpoint": args.checkpoint,
            "note": "Model mode requires an existing CMIN checkpoint.",
        }
        (REPORT_DIR / "negative_controls_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return

    rng = np.random.default_rng(args.seed)
    rows = []
    process_names = ["MRW", "Shuffled MRW", "fGn", "iid Gaussian", "iid Student-t", "GARCH(1,1)", "Regime-switching Gaussian"]
    for process_name in process_names:
        for sample_id in range(args.num_samples):
            if process_name in {"MRW", "Shuffled MRW"}:
                true_h = float(rng.uniform(0.3, 0.8))
                true_lambda2 = float(rng.uniform(0.04, 0.12))
                dx = generate_process_sample(process_name, args.length, rng, h=true_h, lambda2=true_lambda2).x.astype(np.float64)
            elif process_name == "fGn":
                true_h = float(rng.uniform(0.3, 0.8))
                true_lambda2 = np.nan
                dx = generate_process_sample("fGn", args.length, rng, h=true_h).x.astype(np.float64)
            else:
                true_h = np.nan
                true_lambda2 = np.nan
                dx = generate_process_sample(process_name, args.length, rng).x.astype(np.float64)

            est = estimate_window(dx, checkpoint_path=args.checkpoint, mode=args.mode)
            row = {
                "process_type": process_name,
                "sample_id": sample_id,
                "T": args.length,
                "true_H": true_h,
                "true_lambda2": true_lambda2,
                **estimate_to_dict(est),
            }
            if args.model_name:
                row["model_name"] = args.model_name
            rows.append(row)

    df = pd.DataFrame(rows)
    detail_path = TABLE_DIR / "negative_controls_samples.csv"
    df.to_csv(detail_path, index=False)

    summary = (
        df.groupby("process_type")
        .agg(
            n_samples=("sample_id", "count"),
            mean_pred_H=("pred_H", "mean"),
            std_pred_H=("pred_H", "std"),
            mean_pred_lambda2=("pred_lambda2", "mean"),
            std_pred_lambda2=("pred_lambda2", "std"),
            mean_p_MRW=("p_MRW", "mean"),
            mean_residual_norm=("residual_norm", "mean"),
            mean_logvol_slope=("logvol_cov_slope", "mean"),
            mean_zeta_curvature=("empirical_zeta_curvature", "mean"),
            mean_f_alpha_width=("f_alpha_width", "mean"),
            lambda2_boundary_rate=("lambda2_boundary_hit", "mean"),
        )
        .reset_index()
    )
    summary_path = TABLE_DIR / "negative_controls_summary.csv"
    summary.to_csv(summary_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    df.boxplot(column="pred_lambda2", by="process_type", ax=axes[0], grid=False, rot=35)
    axes[0].set_title("Predicted lambda2 by process")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("pred_lambda2")
    df.boxplot(column="p_MRW", by="process_type", ax=axes[1], grid=False, rot=35)
    axes[1].set_title("MRW-validity score by process")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("p_MRW")
    fig.suptitle("Negative controls")
    fig_path = FIG_DIR / "negative_controls_boxplots.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    interpretation = _interpret(summary)
    report_lines = [
        "# Negative Controls Summary",
        "",
        f"- Mode: `{df['mode'].iloc[0]}`",
        f"- Model name: `{df['model_name'].iloc[0]}`",
        f"- Checkpoint: `{df['checkpoint_path'].iloc[0]}`",
        f"- Samples per process: `{args.num_samples}`",
        f"- Sequence length: `{args.length}`",
        f"- Detail CSV: `{detail_path.relative_to(ROOT)}`",
        f"- Summary CSV: `{summary_path.relative_to(ROOT)}`",
        f"- Figure: `{fig_path.relative_to(ROOT)}`",
        "",
        "## Aggregate Table",
        "",
        summary.to_csv(index=False),
        "",
        "## Cautious Interpretation",
        "",
    ]
    report_lines.extend(interpretation if interpretation else ["- The current run produced no strong automatic interpretation signal."])
    report_lines.extend(
        [
            "",
            "## Interpretation Template Notes",
            "",
            "- MRW should have higher lambda2 and wider spectra than simple null processes.",
            "- Shuffled MRW is expected to reduce lambda2 because temporal dependence is destroyed.",
            "- Student-t tests whether heavy tails alone can fool the estimator.",
            "- fGn tests whether long-memory slope can be separated from intermittency.",
            "- GARCH and regime-switching Gaussian are deliberately ambiguous controls and should be interpreted carefully.",
        ]
    )
    report_path = REPORT_DIR / "negative_controls_summary.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    meta = {
        "detail_csv": str(detail_path.relative_to(ROOT)),
        "summary_csv": str(summary_path.relative_to(ROOT)),
        "figure": str(fig_path.relative_to(ROOT)),
        "report": str(report_path.relative_to(ROOT)),
        "mode": df["mode"].iloc[0],
        "model_name": df["model_name"].iloc[0],
        "checkpoint_path": df["checkpoint_path"].iloc[0],
    }
    (REPORT_DIR / "negative_controls_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
