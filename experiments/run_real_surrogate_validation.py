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

from mrw_inverse.proxy import estimate_to_dict, estimate_window
from mrw_inverse.surrogates import block_shuffled_surrogate, phase_randomized_surrogate, shuffled_surrogate

INPUT_RETURNS = ROOT / "data" / "real_processed" / "all_processed_returns.csv"
BASE_REPORT_DIR = ROOT / "outputs" / "reports"
BASE_TABLE_DIR = ROOT / "outputs" / "tables"
BASE_FIG_DIR = ROOT / "outputs" / "figures"


def _display_path(path: Path) -> str:
    try:
        resolved = path.resolve()
        if resolved == ROOT or ROOT in resolved.parents:
            return str(resolved.relative_to(ROOT))
    except Exception:
        pass
    return str(path)


def window_iter(series: np.ndarray, window: int, step: int):
    for end in range(window, len(series) + 1, step):
        yield end - 1, series[end - window : end]


def _interpret(gap_summary: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if gap_summary.empty:
        return ["- No surrogate gap summary rows were produced."]
    for _, row in gap_summary.iterrows():
        asset = row["source_name"]
        lgap = row["mean_lambda2_gap_shuffle"]
        pgap = row["mean_p_MRW_gap_shuffle"]
        fgap = row["mean_f_width_gap_shuffle"]
        if lgap > 0.005:
            lines.append(f"- `{asset}`: original windows show a clearly positive lambda2 gap over shuffled surrogates ({lgap:.4f}), which is consistent with temporal multiscale dependence beyond marginal distribution.")
        elif lgap > 0.0:
            lines.append(f"- `{asset}`: original windows show only a small positive lambda2 gap over shuffled surrogates ({lgap:.4f}); this is directionally supportive but conservative.")
        else:
            lines.append(f"- `{asset}`: lambda2 gap over shuffled surrogates is weak or negative ({lgap:.4f}), so strong intermittency is not clearly activated in this series.")

        if pgap > 0.02:
            lines.append(f"  - The MRW-validity score is also higher in originals than shuffled windows ({pgap:.4f}), supporting a dependence-based interpretation.")
        elif pgap <= 0.0:
            lines.append(f"  - The MRW-validity gap is not positive ({pgap:.4f}), so the surrogate evidence should be treated cautiously.")

        if fgap > 0.0:
            lines.append(f"  - Spectrum width is wider in originals by about {fgap:.4f}, which is qualitatively aligned with stronger multiscale structure.")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Run original vs surrogate validation on real return series.")
    parser.add_argument("--input", default=str(INPUT_RETURNS))
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--step", type=int, default=40)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", choices=["proxy", "auto", "model"], default="auto")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--include-phase-randomized", action="store_true")
    parser.add_argument("--max-series", type=int, default=8)
    parser.add_argument("--series-names", default=None, help="Optional comma-separated source_name list to evaluate.")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    output_tag = args.output_tag or ("real_surrogate_validation_cmin" if args.mode in {"auto", "model"} and args.checkpoint else "real_surrogate_validation_proxy")
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
        (REPORT_DIR / "real_surrogate_validation_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return

    input_path = Path(args.input)
    if not input_path.exists():
        out = {
            "status": "missing_input",
            "needed_input": _display_path(input_path),
            "note": "Processed real return file not found. Run preprocess_real_csvs.py first.",
        }
        (REPORT_DIR / "real_surrogate_validation_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return

    rng = np.random.default_rng(args.seed)
    returns = pd.read_csv(input_path, parse_dates=["date"])
    if args.series_names:
        requested = [x.strip() for x in args.series_names.split(",") if x.strip()]
        available = set(returns["source_name"].unique().tolist())
        series_names = [name for name in requested if name in available]
    else:
        series_names = sorted(returns["source_name"].unique().tolist())[: args.max_series]
    rows = []
    for source_name in series_names:
        series = returns[returns["source_name"] == source_name].sort_values("date")
        x = series["return"].to_numpy(dtype=np.float64)
        dates = series["date"].to_numpy()
        if len(x) < args.window:
            continue
        for sample_id, (end_idx, window_vals) in enumerate(window_iter(x, args.window, args.step)):
            original = window_vals.copy()
            shuffled = shuffled_surrogate(window_vals, rng)
            block = block_shuffled_surrogate(window_vals, args.block_size, rng)
            variants = {
                "original": original,
                "shuffled": shuffled,
                "block_shuffled": block,
            }
            if args.include_phase_randomized:
                variants["phase_randomized"] = phase_randomized_surrogate(window_vals, rng)
            for variant_name, arr in variants.items():
                est = estimate_window(arr, checkpoint_path=args.checkpoint, mode=args.mode)
                rows.append(
                    {
                        "source_name": source_name,
                        "window_end": pd.Timestamp(dates[end_idx]),
                        "window_index": sample_id,
                        "variant": variant_name,
                        "T": args.window,
                        **estimate_to_dict(est),
                    }
                )
                if args.model_name:
                    rows[-1]["model_name"] = args.model_name

    df = pd.DataFrame(rows)
    detail_path = TABLE_DIR / "real_surrogate_window_metrics.csv"
    df.to_csv(detail_path, index=False)

    wide = df.pivot_table(
        index=["source_name", "window_end", "window_index", "T"],
        columns="variant",
        values=["pred_lambda2", "p_MRW", "f_alpha_width", "logvol_cov_slope", "pred_H"],
        aggfunc="first",
    )
    wide.columns = ["_".join(col) for col in wide.columns]
    wide = wide.reset_index()
    for metric in ["pred_lambda2", "p_MRW", "f_alpha_width", "logvol_cov_slope", "pred_H"]:
        if f"{metric}_original" in wide.columns and f"{metric}_shuffled" in wide.columns:
            wide[f"{metric}_gap_shuffle"] = wide[f"{metric}_original"] - wide[f"{metric}_shuffled"]
        if f"{metric}_original" in wide.columns and f"{metric}_block_shuffled" in wide.columns:
            wide[f"{metric}_gap_block"] = wide[f"{metric}_original"] - wide[f"{metric}_block_shuffled"]
        if f"{metric}_original" in wide.columns and f"{metric}_phase_randomized" in wide.columns:
            wide[f"{metric}_gap_phase"] = wide[f"{metric}_original"] - wide[f"{metric}_phase_randomized"]
    gap_path = TABLE_DIR / "real_surrogate_gap_table.csv"
    wide.to_csv(gap_path, index=False)

    agg_spec = {"n_windows": ("window_index", "count")}
    optional_gaps = {
        "mean_lambda2_gap_shuffle": "pred_lambda2_gap_shuffle",
        "mean_p_MRW_gap_shuffle": "p_MRW_gap_shuffle",
        "mean_f_width_gap_shuffle": "f_alpha_width_gap_shuffle",
        "mean_logvol_gap_shuffle": "logvol_cov_slope_gap_shuffle",
        "mean_lambda2_gap_block": "pred_lambda2_gap_block",
        "mean_p_MRW_gap_block": "p_MRW_gap_block",
        "mean_f_width_gap_block": "f_alpha_width_gap_block",
        "mean_logvol_gap_block": "logvol_cov_slope_gap_block",
        "mean_lambda2_gap_phase": "pred_lambda2_gap_phase",
        "mean_p_MRW_gap_phase": "p_MRW_gap_phase",
        "mean_f_width_gap_phase": "f_alpha_width_gap_phase",
        "mean_logvol_gap_phase": "logvol_cov_slope_gap_phase",
    }
    for out_name, col in optional_gaps.items():
        if col in wide.columns:
            agg_spec[out_name] = (col, "mean")
    gap_summary = wide.groupby("source_name").agg(**agg_spec).reset_index()
    summary_path = TABLE_DIR / "real_surrogate_gap_summary.csv"
    gap_summary.to_csv(summary_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    wide.boxplot(column="pred_lambda2_gap_shuffle", by="source_name", ax=axes[0], grid=False, rot=35)
    axes[0].set_title("lambda2 original - shuffled")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("gap")
    wide.boxplot(column="p_MRW_gap_shuffle", by="source_name", ax=axes[1], grid=False, rot=35)
    axes[1].set_title("p_MRW original - shuffled")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("gap")
    fig.suptitle("Real surrogate validation")
    fig_path = FIG_DIR / "real_surrogate_gap_boxplots.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    report_lines = [
        "# Real Surrogate Validation Summary",
        "",
        f"- Input returns: `{_display_path(input_path)}`",
        f"- Window: `{args.window}`",
        f"- Step: `{args.step}`",
        f"- Block size: `{args.block_size}`",
        f"- Mode: `{df['mode'].iloc[0] if not df.empty else 'none'}`",
        f"- Model name: `{df['model_name'].iloc[0] if not df.empty else 'none'}`",
        f"- Checkpoint: `{df['checkpoint_path'].iloc[0] if not df.empty else ''}`",
        f"- Detail CSV: `{detail_path.relative_to(ROOT)}`",
        f"- Gap table: `{gap_path.relative_to(ROOT)}`",
        f"- Gap summary: `{summary_path.relative_to(ROOT)}`",
        f"- Figure: `{fig_path.relative_to(ROOT)}`",
        "",
        "## Gap Summary Table",
        "",
        gap_summary.to_csv(index=False),
        "",
        "## Cautious Interpretation",
        "",
    ]
    report_lines.extend(_interpret(gap_summary))
    report_lines.extend(
        [
            "",
            "## Interpretation Policy",
            "",
            "- Fama-French factor returns are cleaned, aggregated, economically constructed return series.",
            "- Small original-vs-shuffled gaps should be interpreted as conservative weak-intermittency evidence, not as a failure of the framework.",
            "- These surrogate tests are more important than downstream forecasting for validating whether lambda2 responds to temporal dependence rather than only marginal heavy tails.",
            "- Phase-randomized surrogates are not implemented in this minimal runnable version.",
        ]
    )
    report_path = REPORT_DIR / "real_surrogate_validation_summary.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    meta = {
        "detail_csv": str(detail_path.relative_to(ROOT)),
        "gap_table": str(gap_path.relative_to(ROOT)),
        "gap_summary": str(summary_path.relative_to(ROOT)),
        "figure": str(fig_path.relative_to(ROOT)),
        "report": str(report_path.relative_to(ROOT)),
        "mode": df["mode"].iloc[0] if not df.empty else "none",
        "model_name": df["model_name"].iloc[0] if not df.empty else "none",
        "checkpoint_path": df["checkpoint_path"].iloc[0] if not df.empty else "",
        "phase_randomized": bool(args.include_phase_randomized),
    }
    (REPORT_DIR / "real_surrogate_validation_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
