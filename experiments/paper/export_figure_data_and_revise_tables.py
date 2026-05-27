from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ASSET_TABLES = ROOT / "paper_assets" / "tables"
FIG_DATA = ROOT / "paper_assets" / "figure_data"
SUMMARIES = ROOT / "paper_assets" / "summaries"
LATEX_TABLES = ROOT / "paper_writing_workspace" / "latex_tables"


def _read(rel: str) -> pd.DataFrame:
    path = ROOT / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _fmt(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.3f}"


def _escape(s: object) -> str:
    return str(s).replace("_", "\\_")


def _booktabs_table(path: Path, caption: str, label: str, headers: list[str], rows: list[list[object]]) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{" + "l" * len(headers) + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_escape(v) for v in row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def export_fig1() -> None:
    structure = [
        "# Fig.1 Pipeline Structure",
        "",
        "## Nodes",
        "",
        "| id | label | function |",
        "|---|---|---|",
        "| A | Raw finite increments | Finite observed returns/increments; not assumed to reveal a mechanism directly. |",
        "| B | Empirical zeta estimation | Estimate structure-function slopes on a finite q-grid and scale set. |",
        "| C | Monofractal projection | Fit linear spectrum zeta(q)=qH and compute residual. |",
        "| D | MRW projection | Fit parabolic MRW-family spectrum and compute lambda2_proj and residual. |",
        "| E | Residual and geometry features | Compare MRW-vs-mono residuals, curvature, linearity, boundary, and instability. |",
        "| F | Calibrated diagnostic scores | Report p_scaling, p_curved, p_mono, p_MRW, p_boundary as diagnostic scores. |",
        "| G | Conservative interpretation | Organize evidence; do not claim proof of an MRW data-generating mechanism. |",
        "",
        "## Arrows",
        "",
        "A -> B -> {C,D} -> E -> F -> G.",
        "",
        "## Recommended layout",
        "",
        "Use a three-stage horizontal pipeline with subtle vertical grouping:",
        "Stage 1: finite increments to empirical zeta estimation.",
        "Stage 2: parallel monofractal and MRW projections feeding residual geometry.",
        "Stage 3: calibrated diagnostics feeding a warning/interpretable decision box.",
        "",
        "## Caption suggestion",
        "",
        "CMIN-SR diagnostic pipeline. Finite increments are first converted into an empirical scaling spectrum, then projected onto competing monofractal and MRW spectral families. Projection residuals, curvature, boundary behavior, and instability features are converted into diagnostic scores. The output organizes MRW-compatible spectral evidence but does not prove an MRW data-generating mechanism.",
    ]
    (FIG_DATA / "fig1_pipeline_structure.md").write_text("\n".join(structure), encoding="utf-8")
    mermaid = [
        "# Fig.1 Pipeline Mermaid",
        "",
        "```mermaid",
        "flowchart LR",
        "  A[\"Raw finite increments\"] --> B[\"Empirical zeta estimation\"]",
        "  B --> C[\"Monofractal projection<br/>zeta(q)=qH\"]",
        "  B --> D[\"MRW projection<br/>parabolic zeta(q)\"]",
        "  C --> E[\"Residual and geometry features\"]",
        "  D --> E",
        "  E --> F[\"Diagnostic scores<br/>scaling, curved, mono, MRW, boundary\"]",
        "  F --> G[\"Conservative interpretation<br/>evidence organization, not mechanism proof\"]",
        "```",
    ]
    (FIG_DATA / "fig1_pipeline_mermaid.md").write_text("\n".join(mermaid), encoding="utf-8")
    prompt = [
        "# Fig.1 Redesign Prompt",
        "",
        "Core message: CMIN-SR is a three-stage validity-aware diagnostic pipeline, not an MRW mechanism detector.",
        "",
        "Recommended layout: wide three-stage horizontal graphic.",
        "",
        "Stage 1: `Finite-sample spectrum estimation`",
        "- Raw finite increments",
        "- Structure-function / empirical zeta(q)",
        "- Note: finite T, q-grid, scale range",
        "",
        "Stage 2: `Competing spectral projections`",
        "- Monofractal linear projection",
        "- MRW-family parabolic projection",
        "- Residual geometry and lambda2_proj",
        "",
        "Stage 3: `Validity-aware diagnostics`",
        "- p_scaling, p_curved, p_mono, p_MRW, p_boundary",
        "- Tail / instability warning",
        "- Final note: organizes evidence; does not prove MRW mechanism",
        "",
        "Color guidance: neutral gray for data, blue for empirical spectrum, green for monofractal, orange for MRW, muted red for warnings. Avoid gradients and decorative icons.",
        "",
        "PowerPoint/Figma: use aligned rounded rectangles with thin strokes, grouped by stage bands, and one small warning callout at the end.",
        "Python matplotlib: use `FancyBboxPatch`, arrows, and stage background rectangles.",
        "TikZ: use three grouped nodes with parallel projection branches; keep text short.",
        "",
        "Caption: CMIN-SR converts finite increments into empirical scaling spectra, compares monofractal and MRW-family projections, and reports validity-aware diagnostic scores with instability warnings. The output is an evidence organization tool rather than proof of an MRW data-generating mechanism.",
    ]
    (FIG_DATA / "fig1_redesign_prompt.md").write_text("\n".join(prompt), encoding="utf-8")


def export_figure_data() -> None:
    FIG_DATA.mkdir(parents=True, exist_ok=True)
    SUMMARIES.mkdir(parents=True, exist_ok=True)
    export_fig1()

    ts = _read("paper_assets/tables/mrw_simulation_example_timeseries.csv")
    ts = ts.rename(columns={"t": "time", "omega": "omega_t", "dx": "increment", "x": "path"})
    ts[["time", "omega_t", "increment", "path", "H", "lambda2", "L", "seed"]].to_csv(FIG_DATA / "fig2_mrw_simulation_timeseries.csv", index=False)
    z = _read("paper_assets/tables/mrw_simulation_example_zeta.csv")
    z = z.rename(columns={"zeta_linear_lambda2_0": "zeta_linear", "zeta_boundary_lambda2_0p02": "zeta_boundary", "zeta_curved_lambda2": "zeta_curved"})
    z[["q", "zeta_linear", "zeta_boundary", "zeta_curved", "H", "lambda2_curved"]].to_csv(FIG_DATA / "fig2_mrw_simulation_zeta.csv", index=False)
    (FIG_DATA / "fig2_redraw_notes.md").write_text(
        "\n".join([
            "# Fig.2 Redraw Notes",
            "",
            "Recommended as a four-panel figure: omega(t), increments, integrated path, and analytic zeta(q).",
            "Use only the first 600--800 observations for time-series panels to keep detail readable.",
            "Keep line widths thin and avoid dense tick labels. This figure is useful in the methods section.",
        ]),
        encoding="utf-8",
    )

    q = z["q"].to_numpy(float)
    out = pd.DataFrame({"q": q})
    for name, col in [("linear", "zeta_linear"), ("boundary", "zeta_boundary"), ("curved", "zeta_curved")]:
        zz = z[col].to_numpy(float)
        alpha = np.gradient(zz, q)
        f_alpha = q * alpha - zz + 1.0
        out[f"zeta_{name}"] = zz
        out[f"alpha_{name}"] = alpha
        out[f"f_alpha_{name}"] = f_alpha
    out.to_csv(FIG_DATA / "fig3_analytic_spectrum_geometry.csv", index=False)
    (FIG_DATA / "fig3_redraw_notes.md").write_text(
        "Fig.3 can be kept separate as a theory/geometry figure. If page budget is tight, merge Fig.2 and Fig.3 by keeping Fig.2's MRW time-series panels and adding only one zeta(q) panel; move alpha/f(alpha) to appendix.",
        encoding="utf-8",
    )

    sg = _read("outputs/tables/spectral_geometry_calibrator_eval/predictions.csv")
    sg = sg.rename(columns={"spectrum_type": "sample_type", "p_mrw": "p_MRW"})
    sg["run_id"] = "latest_eval"
    sg[["sample_type", "p_curved", "p_MRW", "run_id"]].to_csv(FIG_DATA / "fig4_spectral_geometry_map.csv", index=False)

    ident = _read("paper_assets/tables/table4_finite_sample_identifiability.csv")
    ident[["T", "estimator", "lambda2_corr", "lambda2_mae", "high_lambda_detection_rate", "boundary_accuracy"]].to_csv(FIG_DATA / "fig5_lambda2_recovery.csv", index=False)

    noise = _read("paper_assets/tables/seed_stability_zeta_noise_bridge_by_seed.csv")
    noise = noise.rename(columns={"pmrw_gap_curved_minus_linear": "p_MRW_gap"})
    noise["perturbation_type"] = "mixed_zeta_perturbations"
    noise[["noise_level", "p_MRW_gap", "seed", "perturbation_type"]].to_csv(FIG_DATA / "fig6_zeta_noise_bridge.csv", index=False)

    attr = _read("paper_assets/tables/seed_stability_failure_attribution_by_seed.csv")
    attr = attr.rename(columns={"level": "spectrum_source", "p_mrw": "p_MRW"})
    attr[["process_type", "spectrum_source", "p_MRW", "p_curved", "zeta_mae", "seed"]].to_csv(FIG_DATA / "fig7_failure_attribution.csv", index=False)

    proj = _read("outputs/tables/cmin_sr_v3_eval/process_by_T.csv")
    rename = {
        "mean_residual_norm": "r_MRW",
        "mean_mono_residual_norm": "r_mono",
        "mean_lambda2_proj": "lambda2_proj",
        "mean_p_mrw": "p_MRW",
        "mean_p_MRW": "p_MRW",
        "mean_p_mono": "p_mono",
        "mean_boundary_mrw_score": "p_boundary",
    }
    proj = proj.rename(columns={k: v for k, v in rename.items() if k in proj.columns})
    cols = [c for c in ["T", "process_type", "r_MRW", "r_mono", "lambda2_proj", "p_MRW", "p_mono", "p_boundary"] if c in proj.columns]
    proj[cols].to_csv(FIG_DATA / "fig8_projection_residual_geometry.csv", index=False)

    wide = _read("outputs/tables/real_world_sanity_famafrench_factors_proxy/real_surrogate_gap_table.csv")
    rows = []
    metrics = [
        ("lambda2", "pred_lambda2_original", "pred_lambda2_shuffled", "pred_lambda2_gap_shuffle"),
        ("p_MRW", "p_MRW_original", "p_MRW_shuffled", "p_MRW_gap_shuffle"),
        ("f_alpha_width", "f_alpha_width_original", "f_alpha_width_shuffled", "f_alpha_width_gap_shuffle"),
        ("logvol_cov_slope", "logvol_cov_slope_original", "logvol_cov_slope_shuffled", "logvol_cov_slope_gap_shuffle"),
    ]
    for _, row in wide.iterrows():
        for metric, orig, shuf, gap in metrics:
            if orig in row.index and shuf in row.index:
                rows.append(
                    {
                        "factor": row["source_name"],
                        "window_id": int(row["window_index"]),
                        "time_index": row["window_end"],
                        "metric": metric,
                        "original_value": row[orig],
                        "shuffled_value": row[shuf],
                        "gap": row[orig] - row[shuf] if gap not in row.index else row[gap],
                    }
                )
    pd.DataFrame(rows).to_csv(FIG_DATA / "fig9_real_data_sanity_check.csv", index=False)


def revise_tables() -> None:
    ASSET_TABLES.mkdir(parents=True, exist_ok=True)
    LATEX_TABLES.mkdir(parents=True, exist_ok=True)

    proc = _read("paper_assets/tables/table1_process_family_diagnostics.csv")
    proc1024 = proc[proc["T"].eq(1024)].copy()
    order = ["MRW", "Low-lambda2 MRW", "fGn", "iid Gaussian", "iid Student-t", "GARCH(1,1)", "Regime-switching Gaussian"]
    compact = []
    for name in order:
        sub = proc1024[proc1024["process_type"].eq(name)]
        if sub.empty:
            continue
        if name == "MRW" and "high" in set(sub["lambda_band"]):
            sub = sub[sub["lambda_band"].eq("high")]
        row = sub.iloc[0]
        compact.append([row["process_type"], row["lambda_band"], _fmt(row["p_curved_cal"]), _fmt(row["p_mrw_cal"]), _fmt(row["p_mono_cal"]), _fmt(row["p_boundary_cal"])])
    pd.DataFrame(compact, columns=["process_type", "lambda_band", "p_curved_cal", "p_mrw_cal", "p_mono_cal", "p_boundary_cal"]).to_csv(ASSET_TABLES / "table1_process_family_diagnostics_compact.csv", index=False)
    _booktabs_table(
        LATEX_TABLES / "table1_process_family_diagnostics.tex",
        "Process-family diagnostics at \\(T=1024\\). Scores should be read jointly; high scaling alone is not MRW evidence.",
        "tab:table1_process_family_diagnostics",
        ["Process", "Band", "\\(p_c\\)", "\\(p_{MRW}\\)", "\\(p_{mono}\\)", "\\(p_b\\)"],
        compact,
    )

    proj = _read("paper_assets/tables/table2_mrw_mono_projection.csv")
    p1024 = proj[proj["T"].eq(1024)].copy()
    compact2 = []
    for name in order:
        sub = p1024[p1024["process_type"].eq(name)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        compact2.append([row["process_type"], _fmt(row["mean_lambda2_proj"]), _fmt(row["mean_residual_norm"]), _fmt(row["mean_mono_residual_norm"]), _fmt(row["mean_gain"]), _fmt(row["mean_tail_instability"])])
    pd.DataFrame(compact2, columns=["process_type", "lambda2_proj", "mrw_residual", "mono_residual", "mrw_vs_mono_gain", "tail_instability"]).to_csv(ASSET_TABLES / "table2_mrw_mono_projection_compact.csv", index=False)
    _booktabs_table(
        LATEX_TABLES / "table2_mrw_mono_projection.tex",
        "Projection diagnostics at \\(T=1024\\). Residuals and instability contextualize MRW-family compatibility.",
        "tab:table2_mrw_mono_projection",
        ["Process", "\\(\\lambda^2_p\\)", "\\(r_{MRW}\\)", "\\(r_{mono}\\)", "Gain", "Instab."],
        compact2,
    )

    ident = _read("paper_assets/tables/table4_finite_sample_identifiability.csv")
    ident.to_csv(ASSET_TABLES / "table3_finite_sample_lambda2_recovery_full.csv", index=False)
    ident[["T", "estimator", "lambda2_corr", "lambda2_mae", "high_lambda_detection_rate", "boundary_accuracy"]].to_csv(ASSET_TABLES / "fig_table3_lambda2_recovery_plot_data.csv", index=False)
    rows = []
    for T, g in ident.groupby("T"):
        best = g.loc[g["lambda2_corr"].idxmax()]
        worst = g.loc[g["lambda2_corr"].idxmin()]
        rows.append([
            int(T),
            _fmt(g["lambda2_corr"].mean()),
            _fmt(best["lambda2_corr"]),
            str(best["estimator"]).replace("structure_", ""),
            _fmt(worst["lambda2_corr"]),
            _fmt(g["high_lambda_detection_rate"].mean()),
        ])
    pd.DataFrame(rows, columns=["T", "mean_lambda2_corr", "best_lambda2_corr", "best_estimator", "worst_lambda2_corr", "mean_high_lambda_detection_rate"]).to_csv(ASSET_TABLES / "table3_finite_sample_lambda2_recovery_compact.csv", index=False)
    _booktabs_table(
        LATEX_TABLES / "table3_finite_sample_lambda2_recovery.tex",
        "Compact finite-sample \\(\\lambda^2\\) recovery summary. Full estimator-level results are retained in the paper assets.",
        "tab:table3_finite_sample_lambda2_recovery",
        ["T", "Mean corr.", "Best corr.", "Best est.", "Worst corr.", "Detect"],
        rows,
    )

    ab = _read("paper_assets/tables/table3_ablation_or_version_comparison.csv")
    keep_proc = ["MRW", "fGn", "iid Gaussian", "iid Student-t", "Regime-switching Gaussian"]
    ab1024 = ab[ab["T"].eq(1024) & ab["process_type"].isin(keep_proc)].copy()
    versions = [v for v in ["v1", "v2", "v3", "calibrated", "zeta_aligned", "curvature_preserved"] if v in set(ab1024["version"])]
    if not versions:
        versions = list(dict.fromkeys(ab1024["version"].tolist()))[:4]
    rows4 = []
    for v in versions:
        sub = ab1024[ab1024["version"].eq(v)]
        mrw = sub[sub["process_type"].eq("MRW")]
        fgn = sub[sub["process_type"].eq("fGn")]
        gauss = sub[sub["process_type"].eq("iid Gaussian")]
        student = sub[sub["process_type"].eq("iid Student-t")]
        rows4.append([
            v,
            _fmt(mrw["mean_p_mrw"].iloc[0] if not mrw.empty else np.nan),
            _fmt(fgn["mean_p_mrw"].iloc[0] if not fgn.empty else np.nan),
            _fmt(gauss["mean_p_mrw"].iloc[0] if not gauss.empty else np.nan),
            _fmt(student["mean_p_mrw"].iloc[0] if not student.empty else np.nan),
        ])
    pd.DataFrame(rows4, columns=["version", "MRW_p_MRW", "fGn_p_MRW", "Gaussian_p_MRW", "StudentT_p_MRW"]).to_csv(ASSET_TABLES / "table4_ablation_compact.csv", index=False)
    _booktabs_table(
        LATEX_TABLES / "table4_ablation.tex",
        "Compact ablation summary at \\(T=1024\\). Values are MRW-compatibility diagnostics, not mechanism probabilities.",
        "tab:table4_ablation",
        ["Version", "MRW", "fGn", "Gaussian", "Student-t"],
        rows4,
    )

    decision = [
        "# Table 3 Revision Decision",
        "",
        "The original finite-sample lambda2 table was not too large in row count, but it was visually weak: it mixed several metrics across every estimator and did not foreground the main negative result.",
        "",
        "## Decision",
        "",
        "Use a compact main-text table with one row per T: mean lambda2 correlation, best correlation, best estimator, worst correlation, and mean high-lambda detection rate.",
        "",
        "## Preserved data",
        "",
        "- Full table: `paper_assets/tables/table3_finite_sample_lambda2_recovery_full.csv`",
        "- Plot data: `paper_assets/tables/fig_table3_lambda2_recovery_plot_data.csv`",
        "- Compact table: `paper_assets/tables/table3_finite_sample_lambda2_recovery_compact.csv`",
        "",
        "## Recommended visual use",
        "",
        "Fig.5 should carry the main visual burden for finite-sample recovery. Table 3 should act as a compact numerical support table.",
    ]
    (SUMMARIES / "table3_revision_decision.md").write_text("\n".join(decision), encoding="utf-8")


def main() -> int:
    export_figure_data()
    revise_tables()
    print(json.dumps({"status": "ok", "figure_data": str(FIG_DATA.relative_to(ROOT)), "tables": str(ASSET_TABLES.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
