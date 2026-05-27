from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REPORT_DIR=ROOT/"outputs"/"reports"/"zeta_alignment_comparison"
TABLE_DIR=ROOT/"outputs"/"tables"/"zeta_alignment_comparison"
FIG_DIR=ROOT/"outputs"/"figures"/"zeta_alignment_comparison"

def _read(p): return pd.read_csv(p) if p.exists() else None

def main():
    REPORT_DIR.mkdir(parents=True,exist_ok=True); TABLE_DIR.mkdir(parents=True,exist_ok=True); FIG_DIR.mkdir(parents=True,exist_ok=True)
    before=_read(ROOT/"outputs"/"tables"/"cmin_sr_spectrum_calibrated_eval"/"process_by_T.csv")
    after=_read(ROOT/"outputs"/"tables"/"raw_zeta_alignment_eval"/"process_by_T.csv")
    rows=[]
    for name,df in [("before_spectrum_cal",before),("after_zeta_alignment",after)]:
        if df is None: continue
        for _,r in df.iterrows():
            rows.append({"stage":name,"T":int(r["T"]),"process_type":r["process_type"],"zeta_mae":r.get("zeta_mae",float("nan")),"second_diff_norm":r.get("second_diff_norm",float("nan")),"p_curved_cal":r.get("p_curved_cal",float("nan")),"p_mrw_cal":r.get("p_mrw_cal",float("nan")),"p_mono_cal":r.get("p_mono_cal",float("nan")),"p_boundary_cal":r.get("p_boundary_cal",float("nan"))})
    comp=pd.DataFrame(rows); comp.to_csv(TABLE_DIR/"zeta_alignment_comparison.csv",index=False)
    plot=comp[(comp["T"]==1024)&(comp["process_type"].isin(["MRW","Low-lambda2 MRW","fGn","iid Gaussian","iid Student-t","Regime-switching Gaussian"]))]
    if not plot.empty:
        fig,ax=plt.subplots(figsize=(8,4.8),constrained_layout=True)
        pivot=plot.pivot_table(index="process_type",columns="stage",values="p_mrw_cal",aggfunc="first")
        pivot.plot(kind="bar",ax=ax); ax.set_ylabel("p_MRW_cal"); ax.tick_params(axis="x",rotation=30); fig.savefig(FIG_DIR/"pmrw_cal_before_after.png",dpi=220); plt.close(fig)
    report=REPORT_DIR/"zeta_alignment_comparison_summary.md"; report.write_text("\n".join(["# Zeta Alignment Comparison","",comp.to_csv(index=False)]),encoding="utf-8")
    meta={"comparison":str((TABLE_DIR/"zeta_alignment_comparison.csv").relative_to(ROOT)),"report":str(report.relative_to(ROOT))}
    (REPORT_DIR/"zeta_alignment_comparison_summary.json").write_text(json.dumps(meta,indent=2),encoding="utf-8"); print(json.dumps(meta,indent=2))
if __name__=="__main__": main()
