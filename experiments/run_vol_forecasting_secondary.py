from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "reports" / "vol_forecasting_secondary_plan.md"


def main() -> None:
    text = """# Secondary Volatility Forecasting Plan

Positioning:
- Volatility forecasting is secondary evidence.
- It should not be the main scientific claim of the project.

Current reusable results:
- `outputs/reports/factor_real_world/factor_next20_rv_forecast_metrics.csv`
- `outputs/reports/factor_real_world/factor_real_world_summary.md`

Future market-return targets:
- next 5d realized volatility
- next 20d realized volatility
- intraday BTC horizons if intraday data becomes available
"""
    OUT.write_text(text, encoding="utf-8")
    print(json.dumps({"report": str(OUT.relative_to(ROOT)), "status": "skeleton_ready"}, indent=2))


if __name__ == "__main__":
    main()
