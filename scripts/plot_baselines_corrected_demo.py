from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "outputs" / "baselines" / "baseline_corrected_demo.json"


def main() -> None:
    item = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    axes[0].plot(item["q_vals"], item["zeta_true"], marker="o", label="True")
    axes[0].plot(item["q_vals"], item["zeta_sf"], marker="s", label="SF")
    axes[0].plot(item["q_vals"], item["zeta_mfdfa"], marker="^", label="MFDFA")
    axes[0].set_xlabel("q")
    axes[0].set_ylabel("zeta(q)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].plot(item["alpha_true"], item["f_true"], marker="o", label="True")
    axes[1].plot(item["alpha_sf"], item["f_sf"], marker="s", label="SF")
    axes[1].plot(item["alpha_mfdfa"], item["f_mfdfa"], marker="^", label="MFDFA")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("f(alpha)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=False)

    output_path = ROOT / "outputs" / "baselines" / "baseline_corrected_demo.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"saved_to": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
