from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_dl.data import load_dataset, make_splits, save_splits, summarize_split_sizes


def main() -> None:
    bundle = load_dataset()
    splits = make_splits(bundle.dx.shape[0], train_ratio=0.7, val_ratio=0.15, seed=2026)
    out_path = save_splits(splits)
    result = {
        "saved_to": str(out_path),
        "sizes": summarize_split_sizes(splits),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

