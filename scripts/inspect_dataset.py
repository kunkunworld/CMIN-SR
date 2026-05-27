from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_dl.data import describe_dataset, load_dataset, load_metadata


def main() -> None:
    bundle = load_dataset()
    meta = load_metadata()
    summary = {
        "dataset": describe_dataset(bundle),
        "metadata": meta,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

