"""Small helpers for paper-level wrapper scripts."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--quick", action="store_true", help="Use a small, fast configuration.")
    parser.add_argument("--output-dir", default="paper_assets", help="Destination asset directory.")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def ensure_paper_dirs(output_dir: str | Path) -> Path:
    base = ROOT / output_dir
    for name in ("figures", "tables", "summaries", "latex"):
        (base / name).mkdir(parents=True, exist_ok=True)
    return base


def run_python(script: str, args: list[str]) -> int:
    cmd = [sys.executable, str(ROOT / script), *args]
    completed = subprocess.run(cmd, cwd=ROOT)
    return int(completed.returncode)


def copy_if_exists(src: str | Path, dst: str | Path) -> bool:
    src_path = ROOT / src
    dst_path = ROOT / dst
    if not src_path.exists():
        return False
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.resolve() == dst_path.resolve():
        return True
    shutil.copy2(src_path, dst_path)
    return True


def write_note(path: str | Path, title: str, lines: list[str]) -> None:
    out = ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# " + title + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
