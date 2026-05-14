#!/usr/bin/env python3
"""Summarize file sizes by extension under data/scores, with JSON files split by name."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_env(env_path: Path, *, override: bool = False) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value


def resolve_data_dir() -> Path:
    data_dir = Path(os.environ.get("DATA_DIR", "data")).expanduser()
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    return data_dir.resolve()


def human_size(size: int) -> str:
    value = float(size)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{size} B"
        value /= 1024
    return f"{size} B"


def file_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return path.name
    return suffix if suffix else "[no extension]"


def iter_score_files(scores_dir: Path):
    for root, dirs, files in os.walk(scores_dir):
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        root_path = Path(root)
        for name in files:
            yield root_path / name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize total file sizes by extension under data/scores, with JSON files split by name.")
    parser.add_argument("--env", type=Path, default=PROJECT_ROOT / ".env", help="Base environment file. Default: .env")
    parser.add_argument("--env-local", type=Path, default=PROJECT_ROOT / ".env.local", help="Higher-priority environment file. Default: .env.local")
    parser.add_argument("--scores-dir", type=Path, help="Scores directory. Default: $DATA_DIR/scores")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env(args.env.expanduser().resolve())
    load_env(args.env_local.expanduser().resolve(), override=True)
    scores_dir = args.scores_dir.expanduser().resolve() if args.scores_dir else resolve_data_dir() / "scores"
    if not scores_dir.is_dir():
        raise SystemExit(f"Scores directory does not exist: {scores_dir}")

    counts: dict[str, int] = defaultdict(int)
    sizes: dict[str, int] = defaultdict(int)
    total_count = 0
    total_size = 0
    for path in iter_score_files(scores_dir):
        kind = file_type(path)
        size = path.stat().st_size
        counts[kind] += 1
        sizes[kind] += size
        total_count += 1
        total_size += size

    print(f"scores_dir: {scores_dir}")
    print(f"total_files: {total_count}")
    print(f"total_size: {total_size} ({human_size(total_size)})")
    print()
    print(f"{'type':<16} {'files':>8} {'bytes':>14} {'size':>12}")
    print("-" * 54)
    for kind in sorted(sizes, key=lambda item: (-sizes[item], item)):
        print(f"{kind:<16} {counts[kind]:>8} {sizes[kind]:>14} {human_size(sizes[kind]):>12}")


if __name__ == "__main__":
    main()
