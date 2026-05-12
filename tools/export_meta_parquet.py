#!/usr/bin/env python3
"""Export score metadata summaries to Parquet."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


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
    data_dir_value = os.environ.get("DATA_DIR", "data")
    data_dir = Path(data_dir_value).expanduser()
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    return data_dir.resolve()


def load_meta(meta_path: Path) -> dict[str, Any]:
    with meta_path.open("r", encoding="utf-8") as file:
        meta = yaml.safe_load(file) or {}
    if not isinstance(meta, dict):
        raise ValueError(f"Expected mapping in {meta_path}")
    return meta


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def frame_size_struct(layout: dict[str, Any]) -> dict[str, int | None]:
    frame_size = layout.get("frame_size") or {}
    if not isinstance(frame_size, dict):
        frame_size = {}
    return {
        "width": int_or_none(frame_size.get("width")),
        "height": int_or_none(frame_size.get("height")),
    }


def count_frames_with_areas(layout: dict[str, Any]) -> int:
    frames = layout.get("frames") or []
    if not isinstance(frames, list):
        return 0
    count = 0
    for frame in frames:
        if isinstance(frame, dict) and frame.get("areas"):
            count += 1
    return count


def summarize_meta(meta_path: Path) -> dict[str, Any]:
    meta = load_meta(meta_path)
    layout = meta.get("layout") or {}
    if not isinstance(layout, dict):
        layout = {}
    return {
        "video_id": str_or_none(meta.get("video_id")),
        "video_title": str_or_none(meta.get("video_title")),
        "video_channel": str_or_none(meta.get("video_channel")),
        "duration_seconds": float_or_none(meta.get("duration_seconds")),
        "staff_n": int_or_none(meta.get("staff_n")),
        "staffLayout": str_or_none(meta.get("staffLayout")),
        "layout.frame_size": frame_size_struct(layout),
        "layout.score_grid_rows": int_or_none(layout.get("score_grid_rows")),
        "frames_n": count_frames_with_areas(layout),
    }


def build_table(rows: list[dict[str, Any]]) -> pa.Table:
    schema = pa.schema([
        pa.field("video_id", pa.string()),
        pa.field("video_title", pa.string()),
        pa.field("video_channel", pa.string()),
        pa.field("duration_seconds", pa.float64()),
        pa.field("staff_n", pa.int64()),
        pa.field("staffLayout", pa.string()),
        pa.field("layout.frame_size", pa.struct([
            pa.field("width", pa.int64()),
            pa.field("height", pa.int64()),
        ])),
        pa.field("layout.score_grid_rows", pa.int64()),
        pa.field("frames_n", pa.int64()),
    ])
    return pa.Table.from_pylist(rows, schema=schema)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export score metadata summaries to data/meta.parquet.")
    parser.add_argument("--scores-dir", type=Path, help="Score metadata directory. Default: $DATA_DIR/scores")
    parser.add_argument("--output", type=Path, help="Output Parquet path. Default: $DATA_DIR/meta.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env(PROJECT_ROOT / ".env")
    load_env(PROJECT_ROOT / ".env.local", override=True)
    data_dir = resolve_data_dir()
    scores_dir = args.scores_dir.expanduser().resolve() if args.scores_dir else data_dir / "scores"
    output_path = args.output.expanduser().resolve() if args.output else data_dir / "meta.parquet"
    rows = [summarize_meta(meta_path) for meta_path in sorted(scores_dir.glob("*/meta.yaml"))]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(build_table(rows), output_path)
    print(f"wrote {len(rows)} rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
