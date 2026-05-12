#!/usr/bin/env python3
"""Fetch YouTube title/channel metadata for score directories."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOUTUBE_WATCH_URL = "https://www.youtube.com/watch?v={video_id}"


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


def yaml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def meta_has_video_metadata(meta_path: Path) -> bool:
    lines = meta_path.read_text(encoding="utf-8").splitlines()
    return any(line.startswith("video_title:") for line in lines) and any(line.startswith("video_channel:") for line in lines)


def write_video_metadata(meta_path: Path, title: str, channel: str) -> None:
    lines = meta_path.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    inserted = False
    found_video_id = False
    for line in lines:
        if line.startswith("video_title:") or line.startswith("video_channel:"):
            continue
        output.append(line)
        if line.startswith("video_id:"):
            found_video_id = True
            output.append(f"video_title: {yaml_quote(title)}")
            output.append(f"video_channel: {yaml_quote(channel)}")
            inserted = True
    if not found_video_id or not inserted:
        raise ValueError(f"Missing video_id field in {meta_path}")
    meta_path.write_text("\n".join(output) + "\n", encoding="utf-8")


def fetch_video_metadata(video_id: str) -> dict[str, Any]:
    proc = subprocess.run(
        [
            "yt-dlp",
            "--skip-download",
            "--dump-json",
            "--no-playlist",
            YOUTUBE_WATCH_URL.format(video_id=video_id),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        raise RuntimeError(f"yt-dlp exit code {proc.returncode}")
    return json.loads(proc.stdout)


def iter_meta_paths(scores_dir: Path, wanted: set[str] | None) -> list[Path]:
    paths = []
    for meta_path in sorted(scores_dir.glob("*/meta.yaml")):
        if wanted is not None and meta_path.parent.name not in wanted:
            continue
        paths.append(meta_path)
    return paths


def update_one(meta_path: Path, *, overwrite: bool, fail_fast: bool) -> bool:
    video_id = meta_path.parent.name
    if not overwrite and meta_has_video_metadata(meta_path):
        print(f"skip metadata {video_id}", flush=True)
        return False
    print(f"fetch metadata {video_id}", flush=True)
    try:
        info = fetch_video_metadata(video_id)
        title = str(info.get("title") or "")
        channel = str(info.get("channel") or info.get("uploader") or "")
        write_video_metadata(meta_path, title, channel)
    except Exception as exc:
        print(f"failed metadata {video_id}: {exc}", file=sys.stderr, flush=True)
        if fail_fast:
            raise
        return False
    print(f"wrote metadata {video_id}: {title}", flush=True)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch YouTube title/channel metadata for score meta.yaml files.")
    parser.add_argument("--video-id", action="append", help="Only update this video ID. Repeatable. Use --video-id=-abc for IDs beginning with dash.")
    parser.add_argument("--limit", type=int, help="Maximum number of meta.yaml files to process.")
    parser.add_argument("--overwrite", action="store_true", help="Refetch metadata even when video_title and video_channel already exist.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between yt-dlp metadata requests. Default: 0")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first metadata fetch/write failure.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env(PROJECT_ROOT / ".env")
    load_env(PROJECT_ROOT / ".env.local", override=True)
    data_dir = resolve_data_dir()
    scores_dir = data_dir / "scores"
    wanted = set(args.video_id) if args.video_id else None
    meta_paths = iter_meta_paths(scores_dir, wanted)
    if args.limit is not None:
        meta_paths = meta_paths[: args.limit]
    updated = 0
    for index, meta_path in enumerate(meta_paths):
        if index and args.sleep > 0:
            time.sleep(args.sleep)
        if update_one(meta_path, overwrite=args.overwrite, fail_fast=args.fail_fast):
            updated += 1
    print(f"updated {updated}/{len(meta_paths)} metadata files", flush=True)


if __name__ == "__main__":
    main()
