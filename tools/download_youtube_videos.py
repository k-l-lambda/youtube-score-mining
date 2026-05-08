#!/usr/bin/env python3
"""Download YouTube videos listed in data/id_list.txt with yt-dlp."""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ID_LIST = PROJECT_ROOT / "data" / "id_list.txt"
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


def read_video_ids(id_list_path: Path) -> list[str]:
    video_ids: list[str] = []
    for raw_line in id_list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        video_ids.append(line)
    return video_ids


def target_exists(video_dir: Path, video_id: str) -> bool:
    for path in video_dir.glob(f"{video_id}.*"):
        if not path.is_file():
            continue
        if ".part" in path.name or path.suffix in {".ytdl", ".tmp", ".temp"}:
            continue
        return True
    return False


def run_yt_dlp(video_id: str, video_dir: Path, format_selector: str) -> None:
    url = YOUTUBE_WATCH_URL.format(video_id=video_id)
    output_template = str(video_dir / "%(id)s.%(ext)s")
    subprocess.run([
        "yt-dlp",
        "--no-playlist",
        "-f",
        format_selector,
        "-o",
        output_template,
        url,
    ], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download YouTube videos from data/id_list.txt into $DATA_DIR/video.")
    parser.add_argument("--env", type=Path, default=PROJECT_ROOT / ".env", help="Base environment file to load. Default: .env")
    parser.add_argument("--env-local", type=Path, default=PROJECT_ROOT / ".env.local", help="Higher-priority environment file. Default: .env.local")
    parser.add_argument("--id-list", type=Path, default=DEFAULT_ID_LIST, help="Video ID list path. Default: data/id_list.txt")
    parser.add_argument("--min-sleep", type=float, default=180.0, help="Minimum seconds to wait between downloads. Default: 180")
    parser.add_argument("--max-sleep", type=float, default=600.0, help="Maximum seconds to wait between downloads. Default: 600")
    parser.add_argument("--limit", type=int, help="Download at most this many missing videos.")
    parser.add_argument("--format", default="bv*+ba/b", help="yt-dlp format selector. Default: bv*+ba/b")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading or sleeping.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env(args.env.expanduser().resolve())
    load_env(args.env_local.expanduser().resolve(), override=True)

    data_dir_value = os.environ.get("DATA_DIR")
    if not data_dir_value:
        raise SystemExit("DATA_DIR is not set. Add it to .env before running this script.")

    data_dir = Path(data_dir_value).expanduser()
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    data_dir = data_dir.resolve()
    video_dir = data_dir / "video"
    id_list_path = args.id_list.expanduser().resolve()
    if not id_list_path.is_file():
        raise SystemExit(f"Video ID list does not exist: {id_list_path}")
    if args.min_sleep < 0 or args.max_sleep < args.min_sleep:
        raise SystemExit("Sleep range is invalid: require 0 <= min-sleep <= max-sleep")

    video_dir.mkdir(parents=True, exist_ok=True)
    video_ids = read_video_ids(id_list_path)
    downloaded = 0
    attempted = 0
    failed = 0

    for index, video_id in enumerate(video_ids, start=1):
        if target_exists(video_dir, video_id):
            print(f"[{index}/{len(video_ids)}] skip existing {video_id}", flush=True)
            continue
        if args.limit is not None and attempted >= args.limit:
            break

        attempted += 1
        print(f"[{index}/{len(video_ids)}] download {video_id}", flush=True)
        if not args.dry_run:
            try:
                run_yt_dlp(video_id, video_dir, args.format)
            except subprocess.CalledProcessError as exc:
                failed += 1
                print(f"[{index}/{len(video_ids)}] failed {video_id}: yt-dlp exit code {exc.returncode}", flush=True)
            else:
                downloaded += 1
        else:
            downloaded += 1

        has_more_missing = any(not target_exists(video_dir, later_id) for later_id in video_ids[index:])
        if has_more_missing and (args.limit is None or attempted < args.limit):
            sleep_seconds = random.uniform(args.min_sleep, args.max_sleep)
            print(f"sleep {sleep_seconds:.1f}s before next download", flush=True)
            if not args.dry_run:
                time.sleep(sleep_seconds)

    print(f"attempted {attempted}; downloaded {downloaded} new videos into {video_dir}; failed {failed}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted")
