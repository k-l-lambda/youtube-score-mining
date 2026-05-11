#!/usr/bin/env python3
"""Generate adjacent-frame difference debug plots for a downloaded video."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_ID = "-XkwDuv0hw4"
VIDEO_EXTENSIONS = {".webm", ".mkv", ".mp4", ".mov", ".m4v"}


def run_command(args: list[str], *, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text)


def resolve_video_path(video_id: str, video_dir: Path) -> Path:
    matches = [path for path in sorted(video_dir.glob(f"{video_id}.*")) if path.suffix.lower() in VIDEO_EXTENSIONS and ".part" not in path.name]
    if not matches:
        raise SystemExit(f"No downloaded video found for {video_id} under {video_dir}")
    return matches[0]


def probe_duration(video_path: Path) -> float:
    result = run_command([
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ])
    return float(result.stdout.strip())


def read_scaled_gray_frames(video_path: Path, width: int, height: int) -> list[np.ndarray]:
    proc = subprocess.Popen([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"scale={width}:{height},format=gray",
        "-f",
        "rawvideo",
        "-",
    ], stdout=subprocess.PIPE)
    frame_size = width * height
    frames: list[np.ndarray] = []
    while True:
        chunk = proc.stdout.read(frame_size) if proc.stdout else b""
        if not chunk:
            break
        if len(chunk) != frame_size:
            break
        frames.append(np.frombuffer(chunk, dtype=np.uint8).astype(np.int16))
    exit_code = proc.wait()
    if exit_code:
        raise SystemExit(f"ffmpeg failed with exit code {exit_code}: {video_path}")
    if len(frames) < 2:
        raise SystemExit(f"Not enough decoded frames for adjacent diff: {video_path}")
    return frames


def compute_adjacent_diffs(frames: list[np.ndarray], duration: float) -> list[tuple[int, float, float]]:
    frame_count = len(frames)
    rows: list[tuple[int, float, float]] = []
    for index in range(1, frame_count):
        diff = float(np.mean(np.abs(frames[index] - frames[index - 1])))
        seconds = index * duration / max(frame_count - 1, 1)
        rows.append((index, seconds, diff))
    return rows


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def top_local_peaks(rows: list[tuple[int, float, float]], cutoff: float, limit: int) -> list[dict[str, float | int]]:
    peaks: list[dict[str, float | int]] = []
    for index in range(1, len(rows) - 1):
        value = rows[index][2]
        if value >= cutoff and value >= rows[index - 1][2] and value >= rows[index + 1][2]:
            peaks.append({
                "frame_index": rows[index][0],
                "seconds": round(rows[index][1], 6),
                "mean_abs_gray_diff": round(value, 6),
            })
    return sorted(peaks, key=lambda item: float(item["mean_abs_gray_diff"]), reverse=True)[:limit]


def write_csv(path: Path, rows: list[tuple[int, float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "seconds", "mean_abs_gray_diff"])
        for frame_index, seconds, diff in rows:
            writer.writerow([frame_index, f"{seconds:.6f}", f"{diff:.6f}"])


def write_timeline(path: Path, video_id: str, seconds: np.ndarray, values: np.ndarray, stats: dict[str, float]) -> None:
    plt.figure(figsize=(16, 5))
    plt.plot(seconds, values, linewidth=0.6)
    for label in ["p90", "p95", "p99"]:
        plt.axhline(stats[label], linestyle="--", linewidth=0.8, label=f"{label}={stats[label]:.3f}")
    plt.xlabel("seconds")
    plt.ylabel("mean abs gray diff")
    plt.title(f"{video_id} adjacent-frame difference timeline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def write_histogram(path: Path, video_id: str, values: np.ndarray, stats: dict[str, float]) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=120, log=True)
    for label in ["p90", "p95", "p99"]:
        plt.axvline(stats[label], linestyle="--", linewidth=0.9, label=f"{label}={stats[label]:.3f}")
    plt.xlabel("mean abs gray diff")
    plt.ylabel("frame-pair count (log)")
    plt.title(f"{video_id} adjacent-frame difference histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def build_outputs(video_id: str, video_path: Path, output_dir: Path, width: int, height: int, peak_limit: int) -> dict[str, Any]:
    duration = probe_duration(video_path)
    frames = read_scaled_gray_frames(video_path, width, height)
    rows = compute_adjacent_diffs(frames, duration)
    values = np.array([row[2] for row in rows])
    seconds = np.array([row[1] for row in rows])
    stats = summarize(values)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"{video_id}_adjacent_frame_diff"
    csv_path = prefix.with_suffix(".csv")
    timeline_path = output_dir / f"{video_id}_adjacent_frame_diff_timeline.png"
    histogram_path = output_dir / f"{video_id}_adjacent_frame_diff_histogram.png"
    peaks_path = output_dir / f"{video_id}_adjacent_frame_diff_peaks.json"

    write_csv(csv_path, rows)
    write_timeline(timeline_path, video_id, seconds, values, stats)
    write_histogram(histogram_path, video_id, values, stats)
    peaks = top_local_peaks(rows, stats["p99"], peak_limit)
    peaks_path.write_text(json.dumps({
        "video_id": video_id,
        "video_path": str(video_path),
        "duration": duration,
        "frame_count": len(frames),
        "diff_count": len(rows),
        "scale": {"width": width, "height": height},
        "stats": stats,
        "top_local_peaks_p99": peaks,
    }, indent=2), encoding="utf-8")

    return {
        "video": str(video_path),
        "duration": duration,
        "frames": len(frames),
        "diffs": len(rows),
        "stats": stats,
        "csv": str(csv_path),
        "timeline": str(timeline_path),
        "histogram": str(histogram_path),
        "peaks": str(peaks_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot adjacent-frame mean absolute gray differences for segmentation tuning.")
    parser.add_argument("--video-id", default=DEFAULT_VIDEO_ID, help=f"Video ID to analyze. Default: {DEFAULT_VIDEO_ID}")
    parser.add_argument("--video-dir", type=Path, default=PROJECT_ROOT / "data" / "video", help="Downloaded video directory.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "temp" / "xkw-frame-diff", help="Output directory for CSV, plots, and peaks JSON.")
    parser.add_argument("--width", type=int, default=160, help="Scaled grayscale frame width. Default: 160")
    parser.add_argument("--height", type=int, default=90, help="Scaled grayscale frame height. Default: 90")
    parser.add_argument("--peak-limit", type=int, default=100, help="Maximum local p99 peaks to store. Default: 100")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_dir = args.video_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    video_path = resolve_video_path(args.video_id, video_dir)
    result = build_outputs(args.video_id, video_path, output_dir, args.width, args.height, args.peak_limit)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
