#!/usr/bin/env python3
"""Generate score metadata from downloaded YouTube videos."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTENSIONS = {".webm", ".mkv", ".mp4", ".mov", ".m4v"}
SCENE_RE = re.compile(r"lavfi\.scene_score=([0-9.]+)")
TIME_RE = re.compile(r"pts_time:([0-9.]+)")


@dataclass(frozen=True)
class SceneSample:
    seconds: float
    score: float


@dataclass(frozen=True)
class TimeWindow:
    start: float
    end: float
    maximum: float


@dataclass(frozen=True)
class FrameStats:
    seconds: float
    yavg: float
    satavg: float


@dataclass(frozen=True)
class FramePairStats:
    seconds: float
    yavg: float
    satavg: float
    diff: float


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


def run_command(args: list[str], *, capture: bool = True, text: bool = True) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            args,
            check=True,
            text=text,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"Required executable not found: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            sys.stderr.write(exc.stderr)
        raise SystemExit(f"Command failed with exit code {exc.returncode}: {' '.join(args)}") from exc


def resolve_data_dir() -> Path:
    data_dir_value = os.environ.get("DATA_DIR", "data")
    data_dir = Path(data_dir_value).expanduser()
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    return data_dir.resolve()


def video_id_from_path(video_path: Path) -> str:
    name = video_path.stem
    for marker in [".f137", ".f248", ".f399", ".f401", ".f251"]:
        if name.endswith(marker):
            return name[: -len(marker)]
    return name


def find_videos(video_dir: Path) -> list[Path]:
    videos = []
    for path in sorted(video_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if ".part" in path.name:
            continue
        videos.append(path)
    return videos


def probe_duration(video_path: Path) -> float:
    result = run_command([
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ])
    duration = json.loads(result.stdout).get("format", {}).get("duration")
    if duration is None:
        raise SystemExit(f"Could not read duration from {video_path}")
    return float(duration)


def collect_scene_samples(video_path: Path) -> list[SceneSample]:
    result = run_command([
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(video_path),
        "-vf",
        "select='gte(scene,0)',metadata=print",
        "-f",
        "null",
        "-",
    ])

    samples: list[SceneSample] = []
    pending_seconds: float | None = None
    for line in result.stderr.splitlines():
        time_match = TIME_RE.search(line)
        if time_match is not None:
            pending_seconds = float(time_match.group(1))
            continue
        scene_match = SCENE_RE.search(line)
        if scene_match is None or pending_seconds is None:
            continue
        samples.append(SceneSample(pending_seconds, float(scene_match.group(1))))
        pending_seconds = None
    return samples


def sample_frame_stats(video_path: Path, duration: float, sample_count: int, skip_fraction: float) -> list[FrameStats]:
    if sample_count <= 0:
        return []
    bounded_skip = min(max(skip_fraction, 0.0), 0.45)
    start = duration * bounded_skip
    span = max(duration * (1.0 - 2.0 * bounded_skip), 1.0)
    interval = max(span / sample_count, 0.001)
    result = run_command([
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-ss",
        format_seconds(start),
        "-t",
        format_seconds(span),
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{format_seconds(interval)},scale=32:32,format=rgb24",
        "-f",
        "rawvideo",
        "-",
    ], text=False)
    frame_size = 32 * 32 * 3
    stats: list[FrameStats] = []
    for index in range(0, len(result.stdout), frame_size):
        frame = result.stdout[index:index + frame_size]
        if len(frame) != frame_size:
            continue
        stats.append(analyze_frame(frame, start + len(stats) * interval))
    return stats


def sample_frame_pair_stats(video_path: Path, duration: float, pair_count: int, frame_delta: float) -> list[FramePairStats]:
    if pair_count <= 0:
        return []
    start = duration * 0.35
    span = duration * 0.3
    interval = span / max(pair_count - 1, 1)
    stats: list[FramePairStats] = []
    for index in range(pair_count):
        first_seconds = min(max(start + index * interval, 0.0), max(duration - frame_delta, 0.0))
        first = extract_rgb_frame(video_path, first_seconds)
        second = extract_rgb_frame(video_path, first_seconds + frame_delta)
        if first is None or second is None:
            continue
        frame_stats = analyze_frame(first, first_seconds)
        stats.append(FramePairStats(frame_stats.seconds, frame_stats.yavg, frame_stats.satavg, mean_frame_diff(first, second)))
    return stats


def extract_rgb_frame(video_path: Path, seconds: float) -> bytes | None:
    result = run_command([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        format_seconds(seconds),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        "scale=32:32,format=rgb24",
        "-f",
        "rawvideo",
        "-",
    ], text=False)
    frame_size = 32 * 32 * 3
    if len(result.stdout) < frame_size:
        return None
    return result.stdout[:frame_size]


def analyze_frame(frame: bytes, seconds: float) -> FrameStats:
    pixel_count = len(frame) // 3
    y_total = 0.0
    sat_total = 0.0
    for index in range(0, len(frame), 3):
        red = frame[index]
        green = frame[index + 1]
        blue = frame[index + 2]
        maximum = max(red, green, blue)
        minimum = min(red, green, blue)
        y_total += 0.299 * red + 0.587 * green + 0.114 * blue
        if maximum:
            sat_total += (maximum - minimum) / maximum * 255.0
    return FrameStats(seconds, y_total / pixel_count, sat_total / pixel_count)


def mean_frame_diff(first: bytes, second: bytes) -> float:
    if len(first) != len(second) or not first:
        return 0.0
    return sum(abs(left - right) for left, right in zip(first, second)) / len(first)


def is_score_video(stats: list[FrameStats], yavg_min: float, satavg_max: float, min_fraction: float) -> bool:
    if not stats:
        return True
    qualifying = sum(1 for item in stats if item.yavg >= yavg_min and item.satavg <= satavg_max)
    return qualifying / len(stats) >= min_fraction


def is_score_video_from_pairs(stats: list[FramePairStats], yavg_min: float, satavg_max: float, min_fraction: float, diff_max: float) -> bool:
    if not stats:
        return True
    if all(item.diff > diff_max for item in stats):
        return False
    qualifying = sum(1 for item in stats if item.yavg >= yavg_min and item.satavg <= satavg_max)
    return qualifying / len(stats) >= min_fraction


def build_windows(samples: list[SceneSample], duration: float, window_seconds: float) -> list[TimeWindow]:
    windows: list[TimeWindow] = []
    window_count = max(1, int(duration // window_seconds) + 1)
    for index in range(window_count):
        start = index * window_seconds
        end = min(start + window_seconds, duration)
        scores = [sample.score for sample in samples if start <= sample.seconds < start + window_seconds]
        if scores:
            windows.append(TimeWindow(start, end, max(scores)))
    return windows


def find_stable_start(samples: list[SceneSample], duration: float, stable_max: float, stable_seconds: float, search_until: float, intro_change_threshold: float) -> float:
    end_search = min(search_until, duration)
    intro_changes = [sample for sample in samples if sample.seconds < end_search and sample.score > intro_change_threshold]
    if not intro_changes:
        return 0.0

    search_start = intro_changes[-1].seconds
    for sample in samples:
        if sample.seconds <= search_start:
            continue
        if sample.seconds + stable_seconds > duration:
            break
        scores = [item.score for item in samples if sample.seconds <= item.seconds < sample.seconds + stable_seconds]
        if scores and max(scores) <= stable_max:
            return sample.seconds
    return 0.0


def find_final_frame_before_outro(windows: list[TimeWindow], duration: float, stable_max: float, minimum_outro_seconds: float) -> float:
    if not windows:
        return duration
    trailing: list[TimeWindow] = []
    for window in reversed(windows):
        if window.maximum <= stable_max:
            trailing.append(window)
        else:
            break
    if not trailing:
        return duration
    trailing_start = trailing[-1].start
    if duration - trailing_start >= minimum_outro_seconds:
        return trailing_start
    return duration


def format_timestamp(seconds: float) -> str:
    millis = round((seconds - int(seconds)) * 1000)
    total_seconds = int(seconds)
    if millis == 1000:
        total_seconds += 1
        millis = 0
    minutes, sec = divmod(total_seconds, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}:{sec:02d}.{millis:03d}"


def format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}"


def yaml_scalar(value: str | float) -> str:
    if isinstance(value, float):
        return format_seconds(value)
    return f'"{value}"'


def write_meta(meta_path: Path, video_id: str, duration: float, threshold: float, changes: list[dict[str, str | float]]) -> None:
    lines = [
        f"video_id: {video_id}",
        f"duration: {yaml_scalar(format_timestamp(duration))}",
        f"duration_seconds: {format_seconds(duration)}",
        "shot_detection:",
        "  method: ffmpeg_scene_score",
        f"  threshold: {threshold:g}",
        "  changes:",
    ]
    for change in changes:
        lines.append(f"    - time: {yaml_scalar(str(change['time']))}")
        lines.append(f"      seconds: {format_seconds(float(change['seconds']))}")
        if "role" in change:
            lines.append(f"      role: {change['role']}")
        if "score" in change:
            lines.append(f"      score: {float(change['score']):.4f}")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_audio(video_path: Path, audio_path: Path) -> None:
    run_command([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "44100",
        "-sample_fmt",
        "s16",
        str(audio_path),
    ], capture=False)


def build_score_image(video_path: Path, score_path: Path, boundaries: list[float]) -> None:
    segments = list(zip(boundaries[:-1], boundaries[1:]))
    if not segments:
        return
    with tempfile.TemporaryDirectory(prefix="score_frames_", dir=PROJECT_ROOT / "temp") as temp_name:
        temp_dir = Path(temp_name)
        frame_paths: list[Path] = []
        for index, (start, end) in enumerate(segments, start=1):
            midpoint = start + (end - start) / 2
            frame_path = temp_dir / f"frame_{index:02d}.webp"
            run_command([
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                format_seconds(midpoint),
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                "scale=1440:-1",
                str(frame_path),
            ], capture=False)
            frame_paths.append(frame_path)

        inputs: list[str] = []
        for frame_path in frame_paths:
            inputs.extend(["-i", str(frame_path)])
        filter_complex = "".join(f"[{index}:v]" for index in range(len(frame_paths))) + f"vstack=inputs={len(frame_paths)}[v]"
        run_command([
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-compression_level",
            "6",
            "-quality",
            "90",
            str(score_path),
        ], capture=False)


def segment_video(video_path: Path, scores_dir: Path, args: argparse.Namespace) -> None:
    video_id = video_id_from_path(video_path)
    sample_dir = scores_dir / video_id
    meta_path = sample_dir / "meta.yaml"
    audio_path = sample_dir / "audio.wav"
    score_path = sample_dir / "score.webp"
    if meta_path.exists() and not args.overwrite:
        print(f"skip existing {video_id}")
        return

    duration = probe_duration(video_path)
    frame_stats = sample_frame_pair_stats(video_path, duration, args.score_filter_pairs, args.score_filter_frame_delta)
    if not is_score_video_from_pairs(frame_stats, args.score_filter_yavg_min, args.score_filter_satavg_max, args.score_filter_min_fraction, args.score_filter_diff_max):
        print(f"skip non-score {video_id}", flush=True)
        return
    if args.score_filter_only:
        print(f"pass score-filter {video_id}", flush=True)
        return

    print(f"segment {video_id}: {video_path.name}", flush=True)
    samples = collect_scene_samples(video_path)
    windows = build_windows(samples, duration, args.window_seconds)
    stable_start = find_stable_start(samples, duration, args.stable_max, args.stable_seconds, args.intro_search_seconds, args.intro_change_threshold)
    final_frame = find_final_frame_before_outro(windows, duration, args.stable_max, args.minimum_outro_seconds)
    if final_frame <= stable_start:
        final_frame = duration

    cuts = [sample for sample in samples if stable_start < sample.seconds < final_frame and sample.score > args.threshold]
    changes: list[dict[str, str | float]] = [{
        "time": format_timestamp(stable_start),
        "seconds": stable_start,
        "role": "stable_start_after_intro",
    }]
    changes.extend({
        "time": format_timestamp(sample.seconds),
        "seconds": sample.seconds,
        "score": sample.score,
    } for sample in cuts)
    changes.append({
        "time": format_timestamp(final_frame),
        "seconds": final_frame,
        "role": "final_frame_before_outro",
    })

    sample_dir.mkdir(parents=True, exist_ok=True)
    write_meta(meta_path, video_id, duration, args.threshold, changes)
    if args.audio and (args.overwrite or not audio_path.exists()):
        extract_audio(video_path, audio_path)
    if args.score and (args.overwrite or not score_path.exists()):
        boundaries = [float(change["seconds"]) for change in changes]
        build_score_image(video_path, score_path, boundaries)
    print(f"wrote {meta_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment downloaded videos into score metadata directories.")
    parser.add_argument("--env", type=Path, default=PROJECT_ROOT / ".env", help="Base environment file. Default: .env")
    parser.add_argument("--env-local", type=Path, default=PROJECT_ROOT / ".env.local", help="Higher-priority environment file. Default: .env.local")
    parser.add_argument("--video-dir", type=Path, help="Input video directory. Default: $DATA_DIR/video")
    parser.add_argument("--scores-dir", type=Path, help="Output scores directory. Default: $DATA_DIR/scores")
    parser.add_argument("--threshold", type=float, default=0.35, help="Scene cut threshold. Default: 0.35")
    parser.add_argument("--stable-max", type=float, default=0.001, help="Max scene score for stable intro detection. Default: 0.001")
    parser.add_argument("--stable-seconds", type=float, default=2.0, help="Continuous stable seconds required after intro. Default: 2")
    parser.add_argument("--intro-search-seconds", type=float, default=15.0, help="Search range for intro end. Default: 15")
    parser.add_argument("--intro-change-threshold", type=float, default=0.05, help="Scene score treated as intro motion. Default: 0.05")
    parser.add_argument("--window-seconds", type=float, default=5.0, help="Window size for outro detection. Default: 5")
    parser.add_argument("--minimum-outro-seconds", type=float, default=10.0, help="Minimum trailing stable duration to mark outro. Default: 10")
    parser.add_argument("--limit", type=int, help="Process at most this many videos.")
    parser.add_argument("--video-id", action="append", help="Only process this video ID. Can be used multiple times.")
    parser.add_argument("--score-filter-only", action="store_true", help="Only run the score-video filter without generating sample outputs.")
    parser.add_argument("--score-filter-pairs", type=int, default=3, help="Number of adjacent frame pairs to sample for score filtering. Default: 3")
    parser.add_argument("--score-filter-frame-delta", type=float, default=0.25, help="Seconds between adjacent sampled frames. Default: 0.25")
    parser.add_argument("--score-filter-yavg-min", type=float, default=100.0, help="Minimum average luma for score-like frames. Default: 100")
    parser.add_argument("--score-filter-satavg-max", type=float, default=5.0, help="Maximum average saturation for score-like frames. Default: 5")
    parser.add_argument("--score-filter-min-fraction", type=float, default=0.5, help="Minimum qualifying frame fraction for score filtering. Default: 0.5")
    parser.add_argument("--score-filter-diff-max", type=float, default=8.0, help="Reject if every sampled adjacent frame pair differs above this mean RGB delta. Default: 8")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sample outputs.")
    parser.add_argument("--audio", action="store_true", help="Also extract audio.wav.")
    parser.add_argument("--score", action="store_true", help="Also generate score.webp.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env(args.env.expanduser().resolve())
    load_env(args.env_local.expanduser().resolve(), override=True)
    data_dir = resolve_data_dir()
    video_dir = args.video_dir.expanduser().resolve() if args.video_dir else data_dir / "video"
    scores_dir = args.scores_dir.expanduser().resolve() if args.scores_dir else data_dir / "scores"
    if not video_dir.is_dir():
        raise SystemExit(f"Video directory does not exist: {video_dir}")

    videos = find_videos(video_dir)
    if args.video_id:
        wanted = set(args.video_id)
        videos = [video for video in videos if video_id_from_path(video) in wanted]
    if args.limit is not None:
        videos = videos[: args.limit]

    for video_path in videos:
        segment_video(video_path, scores_dir, args)


if __name__ == "__main__":
    main()
