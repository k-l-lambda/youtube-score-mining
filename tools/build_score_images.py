#!/usr/bin/env python3
"""Build score.webp images from segmented score metadata and attach layout results."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTENSIONS = {".webm", ".mkv", ".mp4", ".mov", ".m4v"}
DEFAULT_LAYOUT_API_URL = "http://localhost:3080/api/predict/layout"
SECONDS_RE = re.compile(r"^\s*seconds:\s*([0-9.]+)\s*$")


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


def run_command(args: list[str], *, capture: bool = True) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            args,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"Required executable not found: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            sys.stderr.write(exc.stderr)
        raise SystemExit(f"Command failed with exit code {exc.returncode}: {' '.join(args)}") from exc


def video_id_from_path(video_path: Path) -> str:
    name = video_path.stem
    for marker in [".f137", ".f248", ".f399", ".f401", ".f251"]:
        if name.endswith(marker):
            return name[: -len(marker)]
    return name


def find_video(video_dir: Path, video_id: str) -> Path | None:
    matches = sorted(path for path in video_dir.glob(f"{video_id}.*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS and ".part" not in path.name)
    return matches[0] if matches else None


def find_samples(video_dir: Path, scores_dir: Path, wanted: set[str] | None) -> list[tuple[str, Path, Path]]:
    samples: list[tuple[str, Path, Path]] = []
    for meta_path in sorted(scores_dir.glob("*/meta.yaml")):
        video_id = meta_path.parent.name
        if wanted is not None and video_id not in wanted:
            continue
        video_path = find_video(video_dir, video_id)
        if video_path is not None:
            samples.append((video_id, video_path, meta_path))
    return samples


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


def read_boundaries(meta_path: Path) -> list[float]:
    boundaries = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        match = SECONDS_RE.match(line)
        if match is not None:
            boundaries.append(float(match.group(1)))
    if len(boundaries) < 2:
        raise SystemExit(f"Need at least two shot_detection changes in {meta_path}")
    return boundaries


def strip_existing_layout(meta_text: str) -> str:
    lines = meta_text.rstrip().splitlines()
    for index, line in enumerate(lines):
        if line == "layout:":
            return "\n".join(lines[:index]).rstrip()
    return "\n".join(lines).rstrip()


def yaml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return yaml_quote(str(value))


def yaml_lines(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {yaml_scalar(item)}")
        return lines
    return [f"{prefix}{yaml_scalar(value)}"]


def write_meta_layout(meta_path: Path, layout: dict[str, Any]) -> None:
    base = strip_existing_layout(meta_path.read_text(encoding="utf-8"))
    layout_text = "\n".join(yaml_lines({"layout": layout}))
    meta_path.write_text(f"{base}\n{layout_text}\n", encoding="utf-8")


def extract_frame(video_path: Path, seconds: float, frame_path: Path, width: int) -> None:
    run_command([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        format_seconds(seconds),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:-1",
        str(frame_path),
    ], capture=False)


def post_layout(api_url: str, image_path: Path, timeout: float) -> dict[str, Any]:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = json.dumps({"images": [image_b64]}).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Layout API HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach layout API at {api_url}: {exc.reason}") from exc


def summarize_staves(staves: dict[str, Any] | None) -> dict[str, Any] | None:
    if not staves:
        return None
    middle_rhos = staves.get("middleRhos") or []
    return {
        "interval": round(float(staves.get("interval", 0)), 4),
        "phi1": round(float(staves.get("phi1", 0)), 2),
        "phi2": round(float(staves.get("phi2", 0)), 2),
        "middleRhos": [round(float(rho), 2) for rho in middle_rhos],
    }


def summarize_layout(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("code") != 0:
        return {"code": result.get("code"), "message": result.get("message")}
    data = result.get("data") or []
    layout = data[0] if data else {}
    detection = layout.get("detection") or {}
    areas = detection.get("areas") or []
    staff_counts = [len(((area or {}).get("staves") or {}).get("middleRhos") or []) for area in areas]
    summarized_areas = []
    for area in areas:
        staves = (area or {}).get("staves") or {}
        middle_rhos = staves.get("middleRhos") or []
        summarized_area = {
            "x": round(float(area.get("x", 0)), 2),
            "y": round(float(area.get("y", 0)), 2),
            "width": round(float(area.get("width", 0)), 2),
            "height": round(float(area.get("height", 0)), 2),
            "staves": len(middle_rhos),
        }
        staff_detection = summarize_staves(staves)
        if staff_detection is not None:
            summarized_area["staff_detection"] = staff_detection
        summarized_areas.append(summarized_area)
    return {
        "sourceSize": layout.get("sourceSize"),
        "theta": layout.get("theta"),
        "interval": layout.get("interval"),
        "systems": len(areas),
        "stavesPerSystem": staff_counts,
        "totalStaves": sum(staff_counts),
        "areas": summarized_areas,
    }


def rotate_frame(frame_path: Path, theta: float) -> None:
    image = Image.open(frame_path).convert("RGB")
    angle_degrees = theta * 180.0 / math.pi
    rotated = image.rotate(angle_degrees, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="black")
    rotated.save(frame_path)


def image_size(image_path: Path) -> dict[str, int]:
    image = Image.open(image_path)
    return {"width": image.width, "height": image.height}


def build_stacked_score(frame_paths: list[Path], score_path: Path) -> None:
    if not frame_paths:
        return
    inputs: list[str] = []
    for frame_path in frame_paths:
        inputs.extend(["-i", str(frame_path)])
    filter_complex = "".join(f"[{index}:v]" for index in range(len(frame_paths))) + f"vstack=inputs={len(frame_paths)}[v]"
    score_path.parent.mkdir(parents=True, exist_ok=True)
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


def build_score_image(video_path: Path, meta_path: Path, score_path: Path, args: argparse.Namespace) -> None:
    if score_path.exists() and not args.overwrite:
        print(f"skip existing score {meta_path.parent.name}", flush=True)
        return
    boundaries = read_boundaries(meta_path)
    layout_frames = []
    frame_size: dict[str, int] | None = None
    with tempfile.TemporaryDirectory(prefix="score_frames_", dir=PROJECT_ROOT / "temp") as temp_name:
        temp_dir = Path(temp_name)
        frame_paths: list[Path] = []
        for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
            midpoint = start + (end - start) / 2
            frame_path = temp_dir / f"frame_{index:02d}.png"
            extract_frame(video_path, midpoint, frame_path, args.score_width)
            if frame_size is None:
                frame_size = image_size(frame_path)
            layout_summary: dict[str, Any] | None = None
            if not args.no_layout:
                result = post_layout(args.layout_api_url, frame_path, args.layout_timeout)
                layout_summary = summarize_layout(result)
                theta = float(layout_summary.get("theta") or 0.0)
                rotate_frame(frame_path, theta)
                layout_frames.append({
                    "segment_index": index,
                    "source_seconds": round(midpoint, 3),
                    "source_time": format_timestamp(midpoint),
                    **layout_summary,
                })
            frame_paths.append(frame_path)
        build_stacked_score(frame_paths, score_path)
    if not args.no_layout:
        write_meta_layout(meta_path, {
            "frame_size": frame_size,
            "frames": layout_frames,
        })
    print(f"wrote {score_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build score.webp images from segmented score metadata and attach layout results.")
    parser.add_argument("--env", type=Path, default=PROJECT_ROOT / ".env", help="Base environment file. Default: .env")
    parser.add_argument("--env-local", type=Path, default=PROJECT_ROOT / ".env.local", help="Higher-priority environment file. Default: .env.local")
    parser.add_argument("--video-dir", type=Path, help="Input video directory. Default: $DATA_DIR/video")
    parser.add_argument("--scores-dir", type=Path, help="Score metadata directory. Default: $DATA_DIR/scores")
    parser.add_argument("--video-id", action="append", help="Only process this video ID. Can be used multiple times.")
    parser.add_argument("--limit", type=int, help="Process at most this many score samples.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing score.webp and layout metadata.")
    parser.add_argument("--score-width", type=int, default=1440, help="Width of frames stacked into score.webp. Default: 1440")
    parser.add_argument("--layout-api-url", help="Layout API URL. Default: $LAYOUT_API_URL or local Starry omr-service")
    parser.add_argument("--layout-timeout", type=float, default=120.0, help="Layout API timeout in seconds. Default: 120")
    parser.add_argument("--no-layout", action="store_true", help="Build legacy score.webp without layout API calls or metadata updates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env(args.env.expanduser().resolve())
    load_env(args.env_local.expanduser().resolve(), override=True)
    data_dir = resolve_data_dir()
    video_dir = args.video_dir.expanduser().resolve() if args.video_dir else data_dir / "video"
    scores_dir = args.scores_dir.expanduser().resolve() if args.scores_dir else data_dir / "scores"
    args.layout_api_url = args.layout_api_url or os.environ.get("LAYOUT_API_URL") or DEFAULT_LAYOUT_API_URL
    if not video_dir.is_dir():
        raise SystemExit(f"Video directory does not exist: {video_dir}")
    if not scores_dir.is_dir():
        raise SystemExit(f"Scores directory does not exist: {scores_dir}")
    wanted = set(args.video_id) if args.video_id else None
    samples = find_samples(video_dir, scores_dir, wanted)
    if args.limit is not None:
        samples = samples[: args.limit]
    for video_id, video_path, meta_path in samples:
        print(f"build score {video_id}: {video_path.name}", flush=True)
        build_score_image(video_path, meta_path, meta_path.parent / "score.webp", args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted")
