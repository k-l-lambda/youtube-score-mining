#!/usr/bin/env python3
"""Test Starry local layout prediction on one frame from a score video."""

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

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTENSIONS = {".webm", ".mkv", ".mp4", ".mov", ".m4v"}
DEFAULT_API_URL = "http://localhost:3080/api/predict/layout"
DATA_URL_RE = re.compile(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.*)$", re.DOTALL)


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


def run_command(args: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"Required executable not found: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            sys.stderr.write(exc.stderr)
        raise SystemExit(f"Command failed with exit code {exc.returncode}: {' '.join(args)}") from exc


def format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}"


def find_video(video_dir: Path, video_id: str) -> Path:
    matches = sorted(path for path in video_dir.glob(f"{video_id}.*") if path.suffix.lower() in VIDEO_EXTENSIONS and ".part" not in path.name)
    if not matches:
        raise SystemExit(f"Video not found for ID {video_id!r} in {video_dir}")
    return matches[0]


def default_video_id(scores_dir: Path, video_dir: Path) -> str:
    for meta_path in sorted(scores_dir.glob("*/meta.yaml")):
        video_id = meta_path.parent.name
        try:
            find_video(video_dir, video_id)
        except SystemExit:
            continue
        return video_id
    raise SystemExit(f"No score sample with both meta.yaml and video found under {scores_dir}")


def read_meta_seconds(meta_path: Path, fallback: float) -> float:
    if not meta_path.is_file():
        return fallback
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("seconds:"):
            try:
                seconds = float(stripped.split(":", 1)[1].strip())
            except ValueError:
                continue
            return max(seconds + 1.0, 0.0)
    return fallback


def extract_frame(video_path: Path, seconds: float, image_path: Path, width: int) -> None:
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
        str(image_path),
    ])


def render_layout_overlay(source_path: Path, result: dict[str, Any], output_path: Path) -> bool:
    data = result.get("data") or []
    layout = data[0] if data else {}
    areas = ((layout.get("detection") or {}).get("areas") or []) if isinstance(layout, dict) else []
    if not areas:
        return False

    image = Image.open(source_path).convert("RGB")
    theta = float(layout.get("theta") or 0.0)
    angle_degrees = theta * 180.0 / math.pi
    rotated = image.rotate(angle_degrees, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white")
    draw = ImageDraw.Draw(rotated)
    center_x = image.width / 2.0
    center_y = image.height / 2.0
    angle = math.radians(angle_degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    def rotate_point(x: float, y: float) -> tuple[float, float]:
        dx = x - center_x
        dy = y - center_y
        return center_x + cos_a * dx - sin_a * dy, center_y + sin_a * dx + cos_a * dy

    for area in areas:
        x = float(area.get("x", 0))
        y = float(area.get("y", 0))
        width = float(area.get("width", 0))
        height = float(area.get("height", 0))
        corners = [
            rotate_point(x, y),
            rotate_point(x + width, y),
            rotate_point(x + width, y + height),
            rotate_point(x, y + height),
        ]
        draw.line([*corners, corners[0]], fill="red", width=3)
        staves = (area.get("staves") or {}) if isinstance(area, dict) else {}
        for rho in staves.get("middleRhos") or []:
            line_y = y + float(rho)
            draw.line([rotate_point(x, line_y), rotate_point(x + width, line_y)], fill="blue", width=2)

    rotated.save(output_path)
    return True


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


def summarize_layout(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("code") != 0:
        return {"code": result.get("code"), "message": result.get("message")}
    data = result.get("data") or []
    layout = data[0] if data else {}
    detection = layout.get("detection") or {}
    areas = detection.get("areas") or []
    staff_counts = [len(((area or {}).get("staves") or {}).get("middleRhos") or []) for area in areas]
    return {
        "sourceSize": layout.get("sourceSize"),
        "theta": layout.get("theta"),
        "interval": layout.get("interval"),
        "systems": len(areas),
        "stavesPerSystem": staff_counts,
        "totalStaves": sum(staff_counts),
        "areas": [
            {
                "x": round(float(area.get("x", 0)), 2),
                "y": round(float(area.get("y", 0)), 2),
                "width": round(float(area.get("width", 0)), 2),
                "height": round(float(area.get("height", 0)), 2),
                "staves": len(((area or {}).get("staves") or {}).get("middleRhos") or []),
                "staffImages": len((area or {}).get("staff_images") or []),
            }
            for area in areas
        ],
    }


def decode_image_value(value: str) -> tuple[bytes, str] | None:
    match = DATA_URL_RE.match(value)
    if match:
        ext = match.group(1).lower().replace("jpeg", "jpg")
        payload = match.group(2)
    else:
        ext = "png"
        payload = value
    try:
        return base64.b64decode(payload, validate=True), ext
    except binascii.Error:
        return None


def collect_debug_images(obj: Any, path: tuple[str, ...] = ()) -> list[tuple[str, bytes, str]]:
    images: list[tuple[str, bytes, str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = (*path, str(key))
            if key == "image" and isinstance(value, str):
                decoded = decode_image_value(value)
                if decoded is not None:
                    data, ext = decoded
                    images.append(("_".join(child_path), data, ext))
            images.extend(collect_debug_images(value, child_path))
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            images.extend(collect_debug_images(item, (*path, f"{index:02d}")))
    return images


def save_debug_images(result: dict[str, Any], debug_dir: Path, video_id: str, frame_path: Path) -> list[Path]:
    debug_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    source_frame = debug_dir / f"{video_id}_source_frame.png"
    source_frame.write_bytes(frame_path.read_bytes())
    saved.append(source_frame)
    overlay_path = debug_dir / f"{video_id}_layout_overlay.png"
    if render_layout_overlay(source_frame, result, overlay_path):
        saved.append(overlay_path)
    for index, (name, data, ext) in enumerate(collect_debug_images(result), start=1):
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "image"
        output_path = debug_dir / f"{video_id}_{index:02d}_{safe_name}.{ext}"
        output_path.write_bytes(data)
        saved.append(output_path)
    return saved


def strip_debug_images(obj: Any) -> Any:
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if key == "image" and isinstance(value, str):
                result[key] = f"<image base64 length={len(value)}>"
            else:
                result[key] = strip_debug_images(value)
        return result
    if isinstance(obj, list):
        return [strip_debug_images(item) for item in obj]
    return obj


def yaml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


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
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                lines.extend(yaml_lines(item, indent + 2))
            elif isinstance(item, list):
                lines.append(f"{prefix}-")
                lines.extend(yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {yaml_scalar(item)}")
        return lines
    return [f"{prefix}{yaml_scalar(value)}"]


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return yaml_quote(str(value))


def write_layout_yaml(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(yaml_lines(summary)) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract one score-video frame and test Starry local /api/predict/layout.")
    parser.add_argument("--env", type=Path, default=PROJECT_ROOT / ".env", help="Base environment file. Default: .env")
    parser.add_argument("--env-local", type=Path, default=PROJECT_ROOT / ".env.local", help="Higher-priority environment file. Default: .env.local")
    parser.add_argument("--video-dir", type=Path, help="Input video directory. Default: $DATA_DIR/video")
    parser.add_argument("--scores-dir", type=Path, help="Score metadata directory. Default: $DATA_DIR/scores")
    parser.add_argument("--video-id", help="Video ID to test. Default: first score directory with meta.yaml and a matching video")
    parser.add_argument("--seconds", type=float, help="Frame timestamp. Default: first meta.yaml seconds value plus 1 second")
    parser.add_argument("--width", type=int, default=1200, help="Frame width sent to layout API. Default: 1200")
    parser.add_argument("--api-url", help="Layout API URL. Default: $LAYOUT_API_URL or http://localhost:3080/api/predict/layout")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds. Default: 120")
    parser.add_argument("--keep-frame", type=Path, help="Write the extracted PNG frame to this path instead of a temporary file")
    parser.add_argument("--debug-dir", type=Path, default=PROJECT_ROOT / "temp" / "layout-debug", help="Directory for source frame, model output images, and YAML summary. Default: temp/layout-debug")
    parser.add_argument("--layout-yaml", type=Path, help="Write numeric layout summary to this YAML path. Default: <debug-dir>/<video_id>_layout.yaml")
    parser.add_argument("--no-debug-images", action="store_true", help="Do not save debug images")
    parser.add_argument("--raw", action="store_true", help="Print the raw API JSON response with image payloads replaced by length markers")
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
    if not scores_dir.is_dir():
        raise SystemExit(f"Scores directory does not exist: {scores_dir}")

    video_id = args.video_id or default_video_id(scores_dir, video_dir)
    video_path = find_video(video_dir, video_id)
    seconds = args.seconds if args.seconds is not None else read_meta_seconds(scores_dir / video_id / "meta.yaml", 10.0)
    api_url = args.api_url or os.environ.get("LAYOUT_API_URL") or DEFAULT_API_URL

    if args.keep_frame:
        frame_path = args.keep_frame.expanduser().resolve()
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        extract_frame(video_path, seconds, frame_path, args.width)
        result = post_layout(api_url, frame_path, args.timeout)
        debug_paths = [] if args.no_debug_images else save_debug_images(result, args.debug_dir.expanduser().resolve(), video_id, frame_path)
    else:
        with tempfile.TemporaryDirectory(prefix="layout_frame_") as temp_name:
            frame_path = Path(temp_name) / f"{video_id}.png"
            extract_frame(video_path, seconds, frame_path, args.width)
            result = post_layout(api_url, frame_path, args.timeout)
            debug_paths = [] if args.no_debug_images else save_debug_images(result, args.debug_dir.expanduser().resolve(), video_id, frame_path)

    output = strip_debug_images(result) if args.raw else summarize_layout(result)
    debug_dir = args.debug_dir.expanduser().resolve()
    yaml_path = args.layout_yaml.expanduser().resolve() if args.layout_yaml else debug_dir / f"{video_id}_layout.yaml"
    yaml_summary = {
        "video_id": video_id,
        "seconds": seconds,
        "api_url": api_url,
        "layout": summarize_layout(result),
    }
    write_layout_yaml(yaml_summary, yaml_path)
    print(json.dumps({
        "video_id": video_id,
        "seconds": seconds,
        "api_url": api_url,
        "layout_yaml": str(yaml_path),
        "debug_images": [str(path) for path in debug_paths],
        "layout": output,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
