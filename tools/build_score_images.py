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
WEBP_MAX_DIMENSION = 16383
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
    return boundaries


def read_layout_frame_times(meta_path: Path) -> list[tuple[int, float, float]]:
    frames: list[tuple[int, float, float]] = []
    current_segment: int | None = None
    current_start: float | None = None
    current_snapshot: float | None = None
    in_layout = False
    in_frames = False

    def append_current() -> None:
        if current_segment is not None and current_start is not None and current_snapshot is not None:
            frames.append((current_segment, current_start, current_snapshot))

    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if line == "layout:":
            in_layout = True
            in_frames = False
            continue
        if not in_layout:
            continue
        if line and not line.startswith(" "):
            break
        if line == "  frames:":
            in_frames = True
            continue
        if not in_frames:
            continue
        if line.startswith("  ") and not line.startswith("    "):
            break
        if line == "    -":
            append_current()
            current_segment = None
            current_start = None
            current_snapshot = None
            continue
        stripped = line.strip()
        if stripped.startswith("segment_index:"):
            current_segment = int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("start_seconds:"):
            current_start = float(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("snapshot_seconds:"):
            current_snapshot = float(stripped.split(":", 1)[1].strip())
    append_current()
    return frames


def segment_times(meta_path: Path) -> list[tuple[int, float, float]]:
    boundaries = read_boundaries(meta_path)
    if len(boundaries) >= 2:
        return [(index, start, start + (end - start) / 2) for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1)]
    frames = read_layout_frame_times(meta_path)
    if frames:
        return frames
    raise SystemExit(f"Need shot_detection changes or existing layout frame times in {meta_path}")


def strip_existing_generated_fields(meta_text: str) -> str:
    lines = meta_text.rstrip().splitlines()
    kept = []
    for line in lines:
        if line == "layout:":
            break
        if line.startswith("staff_n:"):
            continue
        kept.append(line)
    return "\n".join(kept).rstrip()


def meta_has_layout(meta_path: Path) -> bool:
    return any(line == "layout:" for line in meta_path.read_text(encoding="utf-8").splitlines())


def read_layout_staff_n(meta_path: Path) -> int | None:
    lines = meta_path.read_text(encoding="utf-8").splitlines()
    in_layout = False
    in_staves_per_system = False
    values: list[int] = []
    for line in lines:
        if line == "layout:":
            in_layout = True
            in_staves_per_system = False
            continue
        if not in_layout:
            continue
        if line and not line.startswith(" "):
            break
        stripped = line.strip()
        if stripped == "stavesPerSystem:":
            in_staves_per_system = True
            continue
        if in_staves_per_system:
            if stripped.startswith("- "):
                try:
                    values.append(int(stripped[2:].strip()))
                except ValueError:
                    pass
                continue
            if stripped and not stripped.startswith("-"):
                in_staves_per_system = False
    return max(values) if values else 0


def layout_needs_frame_time_backfill(meta_path: Path) -> bool:
    lines = meta_path.read_text(encoding="utf-8").splitlines()
    for frame_lines in iter_layout_frame_blocks(lines):
        has_start_seconds = any(line.strip().startswith("start_seconds:") for line in frame_lines)
        has_source_time = any(line.strip().startswith("source_time:") for line in frame_lines)
        has_source_seconds = any(line.strip().startswith("source_seconds:") for line in frame_lines)
        if has_source_seconds or (has_source_time and not has_start_seconds):
            return True
    return False


def iter_layout_frame_blocks(lines: list[str]) -> list[list[str]]:
    in_layout = False
    in_frames = False
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if line == "layout:":
            in_layout = True
            in_frames = False
            continue
        if not in_layout:
            continue
        if line and not line.startswith(" "):
            break
        if line == "  frames:":
            in_frames = True
            continue
        if not in_frames:
            continue
        if line.startswith("  ") and not line.startswith("    "):
            break
        if line == "    -":
            if current is not None:
                blocks.append(current)
            current = [line]
            continue
        if current is not None:
            current.append(line)
    if current is not None:
        blocks.append(current)
    return blocks


def backfill_meta_layout_frame_times(meta_path: Path, boundaries: list[float]) -> None:
    lines = meta_path.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line == "  frames:":
            output.append(line)
            index += 1
            while index < len(lines):
                if lines[index].startswith("  ") and not lines[index].startswith("    "):
                    break
                if lines[index] != "    -":
                    output.append(lines[index])
                    index += 1
                    continue
                block = [lines[index]]
                index += 1
                while index < len(lines) and lines[index] != "    -" and not (lines[index].startswith("  ") and not lines[index].startswith("    ")):
                    block.append(lines[index])
                    index += 1
                output.extend(backfill_layout_frame_block(block, boundaries))
            continue
        output.append(line)
        index += 1
    meta_path.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")


def find_interval_start(snapshot_seconds: float, boundaries: list[float]) -> float | None:
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if start <= snapshot_seconds < end:
            return start
    if len(boundaries) >= 2 and snapshot_seconds == boundaries[-1]:
        return boundaries[-2]
    return None


def backfill_layout_frame_block(block: list[str], boundaries: list[float]) -> list[str]:
    snapshot_seconds: float | None = None
    has_start_seconds = False
    has_source_time = False
    for line in block:
        stripped = line.strip()
        if stripped.startswith("source_seconds:") or stripped.startswith("snapshot_seconds:"):
            try:
                snapshot_seconds = float(stripped.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif stripped.startswith("start_seconds:"):
            has_start_seconds = True
        elif stripped.startswith("source_time:"):
            has_source_time = True
    start_seconds = find_interval_start(snapshot_seconds, boundaries) if snapshot_seconds is not None else None
    output: list[str] = []
    for line in block:
        stripped = line.strip()
        if stripped.startswith("source_time:"):
            continue
        if stripped.startswith("source_seconds:"):
            output.append(line.replace("source_seconds:", "snapshot_seconds:", 1))
            continue
        output.append(line)
        if stripped.startswith("segment_index:") and not has_start_seconds and has_source_time and start_seconds is not None:
            output.append(f"      start_seconds: {round(start_seconds, 3)}")
            output.append(f"      start_time: {yaml_quote(format_timestamp(start_seconds))}")
    return output


def layout_staff_n(layout: dict[str, Any]) -> int:
    values = []
    for frame in layout.get("frames") or []:
        values.extend(int(value) for value in (frame.get("stavesPerSystem") or []))
    return max(values) if values else 0


def write_meta_layout(meta_path: Path, layout: dict[str, Any]) -> None:
    base = strip_existing_generated_fields(meta_path.read_text(encoding="utf-8"))
    layout_text = "\n".join(yaml_lines({"layout": layout}))
    meta_path.write_text(f"{base}\nstaff_n: {layout_staff_n(layout)}\n{layout_text}\n", encoding="utf-8")


def write_meta_staff_n(meta_path: Path, staff_n: int) -> None:
    meta_text = meta_path.read_text(encoding="utf-8")
    base = strip_existing_generated_fields(meta_text)
    layout_index = meta_text.rstrip().splitlines().index("layout:")
    layout_text = "\n".join(meta_text.rstrip().splitlines()[layout_index:])
    meta_path.write_text(f"{base}\nstaff_n: {staff_n}\n{layout_text}\n", encoding="utf-8")

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



def extract_frame(video_path: Path, seconds: float, frame_path: Path, width: int) -> bool:
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
    return frame_path.is_file() and frame_path.stat().st_size > 0


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


def crop_to_full_offset(source_size: dict[str, int], crop: dict[str, int], theta: float) -> tuple[float, float]:
    left = float(crop["left"])
    right = float(crop["right"])
    full_width = float(source_size["width"])
    cropped_width = full_width - left - right
    crop_center_x = cropped_width / 2.0
    full_center_x = full_width / 2.0
    content_center_dx = crop_center_x + left - full_center_x
    offset_x = full_center_x - crop_center_x + math.cos(theta) * content_center_dx
    offset_y = math.sin(theta) * content_center_dx
    return offset_x, offset_y


def summarize_layout(result: dict[str, Any], *, source_size: dict[str, int] | None = None, crop: dict[str, int] | None = None) -> dict[str, Any]:
    if result.get("code") != 0:
        return {"code": result.get("code"), "message": result.get("message")}
    data = result.get("data") or []
    layout = data[0] if data else {}
    detection = layout.get("detection") or {}
    areas = detection.get("areas") or []
    theta = float(layout.get("theta") or 0.0)
    x_offset = 0.0
    y_offset = 0.0
    if source_size is not None and crop is not None:
        x_offset, y_offset = crop_to_full_offset(source_size, crop, theta)
    staff_counts = [len(((area or {}).get("staves") or {}).get("middleRhos") or []) for area in areas]
    summarized_areas = []
    for area in areas:
        staves = (area or {}).get("staves") or {}
        middle_rhos = staves.get("middleRhos") or []
        summarized_area = {
            "x": round(float(area.get("x", 0)) + x_offset, 2),
            "y": round(float(area.get("y", 0)) + y_offset, 2),
            "width": round(float(area.get("width", 0)), 2),
            "height": round(float(area.get("height", 0)), 2),
            "staves": len(middle_rhos),
        }
        staff_detection = summarize_staves(staves)
        if staff_detection is not None:
            summarized_area["staff_detection"] = staff_detection
        summarized_areas.append(summarized_area)
    return {
        "sourceSize": source_size or layout.get("sourceSize"),
        "theta": layout.get("theta"),
        "interval": layout.get("interval"),
        "systems": len(areas),
        "stavesPerSystem": staff_counts,
        "totalStaves": sum(staff_counts),
        "areas": summarized_areas,
    }


def crop_black_side_blocks(frame_path: Path, threshold: float) -> dict[str, int]:
    image = Image.open(frame_path).convert("RGB")
    pixels = image.load()
    width, height = image.size

    def column_mean(x: int) -> float:
        total = 0.0
        for y in range(height):
            red, green, blue = pixels[x, y]
            total += (red + green + blue) / 3.0
        return total / height

    left = 0
    while left < width and column_mean(left) <= threshold:
        left += 1
    right = width - 1
    while right >= left and column_mean(right) <= threshold:
        right -= 1
    if left >= width or right < left:
        return {"left": 0, "right": 0}
    right_width = width - right - 1

    if left or right_width:
        image.crop((left, 0, right + 1, height)).save(frame_path)
    return {"left": left, "right": right_width}


def rotate_frame(frame_path: Path, theta: float) -> None:
    image = Image.open(frame_path).convert("RGB")
    angle_degrees = theta * 180.0 / math.pi
    rotated = image.rotate(angle_degrees, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="black")
    rotated.save(frame_path)


def image_size(image_path: Path) -> dict[str, int]:
    image = Image.open(image_path)
    return {"width": image.width, "height": image.height}


def layout_has_score_grid_rows(meta_path: Path) -> bool:
    return any(line.strip().startswith("score_grid_rows:") for line in meta_path.read_text(encoding="utf-8").splitlines())


def backfill_meta_score_grid_rows(meta_path: Path) -> None:
    rows = len(read_layout_frame_times(meta_path))
    if rows <= 0 or layout_has_score_grid_rows(meta_path):
        return
    lines = meta_path.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    inserted = False
    for line in lines:
        output.append(line)
        if line == "  frame_size:":
            continue
        if not inserted and line == "layout:":
            output.append(f"  score_grid_rows: {rows}")
            inserted = True
    meta_path.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")


def read_layout_bracket_results(meta_path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    results: dict[tuple[int, int], dict[str, Any]] = {}
    in_layout = False
    in_frames = False
    frame_index = 0
    area_index = 0
    current_area: dict[str, Any] | None = None

    def flush_area() -> None:
        if current_area and ("bracketsAppearance" in current_area or "staffMask" in current_area):
            results[(frame_index, area_index)] = dict(current_area)

    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if line == "layout:":
            in_layout = True
            in_frames = False
            continue
        if not in_layout:
            continue
        if line and not line.startswith(" "):
            break
        if line == "  frames:":
            in_frames = True
            continue
        if not in_frames:
            continue
        if line.startswith("  ") and not line.startswith("    "):
            break
        stripped = line.strip()
        if line == "    -":
            flush_area()
            frame_index += 1
            area_index = 0
            current_area = None
            continue
        if line == "        -":
            flush_area()
            area_index += 1
            current_area = {}
            continue
        if current_area is None:
            continue
        if stripped.startswith("bracketsAppearance:"):
            current_area["bracketsAppearance"] = json.loads(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("staffMask:"):
            value = stripped.split(":", 1)[1].strip()
            current_area["staffMask"] = None if value == "null" else int(value)
    flush_area()
    return results


def restore_layout_bracket_results(layout_frames: list[dict[str, Any]], bracket_results: dict[tuple[int, int], dict[str, Any]]) -> None:
    for frame_index, frame in enumerate(layout_frames, start=1):
        for area_index, area in enumerate(frame.get("areas") or [], start=1):
            result = bracket_results.get((frame_index, area_index))
            if result:
                area.update(result)


def score_grid_rows(frame_count: int, frame_size: dict[str, int]) -> int:
    if frame_count <= 0:
        return 0
    frame_height = frame_size["height"]
    max_rows = WEBP_MAX_DIMENSION // frame_height
    if max_rows < 1:
        raise ValueError(f"frame height {frame_height} exceeds WebP max dimension {WEBP_MAX_DIMENSION}")
    columns = math.ceil(frame_count / max_rows)
    return math.ceil(frame_count / columns)


def build_score_grid(frame_paths: list[Path], score_path: Path, frame_size: dict[str, int]) -> tuple[list[Path], int]:
    if not frame_paths:
        return [], 0
    rows = score_grid_rows(len(frame_paths), frame_size)
    columns = math.ceil(len(frame_paths) / rows)
    output_width = frame_size["width"] * columns
    output_height = frame_size["height"] * rows
    if output_width > WEBP_MAX_DIMENSION:
        raise ValueError(f"score grid width {output_width} exceeds WebP max dimension {WEBP_MAX_DIMENSION}")
    if output_height > WEBP_MAX_DIMENSION:
        raise ValueError(f"score grid height {output_height} exceeds WebP max dimension {WEBP_MAX_DIMENSION}")
    image = Image.new("RGB", (output_width, output_height), "black")
    for index, frame_path in enumerate(frame_paths):
        row = index % rows
        column = index // rows
        with Image.open(frame_path).convert("RGB") as frame:
            image.paste(frame, (column * frame_size["width"], row * frame_size["height"]))
    score_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(score_path, format="WEBP", quality=90, method=6)
    return [score_path], rows


def build_score_image(video_path: Path, meta_path: Path, score_path: Path, args: argparse.Namespace) -> None:
    has_layout = meta_has_layout(meta_path) if not args.no_layout else False
    if score_path.exists() and not args.overwrite and (args.no_layout or has_layout):
        if has_layout:
            if layout_needs_frame_time_backfill(meta_path):
                backfill_meta_layout_frame_times(meta_path, read_boundaries(meta_path))
            if not layout_has_score_grid_rows(meta_path):
                backfill_meta_score_grid_rows(meta_path)
            staff_n = read_layout_staff_n(meta_path)
            if staff_n is not None:
                write_meta_staff_n(meta_path, staff_n)
        print(f"skip existing score {meta_path.parent.name}", flush=True)
        return
    segment_frames = segment_times(meta_path)
    bracket_results = read_layout_bracket_results(meta_path)
    layout_frames = []
    frame_size: dict[str, int] | None = None
    with tempfile.TemporaryDirectory(prefix="score_frames_", dir=PROJECT_ROOT / "temp") as temp_name:
        temp_dir = Path(temp_name)
        frame_paths: list[Path] = []
        for index, start, midpoint in segment_frames:
            frame_path = temp_dir / f"frame_{index:02d}.png"
            if not extract_frame(video_path, midpoint, frame_path, args.score_width):
                print(f"skip {meta_path.parent.name}: missing frame {index} at {format_timestamp(midpoint)}", flush=True)
                return
            if frame_size is None:
                frame_size = image_size(frame_path)
            layout_summary: dict[str, Any] | None = None
            if not args.no_layout:
                layout_frame_path = temp_dir / f"frame_{index:02d}_layout.png"
                layout_frame_path.write_bytes(frame_path.read_bytes())
                crop = crop_black_side_blocks(layout_frame_path, args.black_side_threshold)
                result = post_layout(args.layout_api_url, layout_frame_path, args.layout_timeout)
                layout_summary = summarize_layout(result, source_size=frame_size, crop=crop)
                theta = float(layout_summary.get("theta") or 0.0)
                rotate_frame(frame_path, theta)
                layout_frames.append({
                    "segment_index": index,
                    "start_seconds": round(start, 3),
                    "start_time": format_timestamp(start),
                    "snapshot_seconds": round(midpoint, 3),
                    **layout_summary,
                })
            frame_paths.append(frame_path)
        try:
            written_paths, grid_rows = build_score_grid(frame_paths, score_path, frame_size or image_size(frame_paths[0]))
        except ValueError as exc:
            print(f"skip {meta_path.parent.name}: {exc}", flush=True)
            return
    if not args.no_layout:
        restore_layout_bracket_results(layout_frames, bracket_results)
        write_meta_layout(meta_path, {
            "frame_size": frame_size,
            "score_grid_rows": grid_rows,
            "frames": layout_frames,
        })
    print(f"wrote {', '.join(str(path) for path in written_paths)}", flush=True)


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
    parser.add_argument("--black-side-threshold", type=float, default=8.0, help="Column mean threshold for cropping black side blocks before layout detection. Default: 8")
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
