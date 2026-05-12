#!/usr/bin/env python3
"""Recognize Starry bracket appearances per system and infer staff layout masks."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack
import zmq
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTENSIONS = {".webm", ".mkv", ".mp4", ".mov", ".m4v"}
BRACKETS_PREDICTOR_ENV = "BRACKETS_PREDICTOR"
DEFAULT_BRACKETS_PREDICTOR = "tcp://localhost:12028"
SECONDS_RE = re.compile(r"^\s*seconds:\s*([0-9.]+)\s*$")
CODE_TOKEN_RE = re.compile(r"[{}<>\[\],\-.]")


@dataclass
class SystemLayout:
    frame_index: int
    segment_index: int
    system_index: int
    snapshot_seconds: float
    theta: float
    interval: float
    x: float
    y: float
    staves: int
    phi1: float
    middle_rhos: list[float]
    brackets: str | None = None
    staff_mask: int | None = None
    staff_mask_changed: int | None = None
    bracket_image: str | None = None


@dataclass
class StaffItem:
    left_bounds: list[str]
    right_bounds: list[str]
    conjunction: str = ""

    @classmethod
    def empty(cls) -> "StaffItem":
        return cls(left_bounds=[], right_bounds=[])


@dataclass
class Group:
    type: str
    subs: list["Group"]
    staff: int | None = None

    @property
    def grand(self) -> bool:
        return self.type == "{" and all(sub.staff is not None for sub in self.subs)


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


def run_command(args: list[str], *, capture: bool = True) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE if capture else None, stderr=subprocess.PIPE if capture else None)
    except FileNotFoundError as exc:
        raise SystemExit(f"Required executable not found: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            sys.stderr.write(exc.stderr)
        raise SystemExit(f"Command failed with exit code {exc.returncode}: {' '.join(args)}") from exc


def format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}"


def video_id_from_path(video_path: Path) -> str:
    name = video_path.stem
    for marker in [".f137", ".f248", ".f399", ".f401", ".f251"]:
        if name.endswith(marker):
            return name[: -len(marker)]
    return name


def find_video(video_dir: Path, video_id: str) -> Path | None:
    matches = sorted(path for path in video_dir.glob(f"{video_id}.*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS and ".part" not in path.name)
    return matches[0] if matches else None


def read_scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        return json.loads(value)
    if value == "null":
        return None
    if value in {"true", "false"}:
        return value == "true"
    try:
        if any(char in value for char in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_simple_yaml(lines: list[str], start_index: int, base_indent: int) -> tuple[Any, int]:
    items: list[Any] | dict[str, Any]
    if start_index < len(lines) and lines[start_index].startswith(" " * base_indent + "-"):
        items = []
        index = start_index
        while index < len(lines):
            line = lines[index]
            if not line.startswith(" " * base_indent + "-"):
                break
            rest = line[base_indent + 1 :].strip()
            if rest:
                key, sep, value = rest.partition(":")
                if sep and key.strip():
                    item = {key.strip(): read_scalar(value)}
                    index += 1
                    while index < len(lines):
                        line = lines[index]
                        if line.startswith(" " * base_indent + "-"):
                            break
                        if not line.strip():
                            index += 1
                            continue
                        indent = len(line) - len(line.lstrip(" "))
                        if indent <= base_indent:
                            break
                        stripped = line.strip()
                        sub_key, sub_sep, sub_value = stripped.partition(":")
                        if sub_sep:
                            item[sub_key] = read_scalar(sub_value)
                        index += 1
                    items.append(item)
                else:
                    items.append(read_scalar(rest))
                    index += 1
            else:
                value, index = parse_simple_yaml(lines, index + 1, base_indent + 2)
                items.append(value)
        return items, index
    items = {}
    index = start_index
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent < base_indent:
            break
        if indent > base_indent:
            index += 1
            continue
        stripped = line.strip()
        if stripped.startswith("-"):
            break
        key, sep, value = stripped.partition(":")
        if not sep:
            index += 1
            continue
        if value.strip():
            items[key] = read_scalar(value)
            index += 1
        else:
            nested, index = parse_simple_yaml(lines, index + 1, base_indent + 2)
            items[key] = nested
    return items, index


def read_meta(meta_path: Path) -> dict[str, Any]:
    lines = meta_path.read_text(encoding="utf-8").splitlines()
    meta: dict[str, Any] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line or line.startswith(" "):
            index += 1
            continue
        key, sep, value = line.partition(":")
        if not sep:
            index += 1
            continue
        if value.strip():
            meta[key] = read_scalar(value)
            index += 1
        else:
            nested, index = parse_simple_yaml(lines, index + 1, 2)
            meta[key] = nested
    return meta


def has_staff_layout(meta_path: Path) -> bool:
    return any(line.startswith("staffLayout:") for line in meta_path.read_text(encoding="utf-8").splitlines())


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


def yaml_value(key: str, value: Any) -> str:
    if isinstance(value, str) and key in {"video_id", "method", "role"}:
        return value
    return yaml_scalar(value)


def yaml_lines(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {yaml_value(str(key), item)}")
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


def ordered_meta(meta: dict[str, Any]) -> dict[str, Any]:
    priority = ["video_id", "video_title", "video_channel", "duration", "duration_seconds", "shot_detection", "staff_n", "staffLayout", "layout"]
    output: dict[str, Any] = {}
    for key in priority:
        if key in meta:
            output[key] = meta[key]
    for key, value in meta.items():
        if key not in output:
            output[key] = value
    return output


def update_staff_layout_line(lines: list[str], staff_layout_code: str) -> list[str]:
    output: list[str] = []
    inserted = False
    for line in lines:
        if line.startswith("staffLayout:"):
            if not inserted:
                output.append(f"staffLayout: {yaml_quote(staff_layout_code)}")
                inserted = True
            continue
        output.append(line)
        if not inserted and line.startswith("staff_n:"):
            output.append(f"staffLayout: {yaml_quote(staff_layout_code)}")
            inserted = True
    if not inserted:
        for index, line in enumerate(output):
            if line == "layout:":
                output.insert(index, f"staffLayout: {yaml_quote(staff_layout_code)}")
                inserted = True
                break
    if not inserted:
        output.append(f"staffLayout: {yaml_quote(staff_layout_code)}")
    return output


def patch_area_block(block: list[str], system: SystemLayout | None) -> list[str]:
    output = [
        line for line in block
        if not line.startswith("          bracketsAppearance:") and not line.startswith("          staffMask:")
    ]
    if system is None:
        return output
    output.append(f"          bracketsAppearance: {yaml_quote(system.brackets or '')}")
    output.append(f"          staffMask: {yaml_scalar(system.staff_mask)}")
    return output


def patch_layout_area_blocks(lines: list[str], systems: list[SystemLayout]) -> list[str]:
    by_key = {(system.frame_index, system.system_index): system for system in systems}
    output: list[str] = []
    frame_index = 0
    area_index = 0
    in_layout = False
    in_frames = False
    in_areas = False
    area_block: list[str] | None = None

    def flush_area() -> None:
        nonlocal area_block
        if area_block is not None:
            output.extend(patch_area_block(area_block, by_key.get((frame_index, area_index))))
            area_block = None

    for line in lines:
        if area_block is not None:
            if line == "        -" or line == "    -" or (line and not line.startswith(" ")) or (line.startswith("  ") and not line.startswith("    ")):
                flush_area()
            else:
                area_block.append(line)
                continue

        if line == "layout:":
            in_layout = True
            in_frames = False
            in_areas = False
            output.append(line)
            continue
        if in_layout and line and not line.startswith(" "):
            in_layout = False
            in_frames = False
            in_areas = False
            output.append(line)
            continue
        if in_layout and line == "  frames:":
            in_frames = True
            in_areas = False
            output.append(line)
            continue
        if in_frames and line == "    -":
            frame_index += 1
            area_index = 0
            in_areas = False
            output.append(line)
            continue
        if in_frames and line == "      areas:":
            in_areas = True
            output.append(line)
            continue
        if in_areas and line == "        -":
            area_index += 1
            area_block = [line]
            continue
        output.append(line)

    flush_area()
    return output


def write_bracket_results(meta_path: Path, staff_layout_code: str, systems: list[SystemLayout]) -> None:
    lines = meta_path.read_text(encoding="utf-8").splitlines()
    lines = update_staff_layout_line(lines, staff_layout_code)
    lines = patch_layout_area_blocks(lines, systems)
    meta_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def attach_bracket_results(meta: dict[str, Any], staff_layout_code: str, systems: list[SystemLayout]) -> None:
    meta["staffLayout"] = staff_layout_code
    by_key = {(system.frame_index, system.system_index): system for system in systems}
    for frame_index, frame in enumerate((meta.get("layout") or {}).get("frames") or [], start=1):
        for system_index, area in enumerate(frame.get("areas") or [], start=1):
            system = by_key.get((frame_index, system_index))
            if system is None:
                continue
            area["bracketsAppearance"] = system.brackets
            area["staffMask"] = system.staff_mask




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


def rotate_frame(frame_path: Path, theta: float) -> Image.Image:
    image = Image.open(frame_path).convert("RGB")
    angle_degrees = theta * 180.0 / math.pi
    return image.rotate(angle_degrees, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="black")


def crop_bracket_image(image: Image.Image, system: SystemLayout, output_path: Path) -> None:
    top_mid = system.middle_rhos[0]
    bottom_mid = system.middle_rhos[-1]
    source_x = system.x + system.phi1 - 4 * system.interval
    source_y = system.y + top_mid - 4 * system.interval
    source_width = 8 * system.interval
    source_height = bottom_mid - top_mid + 8 * system.interval
    crop = image.crop((source_x, source_y, source_x + source_width, source_y + source_height))
    output_interval = 8
    output_width = output_interval * 8
    output_height = max(1, round((source_height / system.interval) * output_interval))
    crop.resize((output_width, output_height), Image.Resampling.BICUBIC).save(output_path)


def layout_systems(meta: dict[str, Any]) -> list[SystemLayout]:
    systems: list[SystemLayout] = []
    for frame_index, frame in enumerate((meta.get("layout") or {}).get("frames") or [], start=1):
        theta = float(frame.get("theta") or 0.0)
        interval = float(frame.get("interval") or 0.0)
        snapshot_seconds = float(frame.get("snapshot_seconds") or 0.0)
        segment_index = int(frame.get("segment_index") or frame_index)
        for system_index, area in enumerate(frame.get("areas") or [], start=1):
            staff_detection = area.get("staff_detection") or {}
            middle_rhos = [float(value) for value in staff_detection.get("middleRhos") or []]
            if interval <= 0 or not middle_rhos:
                continue
            systems.append(SystemLayout(
                frame_index=frame_index,
                segment_index=segment_index,
                system_index=system_index,
                snapshot_seconds=snapshot_seconds,
                theta=theta,
                interval=interval,
                x=float(area.get("x") or 0.0),
                y=float(area.get("y") or 0.0),
                staves=int(area.get("staves") or len(middle_rhos)),
                phi1=float(staff_detection.get("phi1") or 0.0),
                middle_rhos=middle_rhos,
            ))
    return systems


def zero_request(address: str, method: str, args: list[Any] | None = None, kwargs: dict[str, Any] | None = None, timeout_ms: int = 300000) -> Any:
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, 15000)
    try:
        socket.connect(address)
        message: dict[str, Any] = {"method": method}
        if args is not None:
            message["args"] = args
        if kwargs is not None:
            message["kwargs"] = kwargs
        socket.send(msgpack.packb(message, use_bin_type=True))
        response = msgpack.unpackb(socket.recv(), raw=False)
    finally:
        socket.close()
        context.term()
    if response.get("code") != 0:
        raise SystemExit(response.get("msg") or "Brackets predictor failed")
    return response.get("data")


def predict_brackets(address: str, image_paths: list[Path], timeout: float) -> list[str | None]:
    buffers = [path.read_bytes() for path in image_paths]
    result = zero_request(address, "predict", kwargs={"buffers": buffers}, timeout_ms=round(timeout * 1000))
    return list(result or [])


def tokenize_code(code: str) -> list[str]:
    return CODE_TOKEN_RE.findall(code or "")


def tokens_to_items(code: str) -> list[StaffItem] | None:
    tokens = tokenize_code(code)
    if not tokens:
        return None
    items = [StaffItem.empty()]
    current = items[0]
    for token in tokens:
        if token in "{<[":
            current.left_bounds.append(token)
            continue
        if token in "}>]":
            current.right_bounds.append(token)
            continue
        if token in {",", "-", "."}:
            current.conjunction = token
            current = StaffItem.empty()
            items.append(current)
    return items


def make_groups_from_sequence(parent: Group, sequence: list[Any]) -> None:
    pairs = {"{": "}", "<": ">", "[": "]"}
    while sequence:
        item = sequence.pop(0)
        if isinstance(item, int):
            parent.subs.append(Group(type="staff", subs=[], staff=item))
            continue
        if item in "}>]" and pairs.get(parent.type) == item:
            return
        if item in pairs:
            group = Group(type=item, subs=[])
            make_groups_from_sequence(group, sequence)
            parent.subs.append(group)


def parse_layout(code: str) -> Group | None:
    try:
        items = tokens_to_items(code)
        if not items:
            return None
        sequence: list[Any] = []
        for index, item in enumerate(items):
            sequence.extend(item.left_bounds)
            sequence.append(index)
            sequence.extend(item.right_bounds)
        root = Group(type="root", subs=[])
        make_groups_from_sequence(root, sequence)
        while len(root.subs) == 1 and root.subs[0].staff is None:
            root = root.subs[0]
        return root
    except Exception:
        return None


def count_staves(code: str) -> int:
    items = tokens_to_items(code)
    return len(items or [])


def partial_mask_code(group: Group, bits: list[int], total_staves: int) -> str:
    staff_status = {index: bits[index] if index < len(bits) else None for index in range(total_staves)}

    def render(node: Group) -> tuple[str | None, bool]:
        if node.staff is not None:
            status = staff_status.get(node.staff)
            return (str(node.staff + 1) if status else None, status is None)
        rendered = [render(sub) for sub in node.subs]
        text = ",".join(value for value, _ in rendered if value)
        partial = any(is_partial for _, is_partial in rendered)
        if not text:
            return None, partial
        if node.type == "{":
            return ("{" + text if partial else "{" + text + "}"), partial
        if node.type == "<":
            return ("<" + text if partial else "<" + text + ">"), partial
        if node.type == "[":
            return ("[" + text if partial else "[" + text + "]"), partial
        return text, partial

    code, _ = render(group)
    return re.sub(r"[_\w]+", "", code or "")


def bits_to_mask(bits: list[int]) -> int:
    return sum((1 << index) for index, bit in enumerate(bits) if bit)


def infer_staff_layout(systems: list[SystemLayout]) -> tuple[str, list[SystemLayout]]:
    staff_total = max((system.staves for system in systems), default=0)
    staff_layout_code = ",".join("" for _ in range(staff_total))
    complete = [system for system in systems if system.staves == staff_total and system.brackets]
    candidates = [system.brackets for system in complete if system.brackets and count_staves(system.brackets) == system.staves and parse_layout(system.brackets)]
    if not candidates:
        for system in systems:
            system.staff_mask = bits_to_mask([1] * system.staves)
            system.staff_mask_changed = system.staff_mask
        return staff_layout_code, systems
    staff_layout_code = max(sorted(set(candidates)), key=candidates.count)
    staff_layout_code = re.sub(r"\{,*\}", lambda match: match.group(0).replace(",", "-"), staff_layout_code)
    layout = parse_layout(staff_layout_code)
    if layout is None:
        return staff_layout_code, systems

    last_system: SystemLayout | None = None
    for system in systems:
        if last_system and system.staves == last_system.staves and system.brackets == last_system.brackets:
            system.staff_mask = last_system.staff_mask
            system.staff_mask_changed = None
            continue
        mask = bits_to_mask([1] * staff_total) if system.staves == staff_total else None
        if system.staves < staff_total and system.brackets and parse_layout(system.brackets):
            def search(bits: list[int]) -> int | None:
                if len(bits) > staff_total:
                    return None
                if sum(bits) == system.staves:
                    return bits_to_mask(bits)
                for bit in [1, 0]:
                    next_bits = [*bits, bit]
                    code = partial_mask_code(layout, next_bits, staff_total)
                    if code == system.brackets:
                        return bits_to_mask(next_bits)
                    if system.brackets.startswith(code):
                        result = search(next_bits)
                        if result is not None:
                            return result
                return None
            mask = search([])
        system.staff_mask = mask
        system.staff_mask_changed = None if last_system and mask == last_system.staff_mask else mask
        last_system = system
    return staff_layout_code, systems


def process_sample(video_id: str, video_path: Path, meta_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    print(f"process {video_id}", flush=True)
    meta = read_meta(meta_path)
    systems = layout_systems(meta)
    if not systems:
        raise SystemExit(f"No layout systems found in {meta_path}")
    print(f"extract {video_id}: frames={len(set(system.frame_index for system in systems))} systems={len(systems)}", flush=True)
    with tempfile.TemporaryDirectory(prefix="brackets_", dir=PROJECT_ROOT / "temp") as temp_name:
        temp_dir = Path(temp_name)
        by_frame: dict[int, list[SystemLayout]] = {}
        for system in systems:
            by_frame.setdefault(system.frame_index, []).append(system)
        bracket_paths: list[Path] = []
        path_systems: list[SystemLayout] = []
        for frame_index, frame_systems in by_frame.items():
            seconds = frame_systems[0].snapshot_seconds
            theta = frame_systems[0].theta
            frame_path = temp_dir / f"frame_{frame_index:03d}.png"
            extract_frame(video_path, seconds, frame_path, args.score_width)
            corrected = rotate_frame(frame_path, theta)
            for system in frame_systems:
                bracket_path = temp_dir / f"frame_{frame_index:03d}_system_{system.system_index:03d}_brackets.png"
                crop_bracket_image(corrected, system, bracket_path)
                bracket_paths.append(bracket_path)
                path_systems.append(system)
        print(f"predict {video_id}: crops={len(bracket_paths)}", flush=True)
        brackets = predict_brackets(args.brackets_predictor, bracket_paths, args.timeout)
        for system, bracket, path in zip(path_systems, brackets, bracket_paths):
            system.brackets = bracket
            if args.save_debug:
                debug_dir = meta_path.parent / "brackets_debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_path = debug_dir / path.name
                Image.open(path).save(debug_path)
                system.bracket_image = debug_path.name
    staff_layout_code, systems = infer_staff_layout(systems)
    write_bracket_results(meta_path, staff_layout_code, systems)
    print(f"done {video_id}: staffLayout={staff_layout_code}", flush=True)
    return {
        "video_id": video_id,
        "staffLayoutCode": staff_layout_code,
        "systems": [
            {
                "frame_index": system.frame_index,
                "segment_index": system.segment_index,
                "system_index": system.system_index,
                "staves": system.staves,
                "bracketsAppearance": system.brackets,
                "staffMask": system.staff_mask,
                "staffMaskBinary": None if system.staff_mask is None else bin(system.staff_mask),
                "staffMaskChanged": system.staff_mask_changed,
                **({"bracketImage": system.bracket_image} if system.bracket_image else {}),
            }
            for system in systems
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Starry bracket recognition per detected system and infer staff layout masks.")
    parser.add_argument("--env", type=Path, default=PROJECT_ROOT / ".env", help="Base environment file. Default: .env")
    parser.add_argument("--env-local", type=Path, default=PROJECT_ROOT / ".env.local", help="Higher-priority environment file. Default: .env.local")
    parser.add_argument("--video-dir", type=Path, help="Input video directory. Default: $DATA_DIR/video")
    parser.add_argument("--scores-dir", type=Path, help="Score metadata directory. Default: $DATA_DIR/scores")
    parser.add_argument("--video-id", action="append", help="Only process this video ID. Can be used multiple times.")
    parser.add_argument("--limit", type=int, help="Process at most this many samples.")
    parser.add_argument("--score-width", type=int, default=1440, help="Width used for frame extraction. Default: 1440")
    parser.add_argument("--brackets-predictor", help=f"ZeroMQ brackets predictor address. Default: ${BRACKETS_PREDICTOR_ENV} or tcp://localhost:12028")
    parser.add_argument("--timeout", type=float, default=300.0, help="Predictor timeout in seconds. Default: 300")
    parser.add_argument("--output", type=Path, help="Write JSON result to this path instead of stdout")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess samples that already have top-level staffLayout")
    parser.add_argument("--save-debug", action="store_true", help="Save per-system bracket crop images under each score directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env(args.env.expanduser().resolve())
    load_env(args.env_local.expanduser().resolve(), override=True)
    data_dir = resolve_data_dir()
    video_dir = args.video_dir.expanduser().resolve() if args.video_dir else data_dir / "video"
    scores_dir = args.scores_dir.expanduser().resolve() if args.scores_dir else data_dir / "scores"
    args.brackets_predictor = args.brackets_predictor or os.environ.get(BRACKETS_PREDICTOR_ENV) or DEFAULT_BRACKETS_PREDICTOR
    wanted = set(args.video_id) if args.video_id else None
    samples = []
    skipped = 0
    for meta_path in sorted(scores_dir.glob("*/meta.yaml")):
        video_id = meta_path.parent.name
        if wanted is not None and video_id not in wanted:
            continue
        if not args.overwrite and has_staff_layout(meta_path):
            print(f"skip existing staffLayout {video_id}", flush=True)
            skipped += 1
            continue
        video_path = find_video(video_dir, video_id)
        if video_path is None:
            print(f"skip missing video {video_id}", flush=True)
            continue
        samples.append((video_id, video_path, meta_path))
    samples.sort(key=lambda sample: (sample[0] != "2AX6vPPVGMw", sample[0]))
    if args.limit is not None:
        samples = samples[: args.limit]
    total = len(samples)
    print(f"selected {total} samples, skipped_existing={skipped}, overwrite={args.overwrite}", flush=True)
    results = []
    for index, (video_id, video_path, meta_path) in enumerate(samples, start=1):
        print(f"[{index}/{total}] start {video_id}", flush=True)
        results.append(process_sample(video_id, video_path, meta_path, args))
    print(f"finished processed={len(results)} skipped_existing={skipped}", flush=True)
    text = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        args.output.expanduser().resolve().write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted")
