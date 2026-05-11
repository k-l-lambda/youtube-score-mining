#!/usr/bin/env python3
"""Extract audio.wav files from downloaded videos into score sample directories."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTENSIONS = {".webm", ".mkv", ".mp4", ".mov", ".m4v"}
DEFAULT_MIDI_NAME = "transkun.mid"
COPY_AUDIO_EXTENSIONS = {
    "aac": ".m4a",
    "alac": ".m4a",
    "mp3": ".mp3",
    "opus": ".webm",
    "vorbis": ".ogg",
    "flac": ".flac",
}


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


def resolve_transkun_paths(weight_arg: Path | None, conf_arg: Path | None) -> tuple[Path, Path]:
    checkpoint_value = os.environ.get("TRANSKUN_V2_CHECKPOINT")
    checkpoint_path = Path(checkpoint_value).expanduser() if checkpoint_value else PROJECT_ROOT / "checkpoints" / "checkpointMSimplerAug"
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if checkpoint_path.is_file() and checkpoint_path.suffix == ".zip":
        checkpoint_path = extract_transkun_checkpoint(checkpoint_path)
    elif not checkpoint_path.exists():
        checkpoint_path = resolve_checkpoint_fallback(checkpoint_path)

    if checkpoint_path.is_dir() and not (checkpoint_path / "checkpoint.pt").is_file():
        nested = next((path.parent for path in checkpoint_path.rglob("checkpoint.pt")), checkpoint_path)
        checkpoint_path = nested
    weight_path = weight_arg.expanduser() if weight_arg else checkpoint_path / "checkpoint.pt"
    conf_path = conf_arg.expanduser() if conf_arg else checkpoint_path / "model.conf"
    return weight_path.resolve(), conf_path.resolve()


def resolve_checkpoint_fallback(checkpoint_path: Path) -> Path:
    candidates = [
        checkpoint_path.with_suffix(".zip"),
        PROJECT_ROOT / "temp" / "checkpointTransformerAug.zip",
        PROJECT_ROOT / "checkpointTransformerAug.zip",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix == ".zip":
            return extract_transkun_checkpoint(candidate)
        if candidate.is_dir():
            return candidate.resolve()
    return checkpoint_path


def extract_transkun_checkpoint(zip_path: Path) -> Path:
    target_dir = PROJECT_ROOT / "temp" / "checkpoints" / zip_path.stem
    checkpoint_file = next(target_dir.rglob("checkpoint.pt"), None) if target_dir.exists() else None
    if checkpoint_file is None:
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(target_dir)
        checkpoint_file = next(target_dir.rglob("checkpoint.pt"), None)
    if checkpoint_file is None:
        raise SystemExit(f"Transkun checkpoint zip does not contain checkpoint.pt: {zip_path}")
    return checkpoint_file.parent


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


def audio_codec(video_path: Path) -> str:
    try:
        result = subprocess.run([
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ], check=True, stdout=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise SystemExit("Required executable not found: ffprobe") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ffprobe failed for {video_path}: exit code {exc.returncode}") from exc
    codec = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not codec:
        raise SystemExit(f"No audio stream found in {video_path}")
    return codec


def copied_audio_path(sample_dir: Path, video_path: Path) -> Path:
    codec = audio_codec(video_path)
    suffix = COPY_AUDIO_EXTENSIONS.get(codec, video_path.suffix.lower())
    return sample_dir / f"audio{suffix}"


def extract_audio_copy(video_path: Path, audio_path: Path) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-map",
            "0:a:0",
            "-c:a",
            "copy",
            str(audio_path),
        ], check=True)
    except FileNotFoundError as exc:
        raise SystemExit("Required executable not found: ffmpeg") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ffmpeg audio copy failed for {video_path}: exit code {exc.returncode}") from exc


def extract_audio(video_path: Path, audio_path: Path) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([
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
        ], check=True)
    except FileNotFoundError as exc:
        raise SystemExit("Required executable not found: ffmpeg") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ffmpeg failed for {video_path}: exit code {exc.returncode}") from exc


def transcribe_source(audio_path: Path, sample_dir: Path) -> Path:
    if audio_path.suffix.lower() == ".wav":
        return audio_path
    wav_path = sample_dir / ".transkun_audio.wav"
    extract_audio(audio_path, wav_path)
    return wav_path


def transcribe_audio(audio_path: Path, midi_path: Path, weight_path: Path, conf_path: Path, device: str) -> None:
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([
            sys.executable,
            "-m",
            "transkun.transcribe",
            str(audio_path),
            str(midi_path),
            "--weight",
            str(weight_path),
            "--conf",
            str(conf_path),
            "--device",
            device,
        ], check=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"Required executable not found: {sys.executable}") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Transkun failed for {audio_path}: exit code {exc.returncode}") from exc


def process_video(video_path: Path, scores_dir: Path, overwrite: bool, midi: bool, copy_audio: bool, weight_path: Path, conf_path: Path, device: str) -> tuple[bool, bool]:
    video_id = video_id_from_path(video_path)
    sample_dir = scores_dir / video_id
    meta_path = sample_dir / "meta.yaml"
    audio_written = False
    midi_written = False
    if not meta_path.is_file():
        print(f"skip missing meta {video_id}", flush=True)
        return audio_written, midi_written
    audio_path = copied_audio_path(sample_dir, video_path) if copy_audio else sample_dir / "audio.wav"
    midi_path = sample_dir / DEFAULT_MIDI_NAME
    if audio_path.exists() and not overwrite:
        print(f"skip existing audio {video_id}: {audio_path.name}", flush=True)
    else:
        if copy_audio:
            print(f"copy audio {video_id}: {video_path.name} -> {audio_path.name}", flush=True)
            extract_audio_copy(video_path, audio_path)
        else:
            print(f"extract audio {video_id}: {video_path.name}", flush=True)
            extract_audio(video_path, audio_path)
        print(f"wrote {audio_path}", flush=True)
        audio_written = True
    if midi:
        if midi_path.exists() and not overwrite:
            print(f"skip existing midi {video_id}", flush=True)
        else:
            transkun_audio_path = transcribe_source(audio_path, sample_dir)
            print(f"transcribe midi {video_id}: {transkun_audio_path.name}", flush=True)
            transcribe_audio(transkun_audio_path, midi_path, weight_path, conf_path, device)
            print(f"wrote {midi_path}", flush=True)
            midi_written = True
    return audio_written, midi_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract audio.wav from downloaded videos into $DATA_DIR/scores/<video_id>/.")
    parser.add_argument("--env", type=Path, default=PROJECT_ROOT / ".env", help="Base environment file. Default: .env")
    parser.add_argument("--env-local", type=Path, default=PROJECT_ROOT / ".env.local", help="Higher-priority environment file. Default: .env.local")
    parser.add_argument("--video-dir", type=Path, help="Input video directory. Default: $DATA_DIR/video")
    parser.add_argument("--scores-dir", type=Path, help="Output scores directory. Default: $DATA_DIR/scores")
    parser.add_argument("--limit", type=int, help="Process at most this many videos.")
    parser.add_argument("--video-id", action="append", help="Only process this video ID. Can be used multiple times.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing audio files and MIDI files.")
    parser.add_argument("--copy-audio", action="store_true", help="Copy the first audio stream without re-encoding, writing audio.m4a/audio.webm/etc. based on codec.")
    parser.add_argument("--midi", action="store_true", help=f"Also transcribe audio to {DEFAULT_MIDI_NAME} using Transkun.")
    parser.add_argument("--transkun-weight", type=Path, help="Transkun checkpoint.pt path. Default: $TRANSKUN_V2_CHECKPOINT/checkpoint.pt")
    parser.add_argument("--transkun-conf", type=Path, help="Transkun model.conf path. Default: $TRANSKUN_V2_CHECKPOINT/model.conf")
    parser.add_argument("--device", default="cpu", help="Transkun device, for example cpu or cuda. Default: cpu")
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
    weight_path, conf_path = resolve_transkun_paths(args.transkun_weight, args.transkun_conf)
    if args.midi:
        if not weight_path.is_file():
            raise SystemExit(f"Transkun weight does not exist: {weight_path}")
        if not conf_path.is_file():
            raise SystemExit(f"Transkun config does not exist: {conf_path}")

    videos = find_videos(video_dir)
    if args.video_id:
        wanted = set(args.video_id)
        videos = [video for video in videos if video_id_from_path(video) in wanted]
    if args.limit is not None:
        videos = videos[: args.limit]

    extracted = 0
    transcribed = 0
    for video_path in videos:
        audio_written, midi_written = process_video(video_path, scores_dir, args.overwrite, args.midi, args.copy_audio, weight_path, conf_path, args.device)
        if audio_written:
            extracted += 1
        if midi_written:
            transcribed += 1
    print(f"extracted {extracted} audio files into {scores_dir}")
    if args.midi:
        print(f"transcribed {transcribed} MIDI files into {scores_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted")
