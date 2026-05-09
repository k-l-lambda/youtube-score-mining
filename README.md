# YouTube Score Mining

A lightweight data pipeline for turning YouTube score-following piano videos into score metadata, audio, and stacked score images.

## Motivation

Many public piano videos show stable sheet-music pages while performance audio plays. This project converts those videos into a compact dataset format for score-image/audio alignment, score following, transcription experiments, and downstream multimodal dataset construction.

The default workflow is configured through `.env.local`; normal full-dataset runs should not need per-command path arguments.

## Environment

Create `.env.local` in the project root:

```bash
DATA_DIR=./data
LAYOUT_API_URL=http://a-starry-service-host/api/predict/layout
TRANSKUN_V2_CHECKPOINT=~/path/to/TRANSKUN_V2_CHECKPOINT
```

Required:

- `DATA_DIR`: data root used by every tool.

Optional:

- `LAYOUT_API_URL`: Starry layout prediction endpoint used by `tools/build_score_images.py`.
- `TRANSKUN_V2_CHECKPOINT`: Transkun checkpoint used only when MIDI transcription is requested.

Required executables:

- `python3`
- `ffmpeg`
- `ffprobe`
- `yt-dlp`

## Data layout

```text
data/
  id_list.txt
  video/
    <youtube_id>.<ext>
  scores/
    <youtube_id>/
      meta.yaml
      audio.wav
      score.webp
      score2.webp        # optional continuation part for tall scores
      transkun.mid       # optional
```

## Default pipeline

Run from the project root:

```bash
python3 tools/download_youtube_videos.py
python3 tools/segment.py
python3 tools/extract_audio.py
python3 tools/build_score_images.py
```

The scripts load `.env` first and `.env.local` second. Existing outputs are skipped by default, so the sequence can be resumed.

## Optional modes

Build score images without Starry layout prediction:

```bash
python3 tools/build_score_images.py --no-layout
```

Run only the score-video filter:

```bash
python3 tools/segment.py --score-filter-only
```

Also transcribe audio to MIDI with Transkun:

```bash
python3 tools/extract_audio.py --midi --device cuda
```
