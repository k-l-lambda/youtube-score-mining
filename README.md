# YouTube Score Mining

This project explores a lightweight pipeline for mining score-following piano videos from YouTube into paired score images, audio, and timing metadata.

## Motivation

Many public piano videos show stable sheet-music pages while the performance audio plays. These videos can be converted into useful multimodal data for score-image/audio alignment, score following, transcription experiments, and downstream dataset construction.

The current pipeline focuses on simple, auditable preprocessing:

1. Detect shot or page-change boundaries from the video.
2. Identify the first stable score frame after the intro and the last useful frame before the outro.
3. Extract one representative score frame from each stable segment.
4. Stack the frames vertically into a single score image.
5. Store the extracted audio and timing metadata next to the score image.

## Data layout

Each sample is stored under `data/scores/<youtube_id>/`, where the directory name is the YouTube video ID.

```text
data/scores/
  <youtube_id>/
    audio.wav
    score.webp
    meta.yaml
```

### `audio.wav`

Extracted performance audio for the YouTube video.

Expected format for current experiments:

- WAV
- 44.1 kHz
- Mono is preferred for transcription and alignment tools

### `score.webp`

A vertically stacked score image built from representative stable frames in the video.

Current convention:

- One frame is sampled from each stable score segment.
- Frames are stacked from top to bottom in video order.
- The image is intended as a compact visual score proxy, not as a cleaned or typeset score file.

### `meta.yaml`

Timing metadata for the sample.

Example:

```yaml
video_id: ZtIW2r1EalM
duration: "00:05:30.061"
duration_seconds: 330.061
shot_detection:
  method: ffmpeg_scene_score
  threshold: 0.35
  changes:
    - time: "00:00:06.680"
      seconds: 6.680
      role: stable_start_after_intro
    - time: "00:00:42.016"
      seconds: 42.016
      score: 0.6193
    - time: "00:05:30.061"
      seconds: 330.061
      role: final_frame_before_outro
```

Fields:

- `video_id`: YouTube video ID. Must match the sample directory name.
- `duration`: Human-readable video duration as `HH:MM:SS.mmm`.
- `duration_seconds`: Video duration in seconds.
- `shot_detection.method`: Method used to detect shot or page boundaries.
- `shot_detection.threshold`: Threshold used by the detector, if applicable.
- `shot_detection.changes`: Ordered list of timing boundaries.

The `changes` list should include both semantic boundary entries and detected page/shot changes in chronological order.
