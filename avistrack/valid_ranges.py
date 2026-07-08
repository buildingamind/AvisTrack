"""
avistrack.valid_ranges
───────────────────────
Frame-level valid-range filtering.

``valid_ranges.json`` lists, per remuxed video, the unix-time windows during
which the experiment was actually running::

    {"<video>.mp4": [{"start_unix": ..., "end_unix": ..., "start": "...", ...}]}

To intersect those windows with the model's per-frame tracking output we map
unix → frame index using the ChamberBroadcaster **segment-start** time base
(``load_segment_starts`` over ``timestamp_calibration.jsonl``):

    frame = (unix - segment_start_unix) * fps

This is deliberately NOT the tracking parquet's own ``unix_time`` column,
which uses a different base. Selecting/sampling frames without this filter
would include pre-experiment empty-cage footage, so callers must apply it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np


def load_valid_ranges(path: Union[str, Path]) -> dict:
    """Load valid_ranges.json → {video_name: [ {start_unix, end_unix, ...}, ... ]}.
    Returns {} if the file is absent."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f) or {}


def frame_windows(ranges: list[dict], seg_start: float, fps: float,
                  n_frames: int) -> list[tuple[int, int]]:
    """Convert a video's unix-time ranges into inclusive [f0, f1] frame
    windows, clamped to [0, n_frames-1]. Empty/degenerate windows dropped."""
    wins: list[tuple[int, int]] = []
    for r in ranges:
        f0 = int(round((r["start_unix"] - seg_start) * fps))
        f1 = int(round((r["end_unix"] - seg_start) * fps))
        f0 = max(0, f0)
        f1 = min(n_frames - 1, f1)
        if f1 > f0:
            wins.append((f0, f1))
    return wins


def valid_mask(frames, windows: list[tuple[int, int]]) -> np.ndarray:
    """Boolean mask over ``frames`` (array-like of frame indices): True where
    the frame falls inside any window."""
    frames = np.asarray(frames)
    mask = np.zeros(len(frames), dtype=bool)
    for f0, f1 in windows:
        mask |= (frames >= f0) & (frames <= f1)
    return mask
