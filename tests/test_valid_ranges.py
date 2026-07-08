"""tests/test_valid_ranges.py – frame-level valid-range helpers."""
from __future__ import annotations

import numpy as np

from avistrack.valid_ranges import frame_windows, valid_mask


def test_frame_windows_maps_unix_to_frames():
    # seg_start 1000, fps 30 → unix [1010, 1020] → frames [300, 600]
    wins = frame_windows([{"start_unix": 1010.0, "end_unix": 1020.0}],
                         seg_start=1000.0, fps=30.0, n_frames=1000)
    assert wins == [(300, 600)]


def test_frame_windows_clamps_and_drops_empty():
    wins = frame_windows(
        [{"start_unix": 900.0, "end_unix": 1005.0},    # starts before seg → f0 clamps to 0
         {"start_unix": 2000.0, "end_unix": 2000.0}],  # past the video / zero-length → dropped
        seg_start=1000.0, fps=30.0, n_frames=200)
    assert wins == [(0, 150)]


def test_valid_mask():
    frames = np.array([0, 50, 150, 300, 400])
    assert list(valid_mask(frames, [(100, 350)])) == [False, False, True, True, False]


def test_valid_mask_multiple_windows():
    frames = np.array([10, 100, 250, 500])
    assert list(valid_mask(frames, [(0, 50), (400, 600)])) == [True, False, False, True]
