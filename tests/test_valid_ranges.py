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


def test_nominal_fps_on_a_fast_segment_silently_drops_the_tail():
    """Regression: VR chamber 105A wave4 recorded at up to 39 fps while
    workspace.yaml declares 30.

    Converting a full-hour valid range with the nominal rate stops at frame
    108000, so the last ~23 % of the segment never enters the candidate pool —
    and nothing reports it, because clamping to n_frames-1 is a legitimate
    operation. Callers must pass the segment's TRUE rate
    (``load_segment_fps``), or select from ``raw_aligned_30fps`` whose ``frame``
    column really is a 30 fps tick.
    """
    seg_start = 1_780_000_000.0
    ranges = [{"start_unix": seg_start, "end_unix": seg_start + 3600.0}]
    n_true = 140_400                                   # 3600 s at 39 fps

    (_, f1), = frame_windows(ranges, seg_start, fps=30.0, n_frames=n_true)
    assert f1 == 108_000
    assert (n_true - 1 - f1) / n_true > 0.22           # >22 % of the segment lost

    (_, g1), = frame_windows(ranges, seg_start, fps=39.0, n_frames=n_true)
    assert g1 == n_true - 1                            # nothing lost


def test_nominal_fps_is_exact_on_the_aligned_grid():
    """The same range on ``raw_aligned_30fps`` output: its frame index IS the
    30 fps tick, so the nominal rate is not an approximation but the definition."""
    seg_start = 1_780_000_000.0
    ranges = [{"start_unix": seg_start + 600.0, "end_unix": seg_start + 1200.0}]
    (f0, f1), = frame_windows(ranges, seg_start, fps=30.0, n_frames=108_000)
    assert (f0, f1) == (18_000, 36_000)
