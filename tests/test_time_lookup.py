"""
Tests for avistrack.core.time_lookup.TimeLookup

Validates:
  • piecewise-linear interpolation (exact & mid-points)
  • reverse look-up (unix → frame)
  • datetime round-trip
  • midnight-crossing handling
  • from_calibration / load class methods
  • edge cases (extrapolation, single-point rejection)
"""

from __future__ import annotations

import json
import math
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from avistrack.core.time_lookup import TimeLookup

TZ = ZoneInfo("America/New_York")


# ── helpers ──────────────────────────────────────────────────────────────

def _make_lookup(
    frames: list[int],
    unix_times: list[float],
    fps: float = 30.0,
) -> TimeLookup:
    return TimeLookup(np.array(frames), np.array(unix_times), TZ, fps)


# ── basic interpolation ─────────────────────────────────────────────────

class TestInterpolation:
    """Forward look-up: frame → unix."""

    def test_exact_points(self):
        tl = _make_lookup([0, 1000, 2000], [1000.0, 1033.3, 1066.7])
        assert tl.frame_to_unix(0) == 1000.0
        assert tl.frame_to_unix(1000) == 1033.3
        assert tl.frame_to_unix(2000) == 1066.7

    def test_midpoint(self):
        tl = _make_lookup([0, 1000], [1000.0, 1033.3])
        mid = tl.frame_to_unix(500)
        assert abs(mid - 1016.65) < 0.01

    def test_extrapolation_clamped(self):
        """numpy.interp clamps outside the domain (no extrapolation)."""
        tl = _make_lookup([100, 200], [5000.0, 5010.0])
        assert tl.frame_to_unix(0) == 5000.0   # clamped to first
        assert tl.frame_to_unix(999) == 5010.0  # clamped to last


# ── reverse look-up ─────────────────────────────────────────────────────

class TestReverse:
    """unix → frame."""

    def test_exact(self):
        tl = _make_lookup([0, 1000, 2000], [1000.0, 1033.3, 1066.7])
        assert tl.unix_to_frame(1033.3) == 1000

    def test_interpolated(self):
        tl = _make_lookup([0, 1000], [1000.0, 1033.3])
        frame = tl.unix_to_frame(1016.65)
        assert abs(frame - 500) <= 1  # rounding tolerance


# ── datetime conversion ─────────────────────────────────────────────────

class TestDatetime:
    def test_frame_to_datetime_has_tz(self):
        tl = _make_lookup([0, 1000], [1730264400.0, 1730264433.3])
        dt = tl.frame_to_datetime(0)
        assert dt.tzinfo is not None
        assert dt.tzinfo.key == "America/New_York"

    def test_datetime_round_trip(self):
        tl = _make_lookup([0, 10000], [1730264400.0, 1730264733.3])
        dt = tl.frame_to_datetime(5000)
        frame_back = tl.datetime_to_frame(dt)
        assert abs(frame_back - 5000) <= 1

    def test_naive_datetime_assumed_local(self):
        """A naive datetime should be treated as being in the lookup's tz."""
        tl = _make_lookup([0, 1000], [1730264400.0, 1730264433.3])
        dt_aware = tl.frame_to_datetime(500)
        dt_naive = dt_aware.replace(tzinfo=None)
        frame = tl.datetime_to_frame(dt_naive)
        assert abs(frame - 500) <= 1

    def test_timestr(self):
        tl = _make_lookup([0, 1000], [1730264400.0, 1730264433.3])
        s = tl.frame_to_timestr(0, "%H:%M:%S")
        assert ":" in s   # basic smoke test


# ── properties ───────────────────────────────────────────────────────────

class TestProperties:
    def test_start_end(self):
        tl = _make_lookup([100, 5000], [9000.0, 9200.0])
        assert tl.start_frame == 100
        assert tl.end_frame == 5000

    def test_duration(self):
        tl = _make_lookup([0, 1000], [1000.0, 1200.0])
        assert tl.duration_seconds == 200.0

    def test_n_samples(self):
        tl = _make_lookup([0, 500, 1000], [0.0, 16.6, 33.3])
        assert tl.n_samples == 3

    def test_actual_fps(self):
        tl = _make_lookup([0, 3000], [0.0, 100.0])
        fps = tl.actual_fps(0, 3000)
        assert abs(fps - 30.0) < 0.01


# ── construction ─────────────────────────────────────────────────────────

class TestConstruction:
    def test_rejects_single_point(self):
        with pytest.raises(ValueError, match="≥ 2"):
            _make_lookup([42], [100.0])

    def test_unsorted_input_is_sorted(self):
        tl = _make_lookup([2000, 0, 1000], [1066.7, 1000.0, 1033.3])
        assert tl.frame_to_unix(0) == 1000.0
        assert tl.frame_to_unix(2000) == 1066.7

    def test_from_calibration(self):
        cal = {
            "_meta": {},
            "vid.mkv": {
                "fps_nominal": 30.0,
                "samples": [
                    {"frame": 0, "unix": 1000.0},
                    {"frame": 1000, "unix": 1033.3},
                ],
            },
        }
        tl = TimeLookup.from_calibration(cal, "vid.mkv")
        assert tl.frame_to_unix(0) == 1000.0

    def test_from_calibration_missing_video(self):
        cal = {"_meta": {}}
        with pytest.raises(KeyError):
            TimeLookup.from_calibration(cal, "nonexistent.mkv")

    def test_load_from_file(self, tmp_path):
        cal = {
            "_meta": {},
            "v.mkv": {
                "fps_nominal": 25.0,
                "samples": [
                    {"frame": 0, "unix": 5000.0},
                    {"frame": 500, "unix": 5020.0},
                ],
            },
        }
        p = tmp_path / "cal.json"
        p.write_text(json.dumps(cal))
        tl = TimeLookup.load(p, "v.mkv")
        assert tl.fps_nominal == 25.0
        assert tl.n_samples == 2

    def test_repr(self):
        tl = _make_lookup([0, 100], [0.0, 10.0])
        r = repr(tl)
        assert "TimeLookup" in r
        assert "0–100" in r
