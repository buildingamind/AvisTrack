"""
Bidirectional frame ↔ time conversion using sparse OCR calibration data.

TimeLookup uses piecewise-linear interpolation (numpy.interp) over
a sparse set of (frame, unix_timestamp) pairs produced by
``tools/calibrate_time.py``.  It provides:

* Frame → Unix timestamp  /  datetime  /  human-readable string
* Unix timestamp / datetime → frame index (reverse lookup)

Both "burn-in local time" and "unix time" are supported.

Usage
-----
>>> from avistrack.core.time_lookup import TimeLookup
>>> tl = TimeLookup.load("time_calibration.json", "Video1_RGB.mkv")
>>> tl.frame_to_datetime(5000)
datetime.datetime(2025, 10, 30, 0, 2, 47, tzinfo=zoneinfo.ZoneInfo(key='America/New_York'))
>>> tl.unix_to_frame(1730264567.0)
5000
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


class TimeLookup:
    """
    Bidirectional frame ↔ time conversion via piecewise-linear interpolation.

    Parameters
    ----------
    frames : np.ndarray
        Sorted 1-D array of calibration frame indices.
    unix_times : np.ndarray
        Corresponding Unix timestamps (float seconds since epoch).
    tz : ZoneInfo
        Timezone for local-time conversions.
    fps_nominal : float
        Nominal FPS reported by the video container (for reference only).
    """

    def __init__(
        self,
        frames: np.ndarray,
        unix_times: np.ndarray,
        tz: ZoneInfo,
        fps_nominal: float = 30.0,
    ):
        if len(frames) < 2:
            raise ValueError(
                f"Need ≥ 2 calibration points, got {len(frames)}"
            )
        order = np.argsort(frames)
        self._frames = frames[order].astype(np.float64)
        self._unix = unix_times[order].astype(np.float64)
        self._tz = tz
        self.fps_nominal = fps_nominal

    # ── Frame → time ─────────────────────────────────────────────

    def frame_to_unix(self, frame_idx: int | float) -> float:
        """Convert frame index to Unix timestamp (piecewise-linear)."""
        return float(np.interp(frame_idx, self._frames, self._unix))

    def frame_to_datetime(self, frame_idx: int | float) -> datetime:
        """Convert frame index to timezone-aware local datetime."""
        return datetime.fromtimestamp(
            self.frame_to_unix(frame_idx), tz=self._tz
        )

    def frame_to_timestr(
        self, frame_idx: int | float, fmt: str = "%Y-%m-%d %H:%M:%S"
    ) -> str:
        """Convert frame index to a formatted local-time string."""
        return self.frame_to_datetime(frame_idx).strftime(fmt)

    # ── Time → frame ─────────────────────────────────────────────

    def unix_to_frame(self, unix_ts: float) -> int:
        """Convert Unix timestamp to the nearest frame index."""
        return int(round(float(np.interp(unix_ts, self._unix, self._frames))))

    def datetime_to_frame(self, dt: datetime) -> int:
        """Convert a datetime to the nearest frame index.

        If *dt* is naive it is assumed to be in `self._tz`.
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self._tz)
        return self.unix_to_frame(dt.timestamp())

    # ── Convenience properties ───────────────────────────────────

    @property
    def start_frame(self) -> int:
        return int(self._frames[0])

    @property
    def end_frame(self) -> int:
        return int(self._frames[-1])

    @property
    def start_time(self) -> datetime:
        return self.frame_to_datetime(self.start_frame)

    @property
    def end_time(self) -> datetime:
        return self.frame_to_datetime(self.end_frame)

    @property
    def duration_seconds(self) -> float:
        """Total duration spanned by the calibration data (seconds)."""
        return float(self._unix[-1] - self._unix[0])

    @property
    def n_samples(self) -> int:
        return len(self._frames)

    def actual_fps(self, frame_a: int, frame_b: int) -> float:
        """Compute the effective FPS between two frame indices."""
        dt = self.frame_to_unix(frame_b) - self.frame_to_unix(frame_a)
        if dt == 0:
            return self.fps_nominal
        return abs(frame_b - frame_a) / dt

    # ── Construction helpers ─────────────────────────────────────

    @classmethod
    def from_calibration(
        cls,
        cal_data: dict,
        video_name: str,
        timezone_str: str = "America/New_York",
    ) -> "TimeLookup":
        """Build from a parsed ``time_calibration.json`` dict.

        Parameters
        ----------
        cal_data : dict
            The full JSON object (contains ``_meta`` + per-video entries).
        video_name : str
            Key identifying the video in *cal_data*.
        timezone_str : str
            IANA timezone name.
        """
        if video_name not in cal_data:
            raise KeyError(f"No calibration data for '{video_name}'")

        vdata = cal_data[video_name]
        tz = ZoneInfo(timezone_str)
        fps = vdata.get("fps_nominal", 30.0)

        frames, unix_times = [], []
        for s in vdata["samples"]:
            frames.append(s["frame"])
            unix_times.append(s["unix"])

        return cls(np.array(frames), np.array(unix_times), tz, fps)

    @classmethod
    def load(
        cls,
        calibration_path: str | Path,
        video_name: str,
        timezone_str: str = "America/New_York",
    ) -> "TimeLookup":
        """Load from a ``time_calibration.json`` file on disk."""
        with open(calibration_path) as f:
            data = json.load(f)
        return cls.from_calibration(data, video_name, timezone_str)

    def __repr__(self) -> str:
        return (
            f"TimeLookup(frames={self.start_frame}–{self.end_frame}, "
            f"samples={self.n_samples}, fps_nom={self.fps_nominal})"
        )
