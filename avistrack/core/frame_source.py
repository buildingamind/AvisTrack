"""
Unified frame source: abstracts over video files and live camera feeds.

Why this exists
---------------
Tracking code should look the same regardless of whether it is processing a
pre-recorded .mkv file or receiving frames streamed from ChamberBroadcaster.

Usage
-----
    # Offline – read a video file
    with FrameSource.from_video("path/to/video.mkv") as src:
        for frame_idx, frame in src:
            detections = tracker.update(frame)

    # Live – receive frames pushed by the caller (e.g. CB processor)
    src = FrameSource.from_live()
    src.push(frame)          # called by CB on each camera tick
    frame_idx, frame = next(src)
"""

import logging
import queue
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Frame = np.ndarray


class FrameSource:
    """
    Iterator that yields (frame_index, frame) tuples.

    Do not instantiate directly – use the class-method factories below.
    """

    # ── Factories ────────────────────────────────────────────────────────

    @classmethod
    def from_video(
        cls,
        video_path: str,
        start_frame: int = 0,
    ) -> "FrameSource":
        """
        Read frames sequentially from a video file.

        Parameters
        ----------
        video_path  : path to a .mkv / .mp4 / etc.
        start_frame : seek to this 0-based frame index before reading
                      (useful when resuming a job or sampling a clip).
        """
        src = cls()
        src._mode  = "file"
        src._path  = Path(video_path)
        src._cap   = cv2.VideoCapture(str(src._path))
        if not src._cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        if start_frame > 0:
            src._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        src._frame_idx = start_frame
        src._total     = int(src._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"[FrameSource] Opened file: {src._path.name} "
                    f"({src._total} frames, starting at {start_frame})")
        return src

    @classmethod
    def from_live(cls, maxsize: int = 2) -> "FrameSource":
        """
        Create a live source. Frames are pushed via push() and consumed
        via iteration.  Intended to be driven by ChamberBroadcaster.

        Parameters
        ----------
        maxsize : internal queue depth (default 2, keeps latency low)
        """
        src = cls()
        src._mode      = "live"
        src._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        src._frame_idx = 0
        src._stopped   = False
        return src

    # ── Live-mode helpers ─────────────────────────────────────────────────

    def push(self, frame: Frame) -> None:
        """Push a frame from an external source (e.g. CB processor thread)."""
        if self._mode != "live":
            raise RuntimeError("push() is only valid for live sources.")
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            # Drop the oldest frame to keep latency low
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(frame)

    def stop(self) -> None:
        """Signal the live source to stop iteration."""
        if self._mode == "live":
            self._stopped = True
            try:
                self._queue.put_nowait(None)   # unblock __next__
            except queue.Full:
                pass

    # ── Iteration ──────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Tuple[int, Frame]]:
        return self

    def __next__(self) -> Tuple[int, Frame]:
        if self._mode == "file":
            return self._next_file()
        else:
            return self._next_live()

    def _next_file(self) -> Tuple[int, Frame]:
        ret, frame = self._cap.read()
        if not ret:
            raise StopIteration
        self._frame_idx += 1
        return self._frame_idx, frame

    def _next_live(self) -> Tuple[int, Frame]:
        frame = self._queue.get()   # blocks until a frame is available
        if frame is None or self._stopped:
            raise StopIteration
        self._frame_idx += 1
        return self._frame_idx, frame

    # ── Context manager ───────────────────────────────────────────────────

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *_) -> None:
        self.release()

    def release(self) -> None:
        if self._mode == "file" and hasattr(self, "_cap"):
            self._cap.release()

    # ── Info ─────────────────────────────────────────────────────────────

    @property
    def total_frames(self) -> Optional[int]:
        """Total frame count (file mode only; None for live)."""
        return getattr(self, "_total", None)

    @property
    def current_index(self) -> int:
        return self._frame_idx
