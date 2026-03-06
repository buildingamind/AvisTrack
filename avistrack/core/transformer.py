"""
Perspective transformer (optional pipeline step).

Merges the two duplicate implementations:
  - ChamberBroadcaster / processors / geometric.py  (auto-ordering corners)
  - yolo-tracking-lab / src / run_video_pipeline.py  (fixed target size)

Usage
-----
    from avistrack.core.transformer import PerspectiveTransformer

    # From ROI JSON (typical offline use)
    tf = PerspectiveTransformer.from_roi_file("camera_rois.json", "video.mkv")
    warped = tf.transform(frame)

    # From explicit corner list
    tf = PerspectiveTransformer(corners=[[x,y], ...], target_size=(640, 640))
    warped = tf.transform(frame)

If no target_size is given, the output is sized to preserve the original
aspect ratio of the ROI (same behaviour as ChamberBroadcaster's CropProcessor).
"""

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PerspectiveTransformer:
    """
    Wraps a single perspective-transform matrix.

    Parameters
    ----------
    corners : list of 4 [x, y] pairs
        Chamber corner coordinates in the original (distorted) frame.
        The order doesn't matter – they are auto-sorted TL/TR/BR/BL.
    target_size : (width, height) or None
        If given, the output is always this size (yolo-tracking-lab style).
        If None, the output is sized to match the true ROI dimensions
        (ChamberBroadcaster style).
    """

    def __init__(
        self,
        corners: list[list[float]],
        target_size: Optional[tuple[int, int]] = None,
    ):
        self._M, self._out_size = self._build_matrix(corners, target_size)

    # ── Factory helpers ──────────────────────────────────────────────────

    @classmethod
    def from_roi_file(
        cls,
        roi_file: str,
        video_name: str,
        target_size: Optional[tuple[int, int]] = None,
    ) -> "PerspectiveTransformer":
        """
        Load corners for a specific video from a camera_rois.json file.

        Parameters
        ----------
        roi_file   : path to camera_rois.json
        video_name : the key inside the JSON (basename of the video file)
        target_size: optional fixed output size
        """
        path = Path(roi_file)
        if not path.exists():
            raise FileNotFoundError(f"ROI file not found: {roi_file}")

        with open(path) as f:
            rois = json.load(f)

        # Allow lookup by full path or basename
        key = video_name if video_name in rois else Path(video_name).name
        if key not in rois:
            raise KeyError(
                f"No ROI entry for '{video_name}' in {roi_file}. "
                f"Available keys: {list(rois.keys())[:5]} ..."
            )

        corners = rois[key]
        logger.info(f"[Transformer] Loaded ROI for '{key}' from {roi_file}")
        return cls(corners, target_size)

    # ── Public API ───────────────────────────────────────────────────────

    def transform(self, frame: np.ndarray) -> np.ndarray:
        """Apply the perspective transform and return the warped frame."""
        return cv2.warpPerspective(frame, self._M, self._out_size)

    @property
    def output_size(self) -> tuple[int, int]:
        """(width, height) of the output frame."""
        return self._out_size

    @property
    def matrix(self) -> np.ndarray:
        """The 3×3 perspective transform matrix (for external use)."""
        return self._M

    # ── Internal ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_matrix(
        corners: list[list[float]],
        target_size: Optional[tuple[int, int]],
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Auto-sort corners (TL, TR, BR, BL) and compute M.
        """
        pts = np.array(corners, dtype="float32")

        # Sort: TL (min sum), BR (max sum), TR (min diff), BL (max diff)
        s    = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).ravel()
        tl   = pts[np.argmin(s)]
        br   = pts[np.argmax(s)]
        tr   = pts[np.argmin(diff)]
        bl   = pts[np.argmax(diff)]
        src  = np.stack([tl, tr, br, bl], axis=0)

        if target_size is not None:
            w, h = target_size
        else:
            # Preserve actual ROI dimensions (CB style)
            w = int(max(
                np.linalg.norm(br - bl),
                np.linalg.norm(tr - tl),
            ))
            h = int(max(
                np.linalg.norm(tr - br),
                np.linalg.norm(tl - bl),
            ))

        dst = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(src, dst)
        return M, (w, h)
