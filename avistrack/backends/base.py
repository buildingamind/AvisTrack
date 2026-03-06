"""
Abstract base class that every backend must implement.

All backends share the same contract:
  - update(frame)  →  list[Detection]

This means the rest of the codebase (CLI, ChamberBroadcaster wrapper, eval)
never needs to know which model is running underneath.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Detection:
    """
    A single detected / tracked entity in one frame.

    For bounding-box models (YOLO):
        x, y, w, h are filled; keypoints is empty.

    For keypoint models (DLC, ViT):
        keypoints is filled; x, y, w, h may be 0.0.
    """
    track_id: int                      # persistent ID across frames (-1 = unassigned)
    x: float                           # top-left x  (perspective-corrected coords)
    y: float                           # top-left y
    w: float                           # bounding box width
    h: float                           # bounding box height
    confidence: float = 1.0
    keypoints: list[dict] = field(default_factory=list)
    # keypoints element: {"label": str, "x": float, "y": float, "likelihood": float}


class TrackerBackend(ABC):
    """Abstract base. Every backend must implement update()."""

    @abstractmethod
    def update(self, frame: Any) -> list[Detection]:
        """
        Process one frame and return the list of detections for that frame.

        Parameters
        ----------
        frame : np.ndarray  (H x W x C, BGR, uint8)
            The frame to process.  Should already be perspective-corrected
            if transformer.py is used upstream.

        Returns
        -------
        list[Detection]
        """

    def release(self) -> None:
        """Release any held resources (GPU memory, threads, etc.)."""
