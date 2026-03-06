"""
YOLO + Kalman Filter – real-time, single-subject, non-blocking backend.

Ported from ChamberBroadcaster/chamber_broadcaster/processors/yolo_kalman_processor.py
Adapted to return standard Detection objects instead of the CB-specific context dict.
"""

import queue
import threading
import logging
from typing import Any

import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

from avistrack.backends.base import TrackerBackend, Detection

logger = logging.getLogger(__name__)


class YoloRealtimeTracker(TrackerBackend):
    """
    Non-blocking YOLO + Kalman tracker for real-time single-subject tracking.

    Designed to be called from ChamberBroadcaster (or any live frame source).
    Each call to update() is non-blocking: it submits the frame to a background
    thread and immediately returns the most recently computed detection.

    Supports only num_subjects = 1 (single-bird tracking).
    For multi-subject offline tracking use YoloOfflineTracker.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        model_cfg = cfg.model

        # ── Config ──────────────────────────────────────────────────
        self.model_path  = model_cfg.weights
        self.keypoint_name = model_cfg.get("keypoint_name", "center")

        kalman_cfg = model_cfg.get("kalman_filter", {})
        self.kalman_R = float(kalman_cfg.get("R", 5.0))
        self.kalman_Q = float(kalman_cfg.get("Q", 0.001))
        self.kalman_P = float(kalman_cfg.get("P", 1000.0))

        # ── State ────────────────────────────────────────────────────
        self._last_detections: list[Detection] = []
        self._frame_idx = 0

        # ── Model ────────────────────────────────────────────────────
        logger.info(f"[YoloRealtime] Loading model: {self.model_path}")
        self._model = YOLO(self.model_path)

        # ── Kalman filter ────────────────────────────────────────────
        self._kf = self._make_kalman()
        self._kf_initialized = False

        # ── Background thread ────────────────────────────────────────
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("[YoloRealtime] Background thread started.")

    # ── Public API ───────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> list[Detection]:
        """Non-blocking update. Returns last known detection immediately."""
        self._frame_idx += 1

        item = (frame.copy(), self._frame_idx)

        # Discard stale queued frame
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(item)
        except queue.Full:
            pass

        return self._last_detections

    def release(self) -> None:
        logger.info("[YoloRealtime] Releasing resources...")
        if self._thread.is_alive():
            self._queue.put(None)   # sentinel
            self._thread.join()

    # ── Internal ─────────────────────────────────────────────────────────

    def _make_kalman(self) -> KalmanFilter:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0],
                          [0, 1, 0, 1],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=float)
        kf.H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]], dtype=float)
        kf.R *= self.kalman_R
        kf.P *= self.kalman_P
        kf.Q *= self.kalman_Q
        return kf

    def _loop(self):
        while True:
            try:
                item = self._queue.get()
                if item is None:
                    break

                frame, frame_idx = item

                # ── YOLO detection ───────────────────────────────────
                results = self._model(frame, verbose=False, half=True)
                detected_center = None
                confidence = 0.0

                if results and results[0].boxes and len(results[0].boxes) > 0:
                    box = results[0].boxes[0]
                    x1, y1, x2, y2 = box.xyxy[0].cpu()
                    confidence = float(box.conf[0].cpu())
                    detected_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

                # ── Kalman update ────────────────────────────────────
                self._kf.predict()
                if detected_center is not None:
                    if not self._kf_initialized:
                        self._kf.x[0] = detected_center[0]
                        self._kf.x[1] = detected_center[1]
                        self._kf_initialized = True
                    else:
                        self._kf.update(detected_center)

                # ── Build Detection ──────────────────────────────────
                if self._kf_initialized:
                    tx = float(self._kf.x[0, 0])
                    ty = float(self._kf.x[1, 0])
                    lk = confidence if detected_center is not None else 0.5
                    det = Detection(
                        track_id=1,
                        x=tx, y=ty, w=0.0, h=0.0,
                        confidence=lk,
                        keypoints=[{"label": self.keypoint_name,
                                    "x": tx, "y": ty, "likelihood": lk}]
                    )
                    self._last_detections = [det]

            except Exception as e:
                logger.error(f"[YoloRealtime] Loop error: {e}", exc_info=True)
