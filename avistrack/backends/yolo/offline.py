"""
YOLO offline multi-subject tracker with IoU matching and linear interpolation.

Ported from yolo-tracking-lab/W3_COLL/src/run_video_pipeline.py
Adapted to return standard Detection objects per frame.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from avistrack.backends.base import TrackerBackend, Detection

logger = logging.getLogger(__name__)


def _iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """Vectorised IoU between two sets of boxes in [x1,y1,x2,y2] format."""
    bb_gt   = np.expand_dims(bb_gt,   0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w   = np.maximum(0., xx2 - xx1)
    h   = np.maximum(0., yy2 - yy1)
    inter = w * h
    area_test = (bb_test[..., 2]-bb_test[..., 0]) * (bb_test[..., 3]-bb_test[..., 1])
    area_gt   = (bb_gt[..., 2]-bb_gt[..., 0])   * (bb_gt[..., 3]-bb_gt[..., 1])
    return inter / (area_test + area_gt - inter + 1e-8)


def _interpolate_track(track_df: pd.DataFrame, max_gap: int = 30) -> pd.DataFrame:
    """
    Fill gaps in a single track's trajectory using linear interpolation.
    Gaps larger than max_gap frames are left as NaN and then dropped.
    """
    track_df = (track_df
                .sort_values("frame")
                .drop_duplicates(subset=["frame"], keep="first"))
    min_f, max_f = track_df["frame"].min(), track_df["frame"].max()
    full_idx = np.arange(min_f, max_f + 1)
    track_df = track_df.set_index("frame").reindex(full_idx)
    track_df[["x", "y", "w", "h"]] = (
        track_df[["x", "y", "w", "h"]]
        .interpolate(method="linear", limit=max_gap)
    )
    track_df["tid"]   = track_df["tid"].ffill().bfill()
    track_df["conf"]  = track_df["conf"].fillna(1.0)
    track_df["class"] = track_df["class"].fillna(1)
    track_df["vis"]   = track_df["vis"].fillna(-1)
    return (track_df
            .dropna(subset=["x"])
            .reset_index()
            .rename(columns={"index": "frame"}))


class YoloOfflineTracker(TrackerBackend):
    """
    Frame-by-frame offline tracker for multi-subject videos.

    Typical usage (see cli/run_batch.py for full pipeline):

        tracker = YoloOfflineTracker(cfg)
        rows = []
        for frame_idx, frame in enumerate(video_frames, start=1):
            detections = tracker.update(frame)
            for d in detections:
                rows.append((frame_idx, d.track_id, d.x, d.y, d.w, d.h,
                             d.confidence, 1, 1.0))
        tracker.flush_interpolation(rows, output_path)
    """

    def __init__(self, cfg):
        self.cfg  = cfg
        model_cfg = cfg.model
        track_cfg = cfg.get("tracking", {})

        self.model_path    = model_cfg.weights
        self.conf_thresh   = float(track_cfg.get("conf_threshold", 0.2))
        self.n_subjects    = int(cfg.chamber.n_subjects)
        self.max_gap       = int(track_cfg.get("max_gap_frames", 30))
        self.batch_size    = int(track_cfg.get("batch_size", 32))

        # target inference size: use chamber target_size if set, else 640×640
        ts = cfg.chamber.get("target_size", [640, 640])
        self.imgsz = ts[0]   # YOLO expects a single int (square)

        logger.info(f"[YoloOffline] Loading model: {self.model_path}")
        self._model = YOLO(self.model_path)

        self._prev_boxes: list  = []
        self._prev_ids:   list  = []

    # ── Public API ───────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> list[Detection]:
        """
        Run detection + IoU-based ID matching for one frame.
        Returns detections with stable track IDs.
        """
        return self.update_batch([frame])[0]

    def update_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """
        Run YOLO inference on a batch of frames in one GPU call,
        then apply sequential IoU matching per frame.

        Parameters
        ----------
        frames : list of H×W×C BGR numpy arrays (same shape expected)

        Returns
        -------
        list of detection lists, one per input frame
        """
        batch_results = self._model.predict(
            frames,
            conf=self.conf_thresh,
            verbose=False,
            imgsz=self.imgsz,
        )

        all_detections = []
        for results in batch_results:
            raw_boxes = results.boxes.data.cpu().numpy()

            if len(raw_boxes) > 0:
                top = raw_boxes[np.argsort(raw_boxes[:, 4])[::-1][:self.n_subjects]]
                curr_xyxy = top[:, :4]
                curr_conf = top[:, 4]
            else:
                curr_xyxy = np.empty((0, 4))
                curr_conf = np.empty((0,))

            curr_ids = self._assign_ids(curr_xyxy)
            self._prev_boxes = curr_xyxy.tolist()
            self._prev_ids   = curr_ids

            detections = []
            for i, box in enumerate(curr_xyxy):
                x1, y1, x2, y2 = box
                detections.append(Detection(
                    track_id   = curr_ids[i],
                    x          = float(x1),
                    y          = float(y1),
                    w          = float(x2 - x1),
                    h          = float(y2 - y1),
                    confidence = float(curr_conf[i]),
                ))
            all_detections.append(detections)

        return all_detections

    def flush_interpolation(
        self,
        rows: list[tuple],
    ) -> pd.DataFrame:
        """
        After processing all frames, apply per-track linear interpolation
        and return the result as a DataFrame.

        Parameters
        ----------
        rows : list of (frame, tid, x, y, w, h, conf, cls, vis) tuples

        Returns
        -------
        pd.DataFrame with columns: frame, id, x, y, w, h, conf, class
            Dtypes: frame=int32, id=int16, x/y/w/h/conf=float32, class=int8
        """
        if not rows:
            logger.warning("[YoloOffline] No rows to interpolate.")
            return pd.DataFrame(
                columns=["frame", "id", "x", "y", "w", "h", "conf", "class"]
            )

        df = pd.DataFrame(
            rows,
            columns=["frame", "tid", "x", "y", "w", "h", "conf", "class", "vis"]
        )

        interpolated = []
        for tid, group in df.groupby("tid"):
            interpolated.append(_interpolate_track(group, max_gap=self.max_gap))

        if not interpolated:
            logger.warning("[YoloOffline] No tracks to interpolate.")
            return pd.DataFrame(
                columns=["frame", "id", "x", "y", "w", "h", "conf", "class"]
            )

        final = (pd.concat(interpolated)
                   .sort_values(["frame", "tid"])
                   .rename(columns={"tid": "id"})
                   [["frame", "id", "x", "y", "w", "h", "conf", "class"]])

        final = final.astype({
            "frame": "int32",
            "id":    "int16",
            "x":     "float32",
            "y":     "float32",
            "w":     "float32",
            "h":     "float32",
            "conf":  "float32",
            "class": "int8",
        })

        logger.info(f"[YoloOffline] Interpolation complete: {len(final)} rows, "
                    f"{final['id'].nunique()} tracks")
        return final

    def release(self) -> None:
        pass   # YOLO model is stateless between videos

    # ── Internal ─────────────────────────────────────────────────────────

    def _assign_ids(self, curr_xyxy: np.ndarray) -> list[int]:
        """Hungarian-algorithm IoU matching against previous frame boxes."""
        curr_ids = [-1] * len(curr_xyxy)

        if len(curr_xyxy) == 0:
            return curr_ids

        if not self._prev_boxes:
            return list(range(1, len(curr_xyxy) + 1))

        cost = 1.0 - _iou_batch(curr_xyxy, np.array(self._prev_boxes))
        r_ind, c_ind = linear_sum_assignment(cost)

        used_ids: set[int] = set()
        for r, c in zip(r_ind, c_ind):
            curr_ids[r] = self._prev_ids[c]
            used_ids.add(self._prev_ids[c])

        avail_ids   = sorted(set(range(1, self.n_subjects + 1)) - used_ids)
        existing_max = max(self._prev_ids, default=0)

        for i in range(len(curr_ids)):
            if curr_ids[i] == -1:
                if avail_ids:
                    curr_ids[i] = avail_ids.pop(0)
                else:
                    existing_max += 1
                    curr_ids[i]   = existing_max

        return curr_ids
