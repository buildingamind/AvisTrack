"""
DeepLabCut backend.

Ported from ChamberBroadcaster/chamber_broadcaster/processors/dlc_tracking.py
Adapted to return standard Detection objects.
"""

import os
import tempfile
import shutil
import logging

import cv2
import pandas as pd

from avistrack.backends.base import TrackerBackend, Detection

logger = logging.getLogger(__name__)


class DLCTracker(TrackerBackend):
    """
    Single-frame DeepLabCut keypoint tracker.

    Requires deeplabcut to be installed in the environment:
        pip install deeplabcut

    Config keys expected under cfg.model:
        weights         : path to DLC project config.yaml
        inference_interval (optional, default 1): run inference every N frames
        temp_dir (optional): directory for temp image files
        cleanup_temp (optional, default True): delete temp files after each frame
    """

    def __init__(self, cfg):
        try:
            import deeplabcut as dlc   # lazy import – only fail if actually used
            self._dlc = dlc
        except ImportError:
            raise ImportError(
                "deeplabcut is not installed.  "
                "Run: pip install deeplabcut"
            )

        model_cfg = cfg.model
        self.model_config_path  = model_cfg.weights
        self.inference_interval = int(model_cfg.get("inference_interval", 1))
        self.cleanup_temp       = bool(model_cfg.get("cleanup_temp", True))
        self.analysis_params    = dict(model_cfg.get("analysis_params", {}))

        # Temp dir for writing single frames to disk (DLC needs a file path)
        temp_dir = model_cfg.get("temp_dir")
        if temp_dir:
            self._temp_dir = temp_dir
            os.makedirs(temp_dir, exist_ok=True)
            self._temp_dir_owned = False
        else:
            self._temp_dir = tempfile.mkdtemp(prefix="avistrack_dlc_")
            self._temp_dir_owned = True

        self._frame_counter   = 0
        self._last_detections: list[Detection] = []

        logger.info(f"[DLC] Initialized. Model config: {self.model_config_path}")

    # ── Public API ───────────────────────────────────────────────────────

    def update(self, frame) -> list[Detection]:
        self._frame_counter += 1

        if (self._frame_counter % self.inference_interval) != 0:
            return self._last_detections

        temp_img = os.path.join(self._temp_dir, "frame.png")
        cv2.imwrite(temp_img, frame)

        try:
            self._dlc.analyze_images(
                config    = self.model_config_path,
                images    = [temp_img],
                destfolder = self._temp_dir,
                **self.analysis_params,
            )
            self._last_detections = self._parse_output()
        except Exception as e:
            logger.error(f"[DLC] Inference failed: {e}", exc_info=True)
        finally:
            if self.cleanup_temp:
                self._clean_temp()

        return self._last_detections

    def release(self) -> None:
        if self._temp_dir_owned and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)

    # ── Internal ─────────────────────────────────────────────────────────

    def _parse_output(self) -> list[Detection]:
        """Parse the CSV DLC writes into Detection objects."""
        csv_files = [f for f in os.listdir(self._temp_dir) if f.endswith(".csv")]
        if not csv_files:
            logger.warning("[DLC] No CSV output found.")
            return []

        df = pd.read_csv(
            os.path.join(self._temp_dir, csv_files[0]),
            header=[0, 1, 2, 3],
            index_col=0,
        )
        scorer      = df.columns.get_level_values(0)[0]
        individuals = df.columns.get_level_values(1).unique()

        detections = []
        for idx, individual in enumerate(individuals):
            data       = df[scorer][individual]
            bodyparts  = data.columns.get_level_values(0).unique()
            keypoints  = []
            for bp in bodyparts:
                if bp in data:
                    bp_data = data[bp]
                    keypoints.append({
                        "label":      bp,
                        "x":          float(bp_data["x"].iloc[0]),
                        "y":          float(bp_data["y"].iloc[0]),
                        "likelihood": float(bp_data["likelihood"].iloc[0]),
                    })
            if keypoints:
                detections.append(Detection(
                    track_id   = idx + 1,
                    x=0.0, y=0.0, w=0.0, h=0.0,
                    keypoints  = keypoints,
                ))
        return detections

    def _clean_temp(self):
        for f in os.listdir(self._temp_dir):
            fp = os.path.join(self._temp_dir, f)
            if os.path.isfile(fp):
                try:
                    os.unlink(fp)
                except Exception:
                    pass
