"""
Config schema using Pydantic v2.

Validates YAML configs before they reach any backend code.
Extra fields are allowed so that users can add experiment-specific
metadata without breaking validation.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, field_validator


class ModelConfig(BaseModel):
    backend: str                        # "yolo" | "dlc" | "vit"
    weights: Optional[Any] = None       # str for YOLO/DLC, dict for ViT
    mode: str = "offline"               # "offline" | "realtime"  (YOLO only)

    model_config = {"extra": "allow"}   # allow backend-specific extra fields

    def get(self, key: str, default=None):
        """Dict-style get for convenience (backend code uses this pattern)."""
        return getattr(self, key, None) or self.__pydantic_extra__.get(key, default)


class ChamberConfig(BaseModel):
    n_subjects: int
    roi_file:   Optional[str] = None
    target_size: Optional[list[int]] = None   # [width, height]
    fps:         Optional[float]     = None

    model_config = {"extra": "allow"}

    def get(self, key: str, default=None):
        return getattr(self, key, None) or self.__pydantic_extra__.get(key, default)


class DriveConfig(BaseModel):
    """
    Paths to data on the external drive / NAS.
    All paths are strings; they are NOT validated to exist at config-load time
    because the drive may not always be mounted when editing configs.
    """
    root:              Optional[str] = None
    raw_videos:        Optional[str] = None
    dataset:           Optional[str] = None
    metadata:          Optional[str] = None
    roi_file:          Optional[str] = None
    valid_ranges:      Optional[str] = None     # path to valid_ranges.json
    ocr_roi:           Optional[str] = None     # path to ocr_roi.json
    time_calibration:  Optional[str] = None     # path to time_calibration.json
    train_manifest:    Optional[str] = None
    val_manifest:      Optional[str] = None
    test_manifest:     Optional[str] = None

    model_config = {"extra": "allow"}


class TimeConfig(BaseModel):
    """
    Settings for burn-in timestamp OCR and time calibration.
    """
    timezone:    str = "America/New_York"   # IANA tz name (handles DST)
    time_format: str = "auto"               # "auto" to detect, or strptime format

    model_config = {"extra": "allow"}


class AvisTrackConfig(BaseModel):
    experiment: str = "unnamed"
    model:      ModelConfig
    chamber:    ChamberConfig
    drive:      DriveConfig = DriveConfig()
    time:       TimeConfig  = TimeConfig()

    model_config = {"extra": "allow"}

    def get(self, key: str, default=None):
        return self.__pydantic_extra__.get(key, default)
