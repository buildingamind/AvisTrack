"""
Config schema using Pydantic v2.

Validates YAML configs before they reach any backend code.
Extra fields are allowed so that users can add experiment-specific
metadata without breaking validation.

Three top-level config kinds coexist:

* ``AvisTrackConfig`` – legacy single-file experiment config keyed off
  ``drive.root``. Preserved for backward compatibility during the
  multi-chamber storage migration.
* ``WorkspaceConfig`` – describes one chamber type's central workspace
  (clips/, annotations/, manifests/, datasets/, models/).
* ``SourcesConfig`` – registers the chamber drives that feed a workspace,
  each with a stable ``drive_uuid`` so the loader can re-bind to the
  current mount point when a drive is plugged in.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, field_validator


# ── Legacy (drive.root) schema ────────────────────────────────────────────

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


# ── Workspace / sources schema (multi-chamber storage) ───────────────────

class WorkspaceLayout(BaseModel):
    """Resolved paths inside a single chamber-type workspace."""
    root:        str
    clips:       Optional[str] = None
    annotations: Optional[str] = None
    manifests:   Optional[str] = None
    dataset:     Optional[str] = None   # dataset views built from clips+annotations
    models:      Optional[str] = None

    model_config = {"extra": "allow"}


class WorkspaceConfig(BaseModel):
    """
    One chamber type's central workspace. Lives at
    ``{workspace_root}/{chamber_type}/workspace.yaml``.
    """
    chamber_type: str
    workspace:    WorkspaceLayout
    chamber:      ChamberConfig
    time:         TimeConfig = TimeConfig()

    model_config = {"extra": "allow"}


class WaveSource(BaseModel):
    """One wave of recordings on a chamber drive."""
    wave_id: str
    layout:  str = "structured"   # "structured" | "legacy"
    wave_subpath: str

    # Structured-layout fields. Resolved against {chamber_root}/{wave_subpath}.
    raw_videos_subpath: Optional[str] = None
    metadata_subpath:   Optional[str] = None

    # Legacy fields. raw_videos_glob is rooted at {chamber_root}/{wave_subpath}.
    raw_videos_glob:    Optional[str] = None

    model_config = {"extra": "allow"}

    @field_validator("layout")
    @classmethod
    def _check_layout(cls, v: str) -> str:
        if v not in {"structured", "legacy"}:
            raise ValueError(f"layout must be 'structured' or 'legacy', got {v!r}")
        return v


class ChamberSource(BaseModel):
    """A physical chamber drive registered to a workspace."""
    chamber_id:   str
    drive_uuid:   str
    drive_label:  Optional[str] = None
    waves:        list[WaveSource] = []

    # Populated at load time when the drive is mounted; otherwise None.
    chamber_root: Optional[str] = None

    model_config = {"extra": "allow"}

    def get_wave(self, wave_id: str) -> WaveSource:
        for w in self.waves:
            if w.wave_id == wave_id:
                return w
        raise KeyError(f"wave_id {wave_id!r} not registered for chamber {self.chamber_id!r}")


class SourcesConfig(BaseModel):
    """
    Registers all chamber drives that feed a workspace. Lives at
    ``{workspace_root}/{chamber_type}/sources.yaml``.
    """
    chamber_type: str
    chambers:     list[ChamberSource] = []

    model_config = {"extra": "allow"}

    def get_chamber(self, chamber_id: str) -> ChamberSource:
        for c in self.chambers:
            if c.chamber_id == chamber_id:
                return c
        raise KeyError(f"chamber_id {chamber_id!r} not registered in sources.yaml")
