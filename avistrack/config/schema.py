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
    frames:      Optional[str] = None   # extracted PNG frames per clip
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


# ── Dataset recipe schema ────────────────────────────────────────────────

class RecipeInclude(BaseModel):
    chambers: list[str] = ["*"]
    waves:    list[str] = ["*"]
    layouts:  list[str] = ["*"]

    model_config = {"extra": "allow"}


class RecipeExclude(BaseModel):
    source_videos: list[str] = []
    clip_paths:    list[str] = []

    model_config = {"extra": "allow"}


class RecipeSplit(BaseModel):
    ratios:    dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}
    stratify:  str = "chamber"   # chamber | wave | clip | none
    seed:      int = 42

    model_config = {"extra": "allow"}

    @field_validator("stratify")
    @classmethod
    def _check_stratify(cls, v: str) -> str:
        if v not in {"chamber", "wave", "clip", "none"}:
            raise ValueError(
                f"split.stratify must be one of chamber|wave|clip|none, got {v!r}"
            )
        return v

    @field_validator("ratios")
    @classmethod
    def _check_ratios(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("split.ratios must not be empty")
        bad = [k for k in v if k not in {"train", "val", "test"}]
        if bad:
            raise ValueError(f"split.ratios keys must be train|val|test, got {bad!r}")
        if any(r < 0 for r in v.values()):
            raise ValueError(f"split.ratios values must be ≥ 0, got {v!r}")
        s = sum(v.values())
        if s <= 0:
            raise ValueError(f"split.ratios sum must be > 0, got {v!r}")
        return v


class RecipeConfig(BaseModel):
    """
    Dataset recipe – describes how to assemble one immutable
    ``datasets/{name}/`` view from a workspace.
    """
    name:         str
    chamber_type: str
    include:      RecipeInclude  = RecipeInclude()
    exclude:      RecipeExclude  = RecipeExclude()
    require_annotations: bool    = True
    split:        RecipeSplit    = RecipeSplit()
    classes:      list[str]      = ["chick"]

    model_config = {"extra": "allow"}

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not v or any(c.isspace() for c in v) or "/" in v or "\\" in v:
            raise ValueError(
                f"recipe name must be a non-empty path-safe identifier, got {v!r}"
            )
        return v


# ── Experiment config (workspace-aware training) ─────────────────────────

class TrainingDefaults(BaseModel):
    """Training kwargs applied to every run unless overridden."""
    epochs:   int = 300
    imgsz:    int = 640
    batch:    int = 16
    device:   Any = 0
    patience: int = 20
    workers:  int = 0
    exist_ok: bool = True
    verbose:  bool = False

    model_config = {"extra": "allow"}


class TrainingRun(BaseModel):
    """One training run inside an experiment.

    ``model`` is one of:
      * ultralytics registry name (``yolov8n.pt``, ``yolo11n.pt``)
      * workspace-relative path inside the current experiment
        (``phase1/yolo8n/weights/best.pt``) — resolved against
        ``{workspace}/models/{experiment_name}/``
      * absolute path
    """
    name:  str
    model: str

    model_config = {"extra": "allow"}   # transparent passthrough for lr0, etc.


class ExperimentConfig(BaseModel):
    """
    Workspace-aware experiment yaml. Lives at e.g.
    ``train/experiments/W2_collective_phase1_v2.yaml`` and is loaded by
    :func:`avistrack.config.loader.load_experiment`.

    The runner derives all output paths from these fields:

    * ``data_yaml``  → ``{workspace}/datasets/{dataset_name}/data.yaml``
    * ``exp_dir``    → ``{workspace}/models/{experiment_name}/``
    * ``phase_dir``  → ``{exp_dir}/phase{phase}/``
    """
    chamber_type:    str
    workspace_yaml:  str
    experiment_name: str
    dataset_name:    str
    phase:           int
    defaults:        TrainingDefaults = TrainingDefaults()
    runs:            list[TrainingRun]

    model_config = {"extra": "allow"}

    @field_validator("phase")
    @classmethod
    def _check_phase(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"phase must be ≥ 1, got {v!r}")
        return v

    @field_validator("experiment_name")
    @classmethod
    def _check_experiment_name(cls, v: str) -> str:
        if not v or any(c.isspace() for c in v) or "/" in v or "\\" in v:
            raise ValueError(
                f"experiment_name must be a non-empty path-safe identifier, got {v!r}"
            )
        return v

    @field_validator("runs")
    @classmethod
    def _check_runs_nonempty(cls, v: list[TrainingRun]) -> list[TrainingRun]:
        if not v:
            raise ValueError("experiment must define at least one run")
        names = [r.name for r in v]
        if len(set(names)) != len(names):
            dups = [n for n in names if names.count(n) > 1]
            raise ValueError(f"duplicate run names in experiment: {sorted(set(dups))!r}")
        return v
