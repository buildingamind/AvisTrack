"""
Workspace resolver – ties workspace.yaml + sources.yaml together.

Most tools that walk a chamber-type workspace need the same shape of
context: "which workspace, which chamber drive, which wave, where do
clips/ go, where do annotations/ go, where is camera_rois.json on the
drive". This module produces that bundle and is shared by

* ``tools/sample_clips.py``       – Step C
* ``tools/scan_legacy_wave.py``   – Step F
* ``tools/build_dataset.py``      – Step D (read-side)
* ``cli/run_batch.py``            – Step G

The resolver accepts already-loaded ``WorkspaceConfig`` / ``SourcesConfig``
objects (cheap to mock in tests) or paths to their yaml files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from avistrack.config.loader import load_sources, load_workspace
from avistrack.config.schema import (
    ChamberSource,
    SourcesConfig,
    WaveSource,
    WorkspaceConfig,
)


VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov"}


@dataclass
class ChamberWaveContext:
    """Fully-resolved paths and metadata for one (chamber_id, wave_id)."""
    workspace:    WorkspaceConfig
    sources:      SourcesConfig
    chamber:      ChamberSource
    wave:         WaveSource
    workspace_root: Path

    # ── Workspace-side paths (always available) ──
    @property
    def chamber_type(self) -> str:
        return self.workspace.chamber_type

    @property
    def workspace_chamber_dir(self) -> Path:
        return Path(self.workspace.workspace.root)

    @property
    def clips_root(self) -> Path:
        return Path(self.workspace.workspace.clips)

    @property
    def frames_root(self) -> Path:
        frames = self.workspace.workspace.frames
        if not frames:
            # Default: sibling of clips/ when an old workspace.yaml omits it.
            return self.workspace_chamber_dir / "frames"
        return Path(frames)

    @property
    def annotations_root(self) -> Path:
        return Path(self.workspace.workspace.annotations)

    @property
    def manifests_root(self) -> Path:
        return Path(self.workspace.workspace.manifests)

    @property
    def datasets_root(self) -> Path:
        return Path(self.workspace.workspace.dataset)

    @property
    def models_root(self) -> Path:
        return Path(self.workspace.workspace.models)

    @property
    def clip_dir(self) -> Path:
        """Where new clips for this (chamber, wave) should be written."""
        return self.clips_root / self.chamber.chamber_id / self.wave.wave_id

    @property
    def frame_dir(self) -> Path:
        """Per-(chamber, wave) parent of all `{clip_stem}/frame_*.png` dirs."""
        return self.frames_root / self.chamber.chamber_id / self.wave.wave_id

    @property
    def annotation_dir(self) -> Path:
        return self.annotations_root / self.chamber.chamber_id / self.wave.wave_id

    def frames_for_clip(self, clip_stem: str) -> Path:
        return self.frame_dir / clip_stem

    def annotations_for_clip(self, clip_stem: str) -> Path:
        return self.annotation_dir / clip_stem

    @property
    def all_clips_csv(self) -> Path:
        return self.manifests_root / "all_clips.csv"

    # ── Drive-side paths (require the chamber drive to be mounted) ──
    @property
    def drive_online(self) -> bool:
        return self.chamber.chamber_root is not None

    @property
    def chamber_root(self) -> Path:
        self._require_online("chamber_root")
        return Path(self.chamber.chamber_root)

    @property
    def wave_root(self) -> Path:
        """`{chamber_root}/{wave_subpath}` – top of this wave on the drive."""
        return self.chamber_root / self.wave.wave_subpath

    @property
    def metadata_dir(self) -> Path:
        """Where camera_rois / valid_ranges / time_calibration live."""
        if not self.wave.metadata_subpath:
            raise ValueError(
                f"wave {self.wave.wave_id!r} on chamber "
                f"{self.chamber.chamber_id!r} has no metadata_subpath"
            )
        return self.chamber_root / self.wave.metadata_subpath

    @property
    def roi_file(self) -> Path:
        return self.metadata_dir / "camera_rois.json"

    @property
    def valid_ranges_file(self) -> Path:
        return self.metadata_dir / "valid_ranges.json"

    @property
    def time_calibration_file(self) -> Path:
        return self.metadata_dir / "time_calibration.json"

    @property
    def ocr_roi_file(self) -> Path:
        return self.metadata_dir / "ocr_roi.json"

    # ── Video discovery ──
    def list_videos(self, modality: str = "rgb") -> list[Path]:
        """
        Enumerate source videos for this wave, filtered by modality keyword
        in the filename ("RGB" or "IR", case-insensitive).

        Structured layout: rglob under ``raw_videos_subpath``.
        Legacy layout:     glob ``raw_videos_glob`` under ``wave_root``.
        """
        self._require_online("list_videos")
        keyword = modality.upper()

        if self.wave.layout == "legacy":
            if not self.wave.raw_videos_glob:
                raise ValueError(
                    f"legacy wave {self.wave.wave_id!r} on chamber "
                    f"{self.chamber.chamber_id!r} missing raw_videos_glob"
                )
            base = self.wave_root
            paths = base.glob(self.wave.raw_videos_glob)
        else:
            if not self.wave.raw_videos_subpath:
                raise ValueError(
                    f"structured wave {self.wave.wave_id!r} on chamber "
                    f"{self.chamber.chamber_id!r} missing raw_videos_subpath"
                )
            base = self.chamber_root / self.wave.raw_videos_subpath
            paths = base.rglob("*")

        out = []
        for p in paths:
            if not p.is_file():
                continue
            if p.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            if keyword and keyword not in p.stem.upper():
                continue
            out.append(p)
        return sorted(out)

    # ── Helpers ──
    def _require_online(self, what: str) -> None:
        if not self.drive_online:
            raise DriveOfflineError(
                f"chamber {self.chamber.chamber_id!r} (uuid "
                f"{self.chamber.drive_uuid}) is not mounted; cannot resolve "
                f"{what}. Plug the drive in and retry."
            )


class DriveOfflineError(RuntimeError):
    """Raised when a workspace operation needs an offline chamber drive."""


# ── Loaders ─────────────────────────────────────────────────────────────

def load_context(
    workspace_yaml: str | Path,
    sources_yaml:   str | Path,
    chamber_id:     str,
    wave_id:        str,
    workspace_root: Optional[str | Path] = None,
    require_drive:  bool = True,
    probe:          bool = True,
) -> ChamberWaveContext:
    """
    Resolve one (chamber_id, wave_id) into a :class:`ChamberWaveContext`.

    Parameters
    ----------
    workspace_yaml, sources_yaml :
        Paths to the chamber-type's workspace.yaml and sources.yaml. By
        convention they live side-by-side at
        ``{workspace_root}/{chamber_type}/``.
    chamber_id, wave_id :
        Lookup keys – must already be registered in sources.yaml.
    workspace_root :
        Optional override. If omitted, derived as the grandparent of
        ``workspace_yaml``.
    require_drive :
        If True (default) and the chamber drive is offline, raise
        :class:`DriveOfflineError` so callers fail fast.
    probe :
        Forwarded to :func:`load_sources`. Set to False in tests where
        platform probing is not desired.
    """
    if workspace_root is None:
        workspace_root = Path(workspace_yaml).resolve().parent.parent

    workspace = load_workspace(workspace_yaml, workspace_root=workspace_root)
    sources   = load_sources(sources_yaml, workspace_root=workspace_root,
                             probe=probe)

    if workspace.chamber_type != sources.chamber_type:
        raise ValueError(
            f"chamber_type mismatch: workspace.yaml says "
            f"{workspace.chamber_type!r}, sources.yaml says "
            f"{sources.chamber_type!r}"
        )

    chamber = sources.get_chamber(chamber_id)
    wave    = chamber.get_wave(wave_id)

    ctx = ChamberWaveContext(
        workspace=workspace, sources=sources,
        chamber=chamber,     wave=wave,
        workspace_root=Path(workspace_root),
    )
    if require_drive and not ctx.drive_online:
        ctx._require_online("chamber drive")
    return ctx
