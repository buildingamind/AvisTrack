"""
tests/test_workspace.py – ChamberWaveContext + load_context resolver.

These tests do not need real video files; structured/legacy discovery is
validated by dropping placeholder .mkv / .mp4 files into a fake chamber
drive directory and checking what the resolver returns.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from avistrack.config.loader import load_workspace, load_sources  # noqa: F401
from avistrack.workspace import (
    ChamberWaveContext,
    DriveOfflineError,
    load_context,
)

WORKSPACE_YAML = """
chamber_type: collective
workspace:
  root:        "{workspace_root}/{chamber_type}"
  clips:       "{root}/clips"
  annotations: "{root}/annotations"
  manifests:   "{root}/manifests"
  dataset:     "{root}/datasets"
  models:      "{root}/models"
chamber:
  n_subjects: 9
  fps: 30
  target_size: [640, 640]
"""

SOURCES_YAML = """
chamber_type: collective
chambers:
  - chamber_id:  collective_104A
    drive_uuid:  "ABCD-1234"
    drive_label: "104-A"
    waves:
      - wave_id: wave2
        layout: structured
        wave_subpath: "Wave2"
        raw_videos_subpath: "{wave_subpath}/00_raw_videos"
        metadata_subpath:   "{wave_subpath}/02_Chamber_Metadata"
      - wave_id: wave1_legacy
        layout: legacy
        wave_subpath: "Wave1_OldDump"
        raw_videos_glob: "**/*.mp4"
        metadata_subpath: "_avistrack_added/wave1_legacy"
"""


def _bootstrap(tmp_path: Path):
    """Write workspace.yaml + sources.yaml on disk."""
    workspace_root = tmp_path / "wkspc"
    chamber_dir = workspace_root / "collective"
    chamber_dir.mkdir(parents=True)
    workspace_yaml = chamber_dir / "workspace.yaml"
    sources_yaml   = chamber_dir / "sources.yaml"
    workspace_yaml.write_text(textwrap.dedent(WORKSPACE_YAML).lstrip())
    sources_yaml.write_text(textwrap.dedent(SOURCES_YAML).lstrip())
    return workspace_root, workspace_yaml, sources_yaml


def test_load_context_offline_drive_raises(tmp_path: Path, monkeypatch):
    workspace_root, w, s = _bootstrap(tmp_path)
    monkeypatch.setattr("avistrack.config.loader.probe_drive_mount",
                        lambda uuid: None)
    with pytest.raises(DriveOfflineError, match="not mounted"):
        load_context(w, s, "collective_104A", "wave2",
                     workspace_root=workspace_root, require_drive=True)


def test_load_context_offline_drive_allowed(tmp_path: Path, monkeypatch):
    """With require_drive=False, the context is returned and drive_online=False."""
    workspace_root, w, s = _bootstrap(tmp_path)
    monkeypatch.setattr("avistrack.config.loader.probe_drive_mount",
                        lambda uuid: None)
    ctx = load_context(w, s, "collective_104A", "wave2",
                       workspace_root=workspace_root, require_drive=False)
    assert ctx.drive_online is False
    assert ctx.chamber.chamber_id == "collective_104A"
    # workspace-side paths still work even when offline.
    assert ctx.clip_dir == ctx.clips_root / "collective_104A" / "wave2"
    assert ctx.all_clips_csv.name == "all_clips.csv"


def test_load_context_online_resolves_metadata(tmp_path: Path, monkeypatch):
    workspace_root, w, s = _bootstrap(tmp_path)
    fake_mount = tmp_path / "drive"
    (fake_mount / "Wave2" / "00_raw_videos").mkdir(parents=True)
    (fake_mount / "Wave2" / "02_Chamber_Metadata").mkdir(parents=True)

    monkeypatch.setattr(
        "avistrack.config.loader.probe_drive_mount",
        lambda uuid: fake_mount if uuid == "ABCD-1234" else None,
    )
    ctx = load_context(w, s, "collective_104A", "wave2",
                       workspace_root=workspace_root, require_drive=True)

    assert ctx.drive_online is True
    assert ctx.chamber_root == fake_mount
    assert ctx.metadata_dir == fake_mount / "Wave2" / "02_Chamber_Metadata"
    assert ctx.roi_file.name == "camera_rois.json"
    assert ctx.time_calibration_file.name == "time_calibration.json"


def test_list_videos_structured(tmp_path: Path, monkeypatch):
    workspace_root, w, s = _bootstrap(tmp_path)
    fake_mount = tmp_path / "drive"
    raw_dir = fake_mount / "Wave2" / "00_raw_videos"
    raw_dir.mkdir(parents=True)
    # Two RGB, one IR, one non-video.
    (raw_dir / "Day1_Cam1_RGB.mkv").touch()
    (raw_dir / "Day2_Cam1_RGB.mkv").touch()
    (raw_dir / "Day3_Cam1_IR.mkv").touch()
    (raw_dir / "notes.txt").touch()

    monkeypatch.setattr(
        "avistrack.config.loader.probe_drive_mount",
        lambda uuid: fake_mount if uuid == "ABCD-1234" else None,
    )
    ctx = load_context(w, s, "collective_104A", "wave2",
                       workspace_root=workspace_root, require_drive=True)

    rgb = [v.name for v in ctx.list_videos(modality="rgb")]
    assert rgb == ["Day1_Cam1_RGB.mkv", "Day2_Cam1_RGB.mkv"]
    ir = [v.name for v in ctx.list_videos(modality="ir")]
    assert ir == ["Day3_Cam1_IR.mkv"]


def test_list_videos_legacy_glob(tmp_path: Path, monkeypatch):
    workspace_root, w, s = _bootstrap(tmp_path)
    fake_mount = tmp_path / "drive"
    legacy_dir = fake_mount / "Wave1_OldDump"
    (legacy_dir / "subdir").mkdir(parents=True)
    (legacy_dir / "ChamberA_RGB.mp4").touch()
    (legacy_dir / "subdir" / "deep_RGB.mp4").touch()
    (legacy_dir / "ignore.txt").touch()

    monkeypatch.setattr(
        "avistrack.config.loader.probe_drive_mount",
        lambda uuid: fake_mount if uuid == "ABCD-1234" else None,
    )
    ctx = load_context(w, s, "collective_104A", "wave1_legacy",
                       workspace_root=workspace_root, require_drive=True)

    names = sorted(v.name for v in ctx.list_videos(modality="rgb"))
    assert names == ["ChamberA_RGB.mp4", "deep_RGB.mp4"]


def test_list_videos_offline_raises(tmp_path: Path, monkeypatch):
    workspace_root, w, s = _bootstrap(tmp_path)
    monkeypatch.setattr("avistrack.config.loader.probe_drive_mount",
                        lambda uuid: None)
    ctx = load_context(w, s, "collective_104A", "wave2",
                       workspace_root=workspace_root, require_drive=False)
    with pytest.raises(DriveOfflineError):
        ctx.list_videos()


def test_load_context_unknown_chamber(tmp_path: Path, monkeypatch):
    workspace_root, w, s = _bootstrap(tmp_path)
    monkeypatch.setattr("avistrack.config.loader.probe_drive_mount",
                        lambda uuid: None)
    with pytest.raises(KeyError):
        load_context(w, s, "collective_999X", "wave2",
                     workspace_root=workspace_root, require_drive=False)


def test_load_context_unknown_wave(tmp_path: Path, monkeypatch):
    workspace_root, w, s = _bootstrap(tmp_path)
    monkeypatch.setattr("avistrack.config.loader.probe_drive_mount",
                        lambda uuid: None)
    with pytest.raises(KeyError):
        load_context(w, s, "collective_104A", "wave_404",
                     workspace_root=workspace_root, require_drive=False)


def test_chamber_type_mismatch_rejected(tmp_path: Path, monkeypatch):
    """workspace.yaml and sources.yaml must agree on chamber_type."""
    workspace_root, w, s = _bootstrap(tmp_path)
    # Clobber sources.yaml with a different chamber_type.
    raw = yaml.safe_load(s.read_text())
    raw["chamber_type"] = "vr"
    s.write_text(yaml.safe_dump(raw))

    monkeypatch.setattr("avistrack.config.loader.probe_drive_mount",
                        lambda uuid: None)
    with pytest.raises(ValueError, match="chamber_type mismatch"):
        load_context(w, s, "collective_104A", "wave2",
                     workspace_root=workspace_root, require_drive=False)
