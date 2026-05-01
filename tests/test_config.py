"""
tests/test_config.py – smoke tests for config loading.

Covers both the legacy ``drive.root`` config flow and the new multi-
chamber storage layout (``workspace.yaml`` + ``sources.yaml``).
"""

from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

import pytest

from avistrack.config import (
    SourcesConfig,
    WorkspaceConfig,
    load_config,
    load_sources,
    load_workspace,
)
from avistrack.config import drive_probe


# ── Legacy drive.root configs ────────────────────────────────────────────

MINIMAL_YAML = """
experiment: "test"
model:
  backend: "yolo"
  weights: "/fake/path/best.pt"
chamber:
  n_subjects: 9
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content).lstrip())
    return p


def test_load_minimal_config(tmp_path: Path):
    cfg = load_config(_write(tmp_path, "mini.yaml", MINIMAL_YAML))
    assert cfg.experiment == "test"
    assert cfg.model.backend == "yolo"
    assert cfg.chamber.n_subjects == 9


def test_root_placeholder_resolution(tmp_path: Path):
    yaml_content = """
        experiment: "test"
        drive:
          root: "/media/drive/Wave3"
          roi_file: "{root}/02_Global_Metadata/camera_rois.json"
        model:
          backend: "yolo"
          weights: "{root}/03_Model_Training/best.pt"
        chamber:
          n_subjects: 9
    """
    cfg = load_config(_write(tmp_path, "legacy.yaml", yaml_content))
    assert cfg.drive.roi_file == "/media/drive/Wave3/02_Global_Metadata/camera_rois.json"
    assert cfg.model.weights  == "/media/drive/Wave3/03_Model_Training/best.pt"


# ── Workspace YAML ───────────────────────────────────────────────────────

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
      n_subjects:  9
      fps:         30
      target_size: [640, 640]
"""


def _write_workspace_layout(workspace_root: Path, chamber_type: str = "collective") -> Path:
    """Set up {workspace_root}/{chamber_type}/workspace.yaml on disk."""
    chamber_dir = workspace_root / chamber_type
    chamber_dir.mkdir(parents=True, exist_ok=True)
    return _write(chamber_dir, "workspace.yaml", WORKSPACE_YAML)


def test_load_workspace_resolves_placeholders(tmp_path: Path):
    workspace_root = tmp_path / "wkspc"
    yaml_path = _write_workspace_layout(workspace_root)

    cfg = load_workspace(yaml_path)

    assert isinstance(cfg, WorkspaceConfig)
    assert cfg.chamber_type == "collective"
    # The yaml uses forward-slash templates ({workspace_root}/{chamber_type}),
    # so the resolved root preserves whatever separator the platform uses
    # for str(workspace_root) followed by literal "/collective". We
    # normalise to forward slashes for cross-platform comparison.
    expected_root = f"{workspace_root.as_posix()}/collective"
    assert Path(cfg.workspace.root).as_posix()        == expected_root
    # The {root} alias inside workspace.* refers to workspace.root and
    # must resolve through the same placeholder pass.
    assert Path(cfg.workspace.clips).as_posix()       == f"{expected_root}/clips"
    assert Path(cfg.workspace.annotations).as_posix() == f"{expected_root}/annotations"
    assert Path(cfg.workspace.models).as_posix()      == f"{expected_root}/models"
    assert cfg.chamber.n_subjects                     == 9


def test_load_workspace_respects_explicit_workspace_root(tmp_path: Path):
    yaml_path = _write_workspace_layout(tmp_path / "ignored")
    cfg = load_workspace(yaml_path, workspace_root=tmp_path / "override")
    expected = f"{(tmp_path / 'override').as_posix()}/collective"
    assert Path(cfg.workspace.root).as_posix() == expected


def test_workspace_inside_git_repo_is_rejected(tmp_path: Path):
    """workspace_root must not resolve inside any git repository."""
    fake_repo = tmp_path / "repo"
    (fake_repo / ".git").mkdir(parents=True)
    workspace_root = fake_repo / "wkspc"
    yaml_path = _write_workspace_layout(workspace_root)
    with pytest.raises(ValueError, match="inside a git repo"):
        load_workspace(yaml_path)


# ── Sources YAML ─────────────────────────────────────────────────────────

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


def test_load_sources_offline_drives(tmp_path: Path, monkeypatch):
    """When no chamber drive is mounted, chamber_root stays None and
    wave subpaths remain unresolved (they need {chamber_root})."""
    chamber_dir = tmp_path / "wkspc" / "collective"
    chamber_dir.mkdir(parents=True)
    yaml_path = _write(chamber_dir, "sources.yaml", SOURCES_YAML)

    monkeypatch.setattr(drive_probe, "probe_drive_mount", lambda uuid: None)
    cfg = load_sources(yaml_path)

    assert isinstance(cfg, SourcesConfig)
    assert cfg.chamber_type == "collective"
    chamber = cfg.get_chamber("collective_104A")
    assert chamber.chamber_root is None
    wave = chamber.get_wave("wave2")
    # Unresolved {chamber_root} still present – this is intentional, we
    # only resolve once the drive is online.
    assert "{chamber_root}" not in wave.wave_subpath
    assert wave.raw_videos_subpath == "Wave2/00_raw_videos"


def test_load_sources_online_drive_resolves_chamber_root(tmp_path: Path, monkeypatch):
    chamber_dir = tmp_path / "wkspc" / "collective"
    chamber_dir.mkdir(parents=True)
    yaml_path = _write(chamber_dir, "sources.yaml", SOURCES_YAML)

    fake_mount = tmp_path / "fake-mount"
    monkeypatch.setattr(
        "avistrack.config.loader.probe_drive_mount",
        lambda uuid: fake_mount if uuid == "ABCD-1234" else None,
    )
    cfg = load_sources(yaml_path)
    chamber = cfg.get_chamber("collective_104A")
    assert chamber.chamber_root == str(fake_mount)


def test_load_sources_unknown_chamber_raises(tmp_path: Path, monkeypatch):
    chamber_dir = tmp_path / "wkspc" / "collective"
    chamber_dir.mkdir(parents=True)
    yaml_path = _write(chamber_dir, "sources.yaml", SOURCES_YAML)

    monkeypatch.setattr("avistrack.config.loader.probe_drive_mount", lambda uuid: None)
    cfg = load_sources(yaml_path)
    with pytest.raises(KeyError):
        cfg.get_chamber("collective_999X")


def test_invalid_wave_layout_rejected(tmp_path: Path, monkeypatch):
    chamber_dir = tmp_path / "wkspc" / "collective"
    chamber_dir.mkdir(parents=True)
    bad = """
        chamber_type: collective
        chambers:
          - chamber_id:  collective_x
            drive_uuid:  "0000-0000"
            waves:
              - wave_id: w
                layout: something_else
                wave_subpath: "x"
    """
    yaml_path = _write(chamber_dir, "sources.yaml", bad)
    monkeypatch.setattr("avistrack.config.loader.probe_drive_mount", lambda uuid: None)
    with pytest.raises(Exception):
        load_sources(yaml_path)


# ── Drive UUID probe helpers ─────────────────────────────────────────────

def test_normalize_uuid_handles_case_and_whitespace():
    assert drive_probe.normalize_uuid("  abcd-1234 ") == "ABCD-1234"
    assert drive_probe.normalize_uuid("abcd-1234") == drive_probe.normalize_uuid("ABCD-1234")


def test_format_windows_serial():
    # Win32 reports the serial as a bare 8-char hex string; we present it
    # in the canonical XXXX-XXXX form used in sources.yaml.
    assert drive_probe._format_windows_serial("ABCD1234") == "ABCD-1234"
    assert drive_probe._format_windows_serial("abcd1234") == "ABCD-1234"
    assert drive_probe._format_windows_serial("ALREADY-DASHED") == "ALREADY-DASHED"
