"""
tests/test_step_f.py – legacy-wave onboarding tools (improve-plan §F).

Covers:

* tools/scan_legacy_wave.py – payload shape, idempotent re-scan, layout
  routing, "no videos found" error.
* tools/pick_rois.py        – workspace-mode path resolution
  (legacy + structured), arg validation.
* tools/edit_valid_ranges.py – workspace-mode path resolution
  (legacy + structured), arg validation.

cv2 is not required: ``scan_legacy_wave`` is invoked with ``probe=False``,
and ``pick_rois`` is exercised through its top-level path resolver,
which never touches OpenCV.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import textwrap
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Stub cv2 only if it's truly missing — keeps tests runnable in CI envs
# without opencv-python. Mirrors the pattern in test_sample_clips.py.
try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cv2"] = types.ModuleType("cv2")


# ── Loaders for tools/*.py (kept out of the package import path) ─────────

def _import_tool(name: str):
    path = REPO_ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tools_{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scan_legacy_wave = _import_tool("scan_legacy_wave")
pick_rois        = _import_tool("pick_rois")
edit_valid_ranges = _import_tool("edit_valid_ranges")


# ── Workspace fixtures ───────────────────────────────────────────────────

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
time:
  timezone:    "America/Chicago"
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
    """Write workspace.yaml + sources.yaml; create a fake mounted drive."""
    workspace_root = tmp_path / "wkspc"
    chamber_dir = workspace_root / "collective"
    chamber_dir.mkdir(parents=True)
    workspace_yaml = chamber_dir / "workspace.yaml"
    sources_yaml   = chamber_dir / "sources.yaml"
    workspace_yaml.write_text(textwrap.dedent(WORKSPACE_YAML).lstrip())
    sources_yaml.write_text(textwrap.dedent(SOURCES_YAML).lstrip())

    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    return workspace_root, workspace_yaml, sources_yaml, drive_root


def _patch_probe(monkeypatch, drive_root: Path) -> None:
    monkeypatch.setattr(
        "avistrack.config.loader.probe_drive_mount",
        lambda uuid: drive_root if uuid == "ABCD-1234" else None,
    )


# ══════════════════════════════════════════════════════════════════════════
# scan_legacy_wave
# ══════════════════════════════════════════════════════════════════════════

def test_classify_modality():
    assert scan_legacy_wave._classify_modality("Day1_Cam1_RGB") == "rgb"
    assert scan_legacy_wave._classify_modality("Cam2_IR_thermal") == "ir"
    assert scan_legacy_wave._classify_modality("plain_clip") == "unknown"


def test_build_payload_legacy_layout(tmp_path: Path, monkeypatch):
    workspace_root, w, s, drive = _bootstrap(tmp_path)
    legacy_root = drive / "Wave1_OldDump"
    (legacy_root / "subdir").mkdir(parents=True)
    v1 = legacy_root / "ChamberA_RGB.mp4"; v1.write_bytes(b"x" * 100)
    v2 = legacy_root / "subdir" / "deep_RGB.mp4"; v2.write_bytes(b"x" * 50)

    _patch_probe(monkeypatch, drive)
    from avistrack.workspace import load_context
    ctx = load_context(w, s, "collective_104A", "wave1_legacy",
                       workspace_root=workspace_root, require_drive=True)

    payload = scan_legacy_wave.build_video_index_payload(
        ctx, ctx.list_videos(modality="rgb"), probe=False)

    assert payload["chamber_id"] == "collective_104A"
    assert payload["wave_id"]    == "wave1_legacy"
    assert payload["layout"]     == "legacy"
    assert payload["drive_uuid"] == "ABCD-1234"
    assert payload["video_count"] == 2

    rels = sorted(v["rel_path"] for v in payload["videos"])
    assert rels == ["ChamberA_RGB.mp4", "subdir/deep_RGB.mp4"]
    sizes = {v["filename"]: v["size_bytes"] for v in payload["videos"]}
    assert sizes == {"ChamberA_RGB.mp4": 100, "deep_RGB.mp4": 50}
    # probe=False → cv2 fields are absent (not just None) in the entry.
    for v in payload["videos"]:
        assert "fps" not in v
        assert v["modality"] == "rgb"


def test_scan_writes_video_index_to_metadata_dir(tmp_path: Path, monkeypatch):
    workspace_root, w, s, drive = _bootstrap(tmp_path)
    legacy_root = drive / "Wave1_OldDump"
    legacy_root.mkdir(parents=True)
    (legacy_root / "first_RGB.mp4").write_bytes(b"x")

    _patch_probe(monkeypatch, drive)
    from avistrack.workspace import load_context
    ctx = load_context(w, s, "collective_104A", "wave1_legacy",
                       workspace_root=workspace_root, require_drive=True)

    target, payload = scan_legacy_wave.scan_legacy_wave(ctx, probe=False)
    assert target == drive / "_avistrack_added" / "wave1_legacy" / "video_index.json"
    assert target.exists()
    on_disk = json.loads(target.read_text())
    assert on_disk == payload
    assert on_disk["video_count"] == 1


def test_scan_idempotent_re_scan_overwrites(tmp_path: Path, monkeypatch):
    workspace_root, w, s, drive = _bootstrap(tmp_path)
    legacy_root = drive / "Wave1_OldDump"
    legacy_root.mkdir(parents=True)
    (legacy_root / "v1_RGB.mp4").write_bytes(b"x")

    _patch_probe(monkeypatch, drive)
    from avistrack.workspace import load_context
    ctx = load_context(w, s, "collective_104A", "wave1_legacy",
                       workspace_root=workspace_root, require_drive=True)
    target1, p1 = scan_legacy_wave.scan_legacy_wave(ctx, probe=False)
    assert p1["video_count"] == 1

    # New video appears, old one disappears – next scan should reflect that.
    (legacy_root / "v1_RGB.mp4").unlink()
    (legacy_root / "v2_RGB.mp4").write_bytes(b"y")
    target2, p2 = scan_legacy_wave.scan_legacy_wave(ctx, probe=False)
    assert target2 == target1
    assert p2["video_count"] == 1
    assert p2["videos"][0]["filename"] == "v2_RGB.mp4"


def test_scan_no_videos_raises(tmp_path: Path, monkeypatch):
    workspace_root, w, s, drive = _bootstrap(tmp_path)
    (drive / "Wave1_OldDump").mkdir(parents=True)

    _patch_probe(monkeypatch, drive)
    from avistrack.workspace import load_context
    ctx = load_context(w, s, "collective_104A", "wave1_legacy",
                       workspace_root=workspace_root, require_drive=True)
    with pytest.raises(SystemExit, match="No videos found"):
        scan_legacy_wave.scan_legacy_wave(ctx, probe=False)


def test_scan_structured_layout(tmp_path: Path, monkeypatch):
    """
    The tool's name says "legacy" but a structured wave with missing
    metadata should still scan cleanly.
    """
    workspace_root, w, s, drive = _bootstrap(tmp_path)
    raw = drive / "Wave2" / "00_raw_videos"
    raw.mkdir(parents=True)
    (raw / "Day1_Cam1_RGB.mkv").write_bytes(b"x")

    _patch_probe(monkeypatch, drive)
    from avistrack.workspace import load_context
    ctx = load_context(w, s, "collective_104A", "wave2",
                       workspace_root=workspace_root, require_drive=True)
    target, payload = scan_legacy_wave.scan_legacy_wave(ctx, probe=False)

    assert payload["layout"] == "structured"
    assert payload["video_count"] == 1
    # Structured waves write into 02_Chamber_Metadata, NOT _avistrack_added.
    assert target == drive / "Wave2" / "02_Chamber_Metadata" / "video_index.json"


# ══════════════════════════════════════════════════════════════════════════
# pick_rois._resolve_paths
# ══════════════════════════════════════════════════════════════════════════

def _ns(**kw) -> argparse.Namespace:
    """Build a Namespace with workspace-mode + legacy fields all set to None."""
    defaults = dict(
        config=None, video_dir=None, roi_file=None,
        workspace_yaml=None, sources_yaml=None,
        chamber_id=None, wave_id=None, modality="rgb",
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_pick_rois_workspace_mode_legacy_wave(tmp_path: Path, monkeypatch):
    workspace_root, w, s, drive = _bootstrap(tmp_path)
    (drive / "Wave1_OldDump").mkdir(parents=True)
    _patch_probe(monkeypatch, drive)

    args = _ns(workspace_yaml=str(w), chamber_id="collective_104A",
               wave_id="wave1_legacy")
    video_dir, roi_file = pick_rois._resolve_paths(args)

    assert video_dir == drive / "Wave1_OldDump"
    # Legacy waves → camera_rois.json under _avistrack_added/{wave_id}/
    assert roi_file == drive / "_avistrack_added" / "wave1_legacy" / "camera_rois.json"


def test_pick_rois_workspace_mode_structured_wave(tmp_path: Path, monkeypatch):
    workspace_root, w, s, drive = _bootstrap(tmp_path)
    (drive / "Wave2" / "02_Chamber_Metadata").mkdir(parents=True)
    _patch_probe(monkeypatch, drive)

    args = _ns(workspace_yaml=str(w), chamber_id="collective_104A",
               wave_id="wave2")
    video_dir, roi_file = pick_rois._resolve_paths(args)

    assert video_dir == drive / "Wave2"
    assert roi_file == drive / "Wave2" / "02_Chamber_Metadata" / "camera_rois.json"


def test_pick_rois_workspace_mode_partial_args_rejected(tmp_path: Path, monkeypatch):
    _, w, _, _ = _bootstrap(tmp_path)
    args = _ns(workspace_yaml=str(w))  # missing --chamber-id / --wave-id
    with pytest.raises(SystemExit):
        pick_rois._resolve_paths(args)


def test_pick_rois_config_plus_workspace_rejected(tmp_path: Path, monkeypatch):
    _, w, _, _ = _bootstrap(tmp_path)
    args = _ns(config="legacy.yaml", workspace_yaml=str(w),
               chamber_id="x", wave_id="y")
    with pytest.raises(SystemExit):
        pick_rois._resolve_paths(args)


def test_pick_rois_workspace_overrides_respected(tmp_path: Path, monkeypatch):
    """Explicit --roi-file / --video-dir win over workspace defaults."""
    workspace_root, w, s, drive = _bootstrap(tmp_path)
    (drive / "Wave2" / "02_Chamber_Metadata").mkdir(parents=True)
    _patch_probe(monkeypatch, drive)

    custom_roi = tmp_path / "custom_rois.json"
    custom_dir = tmp_path / "videos"
    args = _ns(workspace_yaml=str(w), chamber_id="collective_104A",
               wave_id="wave2", roi_file=str(custom_roi),
               video_dir=str(custom_dir))
    video_dir, roi_file = pick_rois._resolve_paths(args)
    assert video_dir == custom_dir
    assert roi_file == custom_roi


# ══════════════════════════════════════════════════════════════════════════
# edit_valid_ranges._resolve_paths
# ══════════════════════════════════════════════════════════════════════════

def _ns_evr(**kw) -> argparse.Namespace:
    defaults = dict(
        config=None, workspace_yaml=None, sources_yaml=None,
        chamber_id=None, wave_id=None,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_evr_workspace_mode_legacy_wave(tmp_path: Path, monkeypatch):
    workspace_root, w, s, drive = _bootstrap(tmp_path)
    (drive / "Wave1_OldDump").mkdir(parents=True)
    _patch_probe(monkeypatch, drive)

    args = _ns_evr(workspace_yaml=str(w), chamber_id="collective_104A",
                   wave_id="wave1_legacy")
    cal, vr, tz = edit_valid_ranges._resolve_paths(args)

    base = drive / "_avistrack_added" / "wave1_legacy"
    assert cal == base / "time_calibration.json"
    assert vr  == base / "valid_ranges.json"
    assert tz  == "America/Chicago"


def test_evr_workspace_mode_structured_wave(tmp_path: Path, monkeypatch):
    workspace_root, w, s, drive = _bootstrap(tmp_path)
    (drive / "Wave2" / "02_Chamber_Metadata").mkdir(parents=True)
    _patch_probe(monkeypatch, drive)

    args = _ns_evr(workspace_yaml=str(w), chamber_id="collective_104A",
                   wave_id="wave2")
    cal, vr, tz = edit_valid_ranges._resolve_paths(args)

    base = drive / "Wave2" / "02_Chamber_Metadata"
    assert cal == base / "time_calibration.json"
    assert vr  == base / "valid_ranges.json"
    assert tz  == "America/Chicago"


def test_evr_no_mode_rejected():
    with pytest.raises(SystemExit):
        edit_valid_ranges._resolve_paths(_ns_evr())


def test_evr_config_plus_workspace_rejected(tmp_path: Path):
    _, w, _, _ = _bootstrap(tmp_path)
    args = _ns_evr(config="legacy.yaml", workspace_yaml=str(w),
                   chamber_id="x", wave_id="y")
    with pytest.raises(SystemExit):
        edit_valid_ranges._resolve_paths(args)


def test_evr_workspace_partial_args_rejected(tmp_path: Path):
    _, w, _, _ = _bootstrap(tmp_path)
    args = _ns_evr(workspace_yaml=str(w))  # no chamber/wave
    with pytest.raises(SystemExit):
        edit_valid_ranges._resolve_paths(args)
