"""
tests/test_rois.py — corner resolution across both on-disk formats.

Covers avistrack/core/rois.py: chamber_corners.json (new, source of truth)
preferred over camera_rois.json (legacy), single-camera template resolution,
per-video overrides, rgb/ir stream inference, stem matching, and validation.
"""

from __future__ import annotations

import json

import pytest

from avistrack.core import rois

RGB = "Wave4_PairedCapacityPlus_Plus_Day1_Cam1_RGB.mp4"
IR = "Wave4_PairedCapacityPlus_Plus_Day1_Cam1_IR.mp4"
BIG_RGB = [[443, 25], [1029, 52], [987, 651], [375, 603]]
BIG_IR = [[511, 127], [908, 157], [876, 553], [464, 526]]


def _write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


# ── chamber_corners.json (new format) ──────────────────────────────────────

def test_chamber_template_resolves_by_stream(tmp_path):
    _write(tmp_path / "chamber_corners.json",
           {"version": 1, "cameras": {"0": {"rgb": BIG_RGB, "ir": BIG_IR}},
            "videos": {}})
    assert rois.resolve_corners(tmp_path, RGB) == BIG_RGB
    assert rois.resolve_corners(tmp_path, IR) == BIG_IR


def test_videos_override_beats_template(tmp_path):
    override = [[9, 9], [8, 8], [7, 7], [6, 6]]
    _write(tmp_path / "chamber_corners.json",
           {"cameras": {"0": {"rgb": BIG_RGB}}, "videos": {RGB: override}})
    assert rois.resolve_corners(tmp_path, RGB) == override


def test_multi_camera_raises(tmp_path):
    _write(tmp_path / "chamber_corners.json",
           {"cameras": {"0": {"rgb": BIG_RGB}, "1": {"rgb": BIG_IR}},
            "videos": {}})
    with pytest.raises(ValueError):
        rois.resolve_corners(tmp_path, RGB)


def test_missing_stream_returns_none(tmp_path):
    # template has rgb only; IR video cannot resolve from the template
    _write(tmp_path / "chamber_corners.json",
           {"cameras": {"0": {"rgb": BIG_RGB}}, "videos": {}})
    assert rois.resolve_corners(tmp_path, IR) is None


# ── camera_rois.json (legacy format) ───────────────────────────────────────

def test_camera_rois_flat_lookup(tmp_path):
    _write(tmp_path / "camera_rois.json", {RGB: BIG_RGB})
    assert rois.resolve_corners(tmp_path, RGB) == BIG_RGB


def test_camera_rois_stem_match(tmp_path):
    # stored with .mkv, queried with .mp4 → stem match
    _write(tmp_path / "camera_rois.json",
           {"Wave4_PairedCapacityPlus_Plus_Day1_Cam1_RGB.mkv": BIG_RGB})
    assert rois.resolve_corners(tmp_path, RGB) == BIG_RGB


# ── precedence + fallback ──────────────────────────────────────────────────

def test_prefers_chamber_over_camera_rois(tmp_path):
    _write(tmp_path / "chamber_corners.json",
           {"cameras": {"0": {"rgb": BIG_RGB}}, "videos": {}})
    _write(tmp_path / "camera_rois.json",
           {RGB: [[0, 0], [0, 0], [0, 0], [0, 0]]})
    assert rois.resolve_corners(tmp_path, RGB) == BIG_RGB
    assert rois.preferred_roi_file(tmp_path).name == "chamber_corners.json"


def test_falls_back_when_chamber_has_no_match(tmp_path):
    # chamber_corners present but empty → fall back to camera_rois
    _write(tmp_path / "chamber_corners.json", {"cameras": {}, "videos": {}})
    _write(tmp_path / "camera_rois.json", {RGB: BIG_RGB})
    assert rois.resolve_corners(tmp_path, RGB) == BIG_RGB


def test_no_source_returns_none(tmp_path):
    assert rois.resolve_corners(tmp_path, RGB) is None
    assert rois.preferred_roi_file(tmp_path) is None


# ── validation ─────────────────────────────────────────────────────────────

def test_validate_all_ok(tmp_path):
    _write(tmp_path / "chamber_corners.json",
           {"cameras": {"0": {"rgb": BIG_RGB, "ir": BIG_IR}}, "videos": {}})
    ok, msgs = rois.validate_corners(tmp_path, [RGB, IR])
    assert ok is True
    assert any("chamber_corners.json" in m for m in msgs)


def test_validate_reports_missing(tmp_path):
    _write(tmp_path / "chamber_corners.json",
           {"cameras": {"0": {"rgb": BIG_RGB}}, "videos": {}})
    ok, msgs = rois.validate_corners(tmp_path, [RGB, IR])
    assert ok is False
    assert any("1/2" in m for m in msgs)


def test_validate_no_source(tmp_path):
    ok, msgs = rois.validate_corners(tmp_path, [RGB])
    assert ok is False


# ── explicit-file API (used by transformer.from_roi_file) ──────────────────

def test_corners_from_file_both_formats(tmp_path):
    cc = tmp_path / "chamber_corners.json"
    _write(cc, {"cameras": {"0": {"rgb": BIG_RGB}}, "videos": {}})
    assert rois.corners_from_file(cc, RGB) == BIG_RGB

    cr = tmp_path / "camera_rois.json"
    _write(cr, {RGB: BIG_RGB})
    assert rois.corners_from_file(cr, RGB) == BIG_RGB
