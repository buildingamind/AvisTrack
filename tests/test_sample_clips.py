"""
tests/test_sample_clips.py – pure helpers in tools/sample_clips.py.

The full sampling driver needs cv2 + real videos and is exercised by a
manual smoke test on the wave2 chamber drive (Step C verification in
improve-plan.md). This file unit-tests the manifest writer and the
overlap / filename helpers, which are the regression-prone pieces.

We stub cv2 in sys.modules before importing the tool so the import does
not require opencv-python in the test environment.
"""

from __future__ import annotations

import csv
import importlib
import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _import_sample_clips():
    """Load tools/sample_clips.py, stubbing cv2 only if it's truly missing.

    A blanket stub would leak into other test modules (e.g.
    test_transformer.py) and break them, so we only install the stub when
    real opencv is unavailable in this environment.
    """
    try:
        import cv2  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["cv2"] = types.ModuleType("cv2")
    path = REPO_ROOT / "tools" / "sample_clips.py"
    spec = importlib.util.spec_from_file_location("sample_clips_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sc = _import_sample_clips()


# ── too_close ────────────────────────────────────────────────────────────

def test_too_close_overlapping_intervals():
    existing = [("vidA.mkv", 100.0, 110.0)]
    # New clip [105, 115) overlaps – reject.
    assert sc.too_close(105, 115, existing, "vidA.mkv", 0) is True


def test_too_close_within_min_gap():
    existing = [("vidA.mkv", 100.0, 110.0)]
    # 30s after the existing clip's end, but min_gap = 60s → reject.
    assert sc.too_close(140, 143, existing, "vidA.mkv", 60) is True


def test_too_close_outside_min_gap():
    existing = [("vidA.mkv", 100.0, 110.0)]
    # 200s after existing end; min_gap 60s → ok.
    assert sc.too_close(310, 313, existing, "vidA.mkv", 60) is False


def test_too_close_different_video():
    existing = [("vidA.mkv", 100.0, 110.0)]
    # Same time window but a different source video → independent.
    assert sc.too_close(100, 110, existing, "vidB.mkv", 60) is False


# ── build_clip_name ──────────────────────────────────────────────────────

def test_build_clip_name_includes_chamber_and_wave():
    name = sc.build_clip_name(
        chamber_id="collective_104A", wave_id="wave2",
        video_stem="Day1_Cam1_RGB", start_sec=37.42, transformed=True,
    )
    assert name == "collective_104A_wave2_Day1_Cam1_RGB_s37_transformed.mp4"


def test_build_clip_name_no_transform_tag():
    name = sc.build_clip_name(
        chamber_id="vr_201", wave_id="wave1_legacy",
        video_stem="raw_clip", start_sec=0.0, transformed=False,
    )
    assert name == "vr_201_wave1_legacy_raw_clip_s0.mp4"


# ── append_to_all_clips / existing_intervals_for_wave ────────────────────

def _row(**overrides):
    base = {
        "clip_path":         "clips/collective_104A/wave2/foo.mp4",
        "chamber_id":        "collective_104A",
        "wave_id":           "wave2",
        "source_video":      "Day1_RGB.mkv",
        "source_drive_uuid": "ABCD-1234",
        "layout":            "structured",
        "start_sec":         "100.00",
        "duration_sec":      "3.00",
        "fps":               "30.000",
        "sampled_at":        "2026-05-01T12:00:00+00:00",
    }
    base.update(overrides)
    return base


def test_append_writes_header_then_appends(tmp_path: Path):
    csv_path = tmp_path / "manifests" / "all_clips.csv"
    sc.append_to_all_clips(csv_path, [_row(start_sec="10.00")])
    sc.append_to_all_clips(csv_path, [_row(start_sec="20.00")])

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert [r["start_sec"] for r in rows] == ["10.00", "20.00"]
    # Header is written exactly once – the file has no second header line.
    text = csv_path.read_text().splitlines()
    assert text[0].startswith("clip_path")
    assert sum(1 for line in text if line.startswith("clip_path")) == 1


def test_append_no_rows_is_noop(tmp_path: Path):
    csv_path = tmp_path / "manifests" / "all_clips.csv"
    sc.append_to_all_clips(csv_path, [])
    assert not csv_path.exists()


def test_existing_intervals_filter_by_chamber_and_wave(tmp_path: Path):
    csv_path = tmp_path / "manifests" / "all_clips.csv"
    sc.append_to_all_clips(csv_path, [
        _row(chamber_id="collective_104A", wave_id="wave2",
             source_video="Day1_RGB.mkv", start_sec="10.00", duration_sec="3.00"),
        _row(chamber_id="collective_104A", wave_id="wave2",
             source_video="Day1_RGB.mkv", start_sec="20.00", duration_sec="3.00"),
        # different wave – must be ignored
        _row(chamber_id="collective_104A", wave_id="wave1_legacy",
             source_video="Day1_RGB.mkv", start_sec="100.00", duration_sec="3.00"),
        # different chamber – must be ignored
        _row(chamber_id="collective_104B", wave_id="wave2",
             source_video="Day1_RGB.mkv", start_sec="200.00", duration_sec="3.00"),
    ])

    intervals = sc.existing_intervals_for_wave(
        csv_path, chamber_id="collective_104A", wave_id="wave2",
    )
    assert intervals == [
        ("Day1_RGB.mkv", 10.0, 13.0),
        ("Day1_RGB.mkv", 20.0, 23.0),
    ]


def test_existing_intervals_missing_csv(tmp_path: Path):
    assert sc.existing_intervals_for_wave(
        tmp_path / "nope.csv", "any", "any") == []


def test_all_clips_fields_match_writer_and_reader():
    """Schema sanity: keys we write are the keys the reader looks for."""
    expected = {
        "clip_path", "chamber_id", "wave_id", "source_video",
        "source_drive_uuid", "layout", "start_sec", "duration_sec",
        "fps", "sampled_at",
    }
    assert set(sc.ALL_CLIPS_FIELDS) == expected
