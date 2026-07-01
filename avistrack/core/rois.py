"""
avistrack/core/rois.py
──────────────────────
Corner (ROI) resolution with backward compatibility across two on-disk
formats, tried in priority order:

1. ``chamber_corners.json`` — written by ChamberBroadcaster at capture time
   (corners must be picked *before* recording starts, so every new wave has
   one). Current source of truth. Schema::

       {"version": 1,
        "cameras": {"<cam_id>": {"rgb": [[x, y] * 4], "ir": [[x, y] * 4]}},
        "videos":  {"<filename>": [[x, y] * 4]}}     # optional per-video

   ``videos`` holds optional per-video overrides; ``cameras`` holds the
   per-camera/stream template applied to every video of that stream.

2. ``camera_rois.json`` — legacy AvisTrack format produced by
   ``tools/pick_rois.py``. Flat ``{"<filename>": [[x, y] * 4]}``. Used by
   pre-ChamberBroadcaster waves (e.g. Wave3) and as a manual override.

New data reads (1) with no manual picking; old data falls back to (2). Only
single-camera chambers are auto-resolved from the ``cameras`` template (the
Plus chamber has one camera); multi-camera chambers must carry explicit
``videos`` entries or a ``camera_rois.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

CHAMBER_CORNERS_NAME = "chamber_corners.json"
CAMERA_ROIS_NAME = "camera_rois.json"


# ── format helpers ─────────────────────────────────────────────────────────

def _is_valid_corners(corners) -> bool:
    """True for a list of exactly four [x, y] numeric pairs."""
    return (
        isinstance(corners, list)
        and len(corners) == 4
        and all(
            isinstance(p, list) and len(p) == 2
            and all(isinstance(v, (int, float)) for v in p)
            for p in corners
        )
    )


def _stream_of(video_name: str) -> str:
    """Infer modality ('ir' / 'rgb') from the video filename's modality token.

    Matches the underscore-delimited token ``IR`` rather than a bare substring,
    so names like ``PairedCapacity`` — whose letters contain "ir" — are not
    misclassified as infrared.
    """
    tokens = Path(video_name).stem.upper().replace("-", "_").split("_")
    return "ir" if "IR" in tokens else "rgb"


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_chamber_corners(data) -> bool:
    """A chamber_corners.json carries a 'cameras' and/or 'videos' section."""
    return isinstance(data, dict) and ("cameras" in data or "videos" in data)


def _match_in_mapping(mapping: dict, video_name: str) -> Optional[list]:
    """Look up corners by exact filename, then by filename stem."""
    val = mapping.get(video_name)
    if _is_valid_corners(val):
        return val
    stem = Path(video_name).stem
    for k, v in mapping.items():
        if Path(k).stem == stem and _is_valid_corners(v):
            return v
    return None


# ── per-format resolution ──────────────────────────────────────────────────

def corners_from_chamber(data: dict, video_name: str) -> Optional[list]:
    """Resolve corners from a parsed chamber_corners.json for one video."""
    # 1. explicit per-video override wins
    hit = _match_in_mapping(data.get("videos") or {}, video_name)
    if hit is not None:
        return hit
    # 2. per-camera/stream template (single-camera chambers only)
    cameras = data.get("cameras") or {}
    if not cameras:
        return None
    if len(cameras) > 1:
        raise ValueError(
            f"chamber_corners.json defines {len(cameras)} cameras "
            f"({sorted(cameras)}); automatic corner resolution supports only "
            f"single-camera chambers. Add a 'videos' entry for {video_name!r} "
            f"or provide a camera_rois.json."
        )
    (cam_streams,) = cameras.values()
    corners = (cam_streams or {}).get(_stream_of(video_name))
    return corners if _is_valid_corners(corners) else None


def corners_from_camera_rois(data: dict, video_name: str) -> Optional[list]:
    """Resolve corners from a parsed legacy camera_rois.json for one video."""
    if not isinstance(data, dict):
        return None
    return _match_in_mapping(data, video_name)


def corners_from_file(roi_path, video_name: str) -> Optional[list]:
    """Resolve corners from an explicit ROI file of either format."""
    data = _load_json(Path(roi_path))
    if is_chamber_corners(data):
        return corners_from_chamber(data, video_name)
    return corners_from_camera_rois(data, video_name)


# ── directory-level API (used by the workspace + sample_clips) ─────────────

def preferred_roi_file(metadata_dir) -> Optional[Path]:
    """Return the corner file to use in ``metadata_dir``, preferring the new
    chamber_corners.json over the legacy camera_rois.json. None if neither
    exists."""
    metadata_dir = Path(metadata_dir)
    for name in (CHAMBER_CORNERS_NAME, CAMERA_ROIS_NAME):
        p = metadata_dir / name
        if p.exists():
            return p
    return None


def resolve_corners(metadata_dir, video_name: str) -> Optional[list]:
    """
    Resolve 4 corners for ``video_name`` from ``metadata_dir``.

    Prefers chamber_corners.json; if that file exists but yields no corners
    for this video, falls back to camera_rois.json. Returns None when neither
    source provides corners.
    """
    metadata_dir = Path(metadata_dir)
    cc = metadata_dir / CHAMBER_CORNERS_NAME
    if cc.exists():
        corners = corners_from_chamber(_load_json(cc), video_name)
        if corners is not None:
            return corners
    cr = metadata_dir / CAMERA_ROIS_NAME
    if cr.exists():
        corners = corners_from_camera_rois(_load_json(cr), video_name)
        if corners is not None:
            return corners
    return None


def validate_corners(metadata_dir, video_names) -> tuple[bool, list[str]]:
    """
    Check that every video in ``video_names`` resolves to valid corners.

    Returns (ok, messages); ``ok`` is True only when all videos resolve.
    Message style mirrors tools/pick_rois.validate_roi_file but works across
    both on-disk formats.
    """
    metadata_dir = Path(metadata_dir)
    msgs: list[str] = []
    src = preferred_roi_file(metadata_dir)
    if src is None:
        msgs.append(
            f"❌ No corner file in {metadata_dir} "
            f"(looked for {CHAMBER_CORNERS_NAME}, {CAMERA_ROIS_NAME})"
        )
        return False, msgs
    msgs.append(f"✅ Corner source: {src.name}")

    missing: list[str] = []
    for name in video_names:
        try:
            corners = resolve_corners(metadata_dir, name)
        except ValueError as exc:
            msgs.append(f"❌ {exc}")
            return False, msgs
        if corners is None:
            missing.append(name)

    if missing:
        msgs.append(f"❌ {len(missing)}/{len(video_names)} videos have NO corners:")
        for m in missing[:10]:
            msgs.append(f"   {m}")
        if len(missing) > 10:
            msgs.append(f"   … and {len(missing) - 10} more")
        return False, msgs

    msgs.append(f"✅ All {len(video_names)} videos resolved to valid corners")
    return True, msgs
