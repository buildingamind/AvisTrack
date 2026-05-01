#!/usr/bin/env python3
"""
tools/scan_legacy_wave.py
─────────────────────────
Scan a chamber drive for a legacy wave's videos and write a
``video_index.json`` manifest into the chamber's metadata directory.

This is the entry point for onboarding a wave that pre-dates the
structured layout (no ``00_raw_videos/`` folder, no
``02_Chamber_Metadata/``). The tool

  1. Resolves ``(chamber_id, wave_id)`` via the workspace + sources
     yamls — chamber drive must be mounted.
  2. Walks ``raw_videos_glob`` (legacy) or ``raw_videos_subpath``
     (structured fallback) under the wave root.
  3. Probes each video with cv2 for fps / frame count / resolution
     (best-effort; unreadable files are still listed with cv2 fields
     null so their existence is captured).
  4. Writes ``{metadata_dir}/video_index.json`` — for legacy waves the
     metadata_dir is ``{chamber_root}/_avistrack_added/{wave_id}/``.
  5. Prints next-step hints for ``pick_rois.py`` and
     ``edit_valid_ranges.py`` in workspace mode.

Re-running is idempotent: the existing ``video_index.json`` is
overwritten with fresh probe results. Videos that have disappeared
since the last scan simply drop out.

Usage
-----
    python tools/scan_legacy_wave.py \\
        --workspace-yaml /media/ssd/avistrack/collective/workspace.yaml \\
        --chamber-id     collective_104A \\
        --wave-id        wave1_legacy
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from avistrack.workspace import ChamberWaveContext, load_context

VIDEO_INDEX_FILENAME = "video_index.json"


def _classify_modality(stem: str) -> str:
    up = stem.upper()
    if "RGB" in up:
        return "rgb"
    if "IR" in up:
        return "ir"
    return "unknown"


def _probe_video(path: Path) -> dict:
    """
    Best-effort cv2 probe. Returns a dict with
    ``fps / frame_count / duration_sec / width / height`` populated when
    the file opens, otherwise the same keys with ``None`` values.

    cv2 is imported lazily so test environments without opencv-python
    can still import this module to exercise the path-listing helpers.
    """
    try:
        import cv2
    except ImportError:
        return {"fps": None, "frame_count": None, "duration_sec": None,
                "width": None, "height": None}

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return {"fps": None, "frame_count": None, "duration_sec": None,
                    "width": None, "height": None}
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    if not fps or n < 1:
        return {"fps": fps, "frame_count": n if n > 0 else None,
                "duration_sec": None,
                "width": w or None, "height": h or None}
    return {
        "fps":          float(fps),
        "frame_count":  int(n),
        "duration_sec": round(n / fps, 3),
        "width":        int(w),
        "height":       int(h),
    }


def _list_videos_for_index(ctx: ChamberWaveContext) -> list[Path]:
    """
    Enumerate every video the wave might use, regardless of modality.
    For legacy waves we want the index to capture both RGB and IR (and
    anything in between) so the user can decide later what to keep.
    """
    rgb = ctx.list_videos(modality="rgb")
    ir  = ctx.list_videos(modality="ir")
    seen: dict[Path, None] = {}
    for v in rgb + ir:
        seen.setdefault(v, None)
    return sorted(seen.keys())


def build_video_index_payload(
    ctx: ChamberWaveContext,
    videos: list[Path],
    *,
    probe: bool = True,
    now: Optional[datetime] = None,
) -> dict:
    """
    Build the ``video_index.json`` payload.

    Split out from :func:`scan_legacy_wave` so tests can exercise the
    JSON shape without a cv2 install. ``probe=False`` skips the cv2
    probe entirely (useful when the videos are placeholders).
    """
    if now is None:
        now = datetime.now(timezone.utc)

    wave_root = ctx.wave_root
    entries: list[dict] = []
    for v in videos:
        try:
            rel = v.relative_to(wave_root)
        except ValueError:
            rel = v
        size = v.stat().st_size if v.exists() else None
        entry = {
            "filename":     v.name,
            "rel_path":     str(rel).replace("\\", "/"),
            "abs_path":     str(v),
            "size_bytes":   size,
            "modality":     _classify_modality(v.stem),
        }
        if probe:
            entry.update(_probe_video(v))
        entries.append(entry)

    return {
        "scan_at":     now.isoformat(timespec="seconds"),
        "chamber_id":  ctx.chamber.chamber_id,
        "wave_id":     ctx.wave.wave_id,
        "layout":      ctx.wave.layout,
        "drive_uuid":  ctx.chamber.drive_uuid,
        "wave_root":   str(wave_root),
        "video_count": len(entries),
        "videos":      entries,
    }


def write_video_index(payload: dict, metadata_dir: Path) -> Path:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    target = metadata_dir / VIDEO_INDEX_FILENAME
    with open(target, "w") as f:
        json.dump(payload, f, indent=2)
    return target


def scan_legacy_wave(
    ctx: ChamberWaveContext,
    *,
    probe: bool = True,
) -> tuple[Path, dict]:
    """Run a full scan + write. Returns (video_index_path, payload)."""
    videos = _list_videos_for_index(ctx)
    if not videos:
        raise SystemExit(
            f"No videos found for chamber {ctx.chamber.chamber_id!r} "
            f"wave {ctx.wave.wave_id!r} under {ctx.wave_root}. "
            "Check the wave_subpath / raw_videos_glob in sources.yaml."
        )
    payload = build_video_index_payload(ctx, videos, probe=probe)
    target  = write_video_index(payload, ctx.metadata_dir)
    return target, payload


# ── CLI ──────────────────────────────────────────────────────────────────

def _print_next_steps(ctx: ChamberWaveContext, target: Path) -> None:
    print()
    print(f"📋 Wrote {target}")
    print()
    print("Next steps for this wave:")
    print()
    print("  1. Pick chamber ROIs (interactive):")
    print(f"     python tools/pick_rois.py \\")
    print(f"         --workspace-yaml {ctx.workspace_chamber_dir / 'workspace.yaml'} \\")
    print(f"         --chamber-id {ctx.chamber.chamber_id} \\")
    print(f"         --wave-id    {ctx.wave.wave_id}")
    print()
    print("  2. Calibrate time + record valid ranges:")
    print(f"     python tools/calibrate_time.py calibrate \\")
    print(f"         --workspace-yaml {ctx.workspace_chamber_dir / 'workspace.yaml'} \\")
    print(f"         --chamber-id {ctx.chamber.chamber_id} \\")
    print(f"         --wave-id    {ctx.wave.wave_id}")
    print(f"     python tools/edit_valid_ranges.py \\")
    print(f"         --workspace-yaml {ctx.workspace_chamber_dir / 'workspace.yaml'} \\")
    print(f"         --chamber-id {ctx.chamber.chamber_id} \\")
    print(f"         --wave-id    {ctx.wave.wave_id}")
    print()
    print("  3. Sample clips:")
    print(f"     python tools/sample_clips.py \\")
    print(f"         --workspace-yaml {ctx.workspace_chamber_dir / 'workspace.yaml'} \\")
    print(f"         --chamber-id {ctx.chamber.chamber_id} \\")
    print(f"         --wave-id    {ctx.wave.wave_id}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Scan a chamber drive's wave videos into video_index.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--workspace-yaml", required=True, type=Path,
                   help="Path to {workspace_root}/{chamber_type}/workspace.yaml")
    p.add_argument("--sources-yaml",   type=Path,
                   help="Path to sources.yaml (default: sibling of workspace.yaml)")
    p.add_argument("--chamber-id",     required=True,
                   help="Chamber identifier from sources.yaml")
    p.add_argument("--wave-id",        required=True,
                   help="Wave identifier from sources.yaml")
    p.add_argument("--no-probe", action="store_true",
                   help="Skip cv2 probe; record paths only")
    args = p.parse_args()

    sources_yaml = args.sources_yaml or args.workspace_yaml.with_name("sources.yaml")
    if not sources_yaml.exists():
        raise SystemExit(
            f"sources.yaml not found at {sources_yaml}. "
            "Run tools/init_chamber_workspace.py + "
            "tools/register_chamber_source.py first.")

    ctx = load_context(
        workspace_yaml=args.workspace_yaml,
        sources_yaml=sources_yaml,
        chamber_id=args.chamber_id,
        wave_id=args.wave_id,
        require_drive=True,
    )

    print(f"🔍 Scanning {ctx.chamber.chamber_id} / {ctx.wave.wave_id} "
          f"(layout={ctx.wave.layout}) under {ctx.wave_root} ...")
    target, payload = scan_legacy_wave(ctx, probe=not args.no_probe)
    print(f"   {payload['video_count']} video(s) indexed.")
    _print_next_steps(ctx, target)


if __name__ == "__main__":
    main()
