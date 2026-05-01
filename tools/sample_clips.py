#!/usr/bin/env python3
"""
tools/sample_clips.py
──────────────────────
Sample short clips from a (chamber_id, wave_id)'s raw videos into the
chamber-type workspace.

Refactored for the multi-chamber storage layout (improve-plan.md §1):

* clips land in
  ``{workspace}/{chamber_type}/clips/{chamber_id}/{wave_id}/``
* every clip is appended to ``manifests/all_clips.csv`` – the single
  source-of-truth for what has been sampled across all splits.
* per-split manifests are gone; splits are decided later by recipes
  consumed by ``tools/build_dataset.py``.

The sampling logic itself is unchanged: weighted by frame count,
``--min-gap`` enforced between clips of the same source video,
``valid_ranges.json`` honoured if present, and an ROI-driven
PerspectiveTransformer applied unless ``--no-transform`` is passed.

Usage
-----
    python tools/sample_clips.py \\
        --workspace-yaml /media/ssd/avistrack/collective/workspace.yaml \\
        --chamber-id     collective_104A \\
        --wave-id        wave2 \\
        --n              20 \\
        --duration       3
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from avistrack.core.time_lookup import TimeLookup
from avistrack.core.transformer import PerspectiveTransformer
from avistrack.workspace import ChamberWaveContext, load_context
from tools.pick_rois import validate_roi_file


VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov"}

ALL_CLIPS_FIELDS = [
    "clip_path",         # relative to workspace_chamber_dir
    "chamber_id",
    "wave_id",
    "source_video",      # basename of source on chamber drive
    "source_drive_uuid", # for offline traceback
    "layout",            # "structured" | "legacy"
    "start_sec",
    "duration_sec",
    "fps",
    "sampled_at",        # ISO 8601 UTC
]


# ── Video metadata ────────────────────────────────────────────────────────

def probe_videos(videos: list[Path]) -> dict[Path, dict]:
    """Return {path: {fps, n_frames, total_sec}} for openable videos."""
    info = {}
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            cap.release()
            continue
        fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n_frames < 1:
            continue
        info[v] = {
            "fps":       fps,
            "n_frames":  n_frames,
            "total_sec": n_frames / fps,
        }
    return info


# ── Manifest helpers ─────────────────────────────────────────────────────

def existing_intervals_for_wave(
    all_clips_csv: Path,
    chamber_id: str,
    wave_id: str,
) -> list[tuple[str, float, float]]:
    """
    Return [(source_video, start_sec, end_sec)] already sampled for this
    (chamber, wave). Matching is by chamber_id + wave_id only – the same
    physical drive may be re-mounted under a different path between runs.
    """
    if not all_clips_csv.exists():
        return []
    out = []
    with open(all_clips_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("chamber_id") != chamber_id:
                continue
            if row.get("wave_id") != wave_id:
                continue
            try:
                start = float(row.get("start_sec", 0))
                dur   = float(row.get("duration_sec", 0))
            except ValueError:
                continue
            out.append((row.get("source_video", ""), start, start + dur))
    return out


def append_to_all_clips(all_clips_csv: Path, rows: list[dict]) -> None:
    if not rows:
        return
    all_clips_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not all_clips_csv.exists() or all_clips_csv.stat().st_size == 0
    with open(all_clips_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_CLIPS_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Appended {len(rows)} rows → {all_clips_csv}")


def too_close(new_start: float, new_end: float,
              existing: list[tuple[str, float, float]],
              video_name: str,
              min_gap_sec: float) -> bool:
    """True if the new interval overlaps or is within ``min_gap_sec`` of an existing one."""
    for src, s, e in existing:
        if src != video_name:
            continue
        if new_start < (e + min_gap_sec) and new_end > (s - min_gap_sec):
            return True
    return False


# ── ROI helpers ───────────────────────────────────────────────────────────

def load_roi_json(roi_path: Path) -> dict:
    with open(roi_path) as f:
        return json.load(f)


def find_roi_for_video(rois: dict, video_name: str) -> Optional[list]:
    if video_name in rois:
        return rois[video_name]
    stem = Path(video_name).stem
    for key, corners in rois.items():
        if Path(key).stem == stem:
            return corners
    return None


# ── Valid range filtering ────────────────────────────────────────────────

def load_valid_ranges(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _parse_ranges(valid_ranges: dict, tz: ZoneInfo) -> dict:
    parsed: dict[str, list[tuple[datetime, datetime]]] = {}
    for vname, vranges in valid_ranges.items():
        parsed[vname] = []
        for r in vranges:
            s = datetime.fromisoformat(r["start"]).replace(tzinfo=tz)
            e = datetime.fromisoformat(r["end"]).replace(tzinfo=tz)
            parsed[vname].append((s, e))
    return parsed


def in_valid_range(
    video_name: str,
    start_sec: float,
    end_sec: float,
    fps: float,
    lookups: dict[str, TimeLookup],
    parsed_ranges: dict[str, list[tuple[datetime, datetime]]],
) -> bool:
    if video_name not in parsed_ranges:
        return False
    if video_name not in lookups:
        return True
    lookup = lookups[video_name]
    start_dt = lookup.frame_to_datetime(int(start_sec * fps))
    end_dt   = lookup.frame_to_datetime(int(end_sec * fps))
    for rs, re_ in parsed_ranges[video_name]:
        if start_dt >= rs and end_dt <= re_:
            return True
    return False


# ── Clip extraction ──────────────────────────────────────────────────────

def extract_clip(
    video_path: str,
    start_sec: float,
    duration: float,
    output_path: str,
    transformer: Optional[PerspectiveTransformer] = None,
) -> bool:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ Cannot open: {video_path}")
        return False

    fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_sec * fps)
    n_frames    = int(duration  * fps)

    if transformer is not None:
        out_w, out_h = transformer.output_size
    else:
        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    while written < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if transformer is not None:
            frame = transformer.transform(frame)
        out.write(frame)
        written += 1

    cap.release()
    out.release()
    return written > 0


# ── Filename convention ──────────────────────────────────────────────────

def build_clip_name(
    chamber_id: str,
    wave_id: str,
    video_stem: str,
    start_sec: float,
    transformed: bool,
) -> str:
    """Names always include chamber_id + wave_id so cross-drive moves are safe."""
    tag = "_transformed" if transformed else ""
    return f"{chamber_id}_{wave_id}_{video_stem}_s{int(start_sec)}{tag}.mp4"


# ── Sampling driver ──────────────────────────────────────────────────────

def sample_clips(
    ctx: ChamberWaveContext,
    n: int,
    duration: float,
    seed: int,
    modality: str,
    min_gap_min: float,
    no_transform: bool,
) -> int:
    """Run the sampling loop. Returns the number of clips successfully written."""
    random.seed(seed)
    min_gap_sec = min_gap_min * 60.0

    # ── 1. Source videos on the chamber drive ────────────────────────
    all_videos = ctx.list_videos(modality=modality)
    if not all_videos:
        raise SystemExit(
            f"No {modality.upper()} videos found for chamber "
            f"{ctx.chamber.chamber_id!r}/{ctx.wave.wave_id!r} under "
            f"{ctx.chamber_root}"
        )
    print(f"📹 Found {len(all_videos)} {modality.upper()} video(s) on "
          f"chamber drive {ctx.chamber.chamber_id} (wave {ctx.wave.wave_id}).")

    # ── 2. ROI validation (unless --no-transform) ────────────────────
    rois: dict = {}
    if not no_transform:
        roi_path = ctx.roi_file
        ok, msgs = validate_roi_file(str(roi_path), [v.name for v in all_videos])
        for m in msgs:
            print(f"  {m}")
        if not ok:
            print()
            print("   Pick missing ROIs with:")
            print(f"     python tools/pick_rois.py pick \\")
            print(f"         --video-dir {ctx.wave_root} \\")
            print(f"         --roi-file  {roi_path}")
            print("   Or pass --no-transform to skip perspective-correction.")
            raise SystemExit(1)
        rois = load_roi_json(roi_path)

    target_size = (tuple(ctx.workspace.chamber.target_size)
                   if ctx.workspace.chamber.target_size else None)

    # ── 3. Probe videos + build weighted pool ────────────────────────
    video_info = probe_videos(all_videos)
    min_total_sec = duration + 10
    eligible = {v: info for v, info in video_info.items()
                if info["total_sec"] >= min_total_sec}
    if not eligible:
        raise SystemExit(f"No videos long enough (need ≥ {min_total_sec:.0f}s)")
    print(f"   {len(eligible)}/{len(all_videos)} are long enough "
          f"(≥ {min_total_sec:.0f}s)")

    pool_videos  = list(eligible.keys())
    pool_weights = [eligible[v]["n_frames"] for v in pool_videos]
    total_frames = sum(pool_weights)
    print(f"   Total frames across pool: {total_frames:,}")

    # ── 4. Existing intervals from manifest ──────────────────────────
    existing = existing_intervals_for_wave(
        ctx.all_clips_csv, ctx.chamber.chamber_id, ctx.wave.wave_id)
    print(f"ℹ️  {len(existing)} clips already in manifest for this wave; "
          f"min gap = {min_gap_min} min ({min_gap_sec:.0f}s).")

    # ── 5. Optional valid-range filtering ────────────────────────────
    parsed_ranges: dict = {}
    lookups: dict[str, TimeLookup] = {}
    valid_ranges_raw = load_valid_ranges(ctx.valid_ranges_file)
    if valid_ranges_raw:
        cal_path = ctx.time_calibration_file
        if not cal_path.exists():
            raise SystemExit(
                f"valid_ranges.json exists at {ctx.valid_ranges_file} but "
                f"time_calibration.json not found at {cal_path}. Run "
                f"`tools/calibrate_time.py calibrate` first."
            )
        with open(cal_path) as f:
            calibration = json.load(f)
        tz_str = ctx.workspace.time.timezone
        tz = ZoneInfo(tz_str)
        for vname in valid_ranges_raw:
            if vname in calibration:
                lookups[vname] = TimeLookup.from_calibration(
                    calibration, vname, tz_str)
        parsed_ranges = _parse_ranges(valid_ranges_raw, tz)

        n_ranges = sum(len(v) for v in valid_ranges_raw.values())
        print(f"⏱️  Valid ranges: {n_ranges} range(s) across "
              f"{len(valid_ranges_raw)} video(s).")
        pool_videos  = [v for v in pool_videos if v.name in parsed_ranges]
        pool_weights = [eligible[v]["n_frames"] for v in pool_videos]
        if not pool_videos:
            raise SystemExit("No eligible videos have valid ranges defined.")
        print(f"   Pool narrowed to {len(pool_videos)} video(s).")

    # ── 6. Pre-build transformers (one per video) ────────────────────
    transformers: dict[str, PerspectiveTransformer] = {}
    if rois:
        for v in pool_videos:
            corners = find_roi_for_video(rois, v.name)
            if corners:
                transformers[v.name] = PerspectiveTransformer(corners, target_size)

    # ── 7. Output dir + sampling loop ───────────────────────────────
    output_dir = ctx.clip_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    new_rows: list[dict] = []
    attempts = 0
    max_attempts = n * 100

    print(f"\n🎲 Sampling {n} clips × {duration}s (seed={seed}) "
          f"into {output_dir} ...\n")

    while len(new_rows) < n and attempts < max_attempts:
        attempts += 1

        video = random.choices(pool_videos, weights=pool_weights, k=1)[0]
        info  = eligible[video]
        start_sec = random.uniform(5, info["total_sec"] - duration - 5)
        end_sec   = start_sec + duration

        if too_close(start_sec, end_sec, existing, video.name, min_gap_sec):
            continue

        if parsed_ranges:
            if not in_valid_range(video.name, start_sec, end_sec,
                                  info["fps"], lookups, parsed_ranges):
                continue

        clip_name = build_clip_name(
            chamber_id=ctx.chamber.chamber_id,
            wave_id=ctx.wave.wave_id,
            video_stem=video.stem,
            start_sec=start_sec,
            transformed=bool(rois),
        )
        clip_path = output_dir / clip_name

        tf = transformers.get(video.name)
        print(f"  [{len(new_rows)+1}/{n}] {clip_name} …", end=" ", flush=True)
        ok = extract_clip(str(video), start_sec, duration,
                          str(clip_path), transformer=tf)
        if not ok:
            print("failed")
            continue
        print("✅")

        rel_path = clip_path.relative_to(ctx.workspace_chamber_dir)
        new_rows.append({
            "clip_path":         str(rel_path).replace("\\", "/"),
            "chamber_id":        ctx.chamber.chamber_id,
            "wave_id":           ctx.wave.wave_id,
            "source_video":      video.name,
            "source_drive_uuid": ctx.chamber.drive_uuid,
            "layout":            ctx.wave.layout,
            "start_sec":         f"{start_sec:.2f}",
            "duration_sec":      f"{duration:.2f}",
            "fps":               f"{info['fps']:.3f}",
            "sampled_at":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        existing.append((video.name, start_sec, end_sec))

    if len(new_rows) < n:
        print(f"\n⚠️  Only found {len(new_rows)}/{n} non-overlapping clips "
              f"after {attempts} attempts.")

    append_to_all_clips(ctx.all_clips_csv, new_rows)
    return len(new_rows)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Sample clips from a chamber drive into the workspace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--workspace-yaml", required=True, type=Path,
                   help="Path to {workspace_root}/{chamber_type}/workspace.yaml")
    p.add_argument("--sources-yaml",   type=Path,
                   help="Path to sources.yaml (default: sibling of workspace.yaml)")
    p.add_argument("--chamber-id", required=True,
                   help="Chamber identifier from sources.yaml (e.g. collective_104A)")
    p.add_argument("--wave-id",    required=True,
                   help="Wave identifier from sources.yaml (e.g. wave2)")
    p.add_argument("--n",        type=int,   default=20,
                   help="Number of new clips (default: 20)")
    p.add_argument("--duration", type=float, default=3.0,
                   help="Clip duration in seconds (default: 3)")
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--modality", default="rgb", choices=["rgb", "ir"],
                   help="Filter source videos by modality keyword (default: rgb)")
    p.add_argument("--min-gap",  type=float, default=5.0,
                   help="Minimum minutes between clips from the same source video (default: 5)")
    p.add_argument("--no-transform", action="store_true",
                   help="Skip ROI crop / perspective-correction")
    args = p.parse_args()

    sources_yaml = args.sources_yaml or args.workspace_yaml.with_name("sources.yaml")
    if not sources_yaml.exists():
        raise SystemExit(f"sources.yaml not found at {sources_yaml}. "
                         "Did you run tools/init_chamber_workspace.py + "
                         "tools/register_chamber_source.py?")

    ctx = load_context(
        workspace_yaml=args.workspace_yaml,
        sources_yaml=sources_yaml,
        chamber_id=args.chamber_id,
        wave_id=args.wave_id,
        require_drive=True,
    )

    written = sample_clips(
        ctx=ctx,
        n=args.n,
        duration=args.duration,
        seed=args.seed,
        modality=args.modality,
        min_gap_min=args.min_gap,
        no_transform=args.no_transform,
    )
    print(f"\nDone: wrote {written} clip(s) to {ctx.clip_dir}")


if __name__ == "__main__":
    main()
