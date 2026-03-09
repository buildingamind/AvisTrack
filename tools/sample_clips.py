#!/usr/bin/env python3
"""
tools/sample_clips.py
──────────────────────
Sample short clips from raw RGB videos, apply ROI perspective-correction,
and append records to the wave's manifest file.

Features
--------
* Only RGB videos by default (``--modality ir`` to switch)
* Weight-based sampling: longer videos get proportionally more clips
* ``--min-gap`` minimum minutes between any two clips in the same video
* Each clip is immediately cropped + perspective-corrected using the ROI
  JSON (``drive.roi_file``); aborts early if the ROI file is missing
* Fixed seed (``--seed``) for full reproducibility

Usage
-----
    # Minimal — everything derived from config:
    python tools/sample_clips.py --config configs/wave2_collective.yaml

    # Override defaults:
    python tools/sample_clips.py \\
        --config  configs/wave2_collective.yaml \\
        --split   test \\
        --n       20 \\
        --duration 20

Options
-------
    --config        AvisTrack YAML config  (ONLY required arg)
    --split         Which split to add to: train | val | test (default: train)
    --n             Number of new clips (default: 20)
    --duration      Clip duration in seconds (default 3)
    --output-dir    Where to write .mp4 files (default: {dataset}/{split_dir})
    --seed          Random seed for reproducibility (default 42)
    --modality      rgb (default) or ir — filters by filename keyword
    --min-gap       Minimum gap in minutes between clips from same video (default 5)
    --no-transform  Skip ROI crop / perspective-correction (extract raw clips)
"""

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import cv2
import numpy as np

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))
from avistrack.config.loader import load_config
from avistrack.core.time_lookup import TimeLookup
from avistrack.core.transformer import PerspectiveTransformer
from tools.pick_rois import validate_roi_file


SPLIT_TO_MANIFEST = {
    "train": "train_manifest",
    "val":   "val_manifest",
    "test":  "test_manifest",
}

SPLIT_TO_DIR = {
    "train": "train",
    "val":   "val_tuning",
    "test":  "test_golden",
}

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov"}


# ── Video discovery ───────────────────────────────────────────────────────

def find_videos(raw_dir: str, modality: str = "rgb") -> list[Path]:
    """
    Recursively find video files, filtered by modality keyword in filename.

    Parameters
    ----------
    raw_dir  : root directory to search
    modality : "rgb" or "ir" — matched case-insensitively in the filename
    """
    keyword = modality.upper()  # "RGB" or "IR"
    found = []
    for p in Path(raw_dir).rglob("*"):
        if p.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if keyword not in p.stem.upper():
            continue
        found.append(p)
    return sorted(found)


# ── Video metadata (cached) ──────────────────────────────────────────────

def probe_videos(videos: list[Path]) -> dict[Path, dict]:
    """
    Return {path: {"fps": float, "n_frames": int, "total_sec": float}}
    for all openable videos.
    """
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


# ── Manifest helpers ──────────────────────────────────────────────────────

def load_manifest(path: str | None) -> list[dict]:
    """Return list of dicts from a CSV manifest, or [] if path is None/missing."""
    if not path or not Path(path).exists():
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # strip whitespace from keys (existing manifests have " Original_Video_Path")
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            rows.append(cleaned)
    return rows


def load_all_sampled_intervals(cfg) -> list[tuple[str, float, float]]:
    """
    Load all already-sampled intervals from all three manifests.
    Returns a list of (video_basename, start_sec, end_sec).
    """
    intervals = []
    for key in SPLIT_TO_MANIFEST.values():
        path = getattr(cfg.drive, key, None)
        for row in load_manifest(path):
            src = row.get("Original_Video_Path", "")
            try:
                start    = float(row.get("Start_Time", 0))
                duration = float(row.get("Duration", 3))
                intervals.append((Path(src).name, start, start + duration))
            except ValueError:
                pass
    return intervals


def too_close(new_start: float, new_end: float,
              existing: list[tuple[str, float, float]],
              video_name: str,
              min_gap_sec: float) -> bool:
    """
    Return True if [new_start, new_end) overlaps or is within min_gap_sec
    of any existing interval for the same video.
    """
    for src, s, e in existing:
        if src != video_name:
            continue
        # Expand existing interval by min_gap on each side
        if new_start < (e + min_gap_sec) and new_end > (s - min_gap_sec):
            return True
    return False


def append_to_manifest(manifest_path: str, rows: list[dict]) -> None:
    path   = Path(manifest_path)
    exists = path.exists() and path.stat().st_size > 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Clip_Filename", "Original_Video_Path",
                           "Start_Time", "Duration"]
        )
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Appended {len(rows)} rows → {manifest_path}")


# ── ROI helpers ───────────────────────────────────────────────────────────

def load_roi_json(roi_path: str) -> dict:
    """Load camera_rois.json → {video_basename: [[x,y], ...]}."""
    with open(roi_path) as f:
        return json.load(f)


def find_roi_for_video(rois: dict, video_name: str) -> Optional[list]:
    """
    Look up ROI corners for a video.  Tries exact name first, then falls
    back to stem-only match.
    """
    if video_name in rois:
        return rois[video_name]
    stem = Path(video_name).stem
    for key, corners in rois.items():
        if Path(key).stem == stem:
            return corners
    return None


# ── Valid range filtering ─────────────────────────────────────────────────

def load_valid_ranges(path: str | None) -> dict:
    """Load valid_ranges.json → {video_name: [{start, end}, ...]}."""
    if not path or not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _parse_ranges(valid_ranges: dict, tz: ZoneInfo) -> dict:
    """Pre-parse ISO strings into tz-aware datetimes for fast lookup."""
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
    """Return True if the clip falls entirely within a valid range."""
    if video_name not in parsed_ranges:
        return False  # video has no valid ranges → exclude
    if video_name not in lookups:
        return True   # no calibration → can't check, allow

    lookup = lookups[video_name]
    start_dt = lookup.frame_to_datetime(int(start_sec * fps))
    end_dt   = lookup.frame_to_datetime(int(end_sec * fps))

    for rs, re_ in parsed_ranges[video_name]:
        if start_dt >= rs and end_dt <= re_:
            return True
    return False


# ── Clip extraction with transform ───────────────────────────────────────

def extract_clip(
    video_path: str,
    start_sec: float,
    duration: float,
    output_path: str,
    transformer: Optional[PerspectiveTransformer] = None,
) -> bool:
    """
    Extract [start_sec, start_sec+duration] from video_path → output_path.
    If a transformer is provided, each frame is perspective-corrected before
    writing.
    """
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


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sample clips from raw videos with ROI perspective-correction.")
    parser.add_argument("--config",     required=True,
                        help="AvisTrack YAML config (only required argument)")
    parser.add_argument("--split",      default="train", choices=["train", "val", "test"],
                        help="Which split (default: train)")
    parser.add_argument("--n",          type=int, default=20,
                        help="Number of new clips (default: 20)")
    parser.add_argument("--duration",   type=float, default=3.0,
                        help="Clip duration in seconds (default: 3)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write .mp4 files "
                             "(default: {dataset}/{split_dir} from config)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--modality",   default="rgb", choices=["rgb", "ir"],
                        help="Filter videos by modality keyword (default: rgb)")
    parser.add_argument("--min-gap",    type=float, default=5.0,
                        help="Minimum gap in MINUTES between clips from the "
                             "same video (default: 5)")
    parser.add_argument("--no-transform", action="store_true",
                        help="Skip ROI crop / perspective-correction")
    args = parser.parse_args()

    random.seed(args.seed)
    cfg = load_config(args.config)

    # ── Derive output-dir from config if not given ────────────────────
    if args.output_dir is None:
        dataset_dir = cfg.drive.dataset
        if not dataset_dir:
            print("❌ --output-dir not given and drive.dataset is empty in config")
            sys.exit(1)
        args.output_dir = str(Path(dataset_dir) / SPLIT_TO_DIR[args.split])
        print(f"📁 Output dir (auto): {args.output_dir}")

    min_gap_sec = args.min_gap * 60.0   # convert minutes → seconds

    # ── 1. Find source videos ─────────────────────────────────────────
    raw_dir = cfg.drive.raw_videos
    if not raw_dir or not Path(raw_dir).exists():
        print(f"❌ raw_videos path not found: {raw_dir}")
        sys.exit(1)

    all_videos = find_videos(raw_dir, modality=args.modality)
    if not all_videos:
        print(f"❌ No {args.modality.upper()} videos found in {raw_dir}")
        sys.exit(1)
    print(f"📹 Found {len(all_videos)} {args.modality.upper()} video(s) in {raw_dir}")

    # ── 2. ROI validation (required unless --no-transform) ────────────
    roi_path = cfg.drive.roi_file
    rois: dict = {}
    if not args.no_transform:
        video_names = [v.name for v in all_videos]
        ok, msgs = validate_roi_file(
            roi_path or "(not set in config)", video_names)
        for m in msgs:
            print(f"  {m}")
        if not ok:
            print()
            print("   → Fix the issues above.  To pick missing ROIs:")
            raw = cfg.drive.raw_videos or "(raw_videos path)"
            roi = roi_path or "(roi_file path)"
            print(f"     python tools/pick_rois.py pick \\")
            print(f"         --video-dir {raw} \\")
            print(f"         --roi-file  {roi}")
            print()
            print("   Or pass --no-transform to skip perspective-correction.")
            sys.exit(1)
        rois = load_roi_json(roi_path)

    # target_size from config (for perspective transform output)
    target_size = None
    if cfg.chamber.target_size:
        target_size = tuple(cfg.chamber.target_size)

    # ── 3. Probe videos (fps, frame count) ────────────────────────────
    video_info = probe_videos(all_videos)
    min_total_sec = args.duration + 10
    eligible = {v: info for v, info in video_info.items()
                if info["total_sec"] >= min_total_sec}
    if not eligible:
        print(f"❌ No videos long enough (need ≥ {min_total_sec:.0f}s)")
        sys.exit(1)
    print(f"   {len(eligible)}/{len(all_videos)} are long enough "
          f"(≥ {min_total_sec:.0f}s)")

    # ── 4. Build weighted sampling pool ───────────────────────────────
    pool_videos = list(eligible.keys())
    pool_weights = [eligible[v]["n_frames"] for v in pool_videos]
    total_frames = sum(pool_weights)
    print(f"   Total frames across pool: {total_frames:,}")
    for v in pool_videos:
        pct = eligible[v]["n_frames"] / total_frames * 100
        print(f"     {v.name}: {eligible[v]['n_frames']:>10,} frames "
              f"({pct:.1f}%)")

    # ── 5. Load existing intervals ────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_key  = SPLIT_TO_MANIFEST[args.split]
    manifest_path = getattr(cfg.drive, manifest_key, None)
    if not manifest_path:
        print(f"❌ Manifest path for '{args.split}' not set in config.")
        sys.exit(1)

    existing_intervals = load_all_sampled_intervals(cfg)
    print(f"ℹ️  {len(existing_intervals)} intervals already sampled "
          f"across all splits.")
    print(f"   Min gap between clips: {args.min_gap} min ({min_gap_sec:.0f}s)")

    # ── 5b. Load valid ranges (optional filtering) ────────────────
    vr_path = getattr(cfg.drive, "valid_ranges", None)
    valid_ranges_raw = load_valid_ranges(vr_path)
    parsed_ranges: dict = {}
    lookups: dict[str, TimeLookup] = {}

    if valid_ranges_raw:
        cal_path = getattr(cfg.drive, "time_calibration", None)
        if not cal_path or not Path(cal_path).exists():
            print("❌ valid_ranges.json exists but time_calibration.json "
                  "not found.")
            print("   Run `calibrate_time.py calibrate` first.")
            sys.exit(1)
        with open(cal_path) as f:
            calibration = json.load(f)
        tz_str = cfg.time.timezone
        tz = ZoneInfo(tz_str)
        for vname in valid_ranges_raw:
            if vname in calibration:
                lookups[vname] = TimeLookup.from_calibration(
                    calibration, vname, tz_str)
        parsed_ranges = _parse_ranges(valid_ranges_raw, tz)

        n_ranges = sum(len(v) for v in valid_ranges_raw.values())
        print(f"⏱️  Valid ranges loaded: {n_ranges} range(s) across "
              f"{len(valid_ranges_raw)} video(s)")

        # Restrict pool to videos that have valid ranges
        pool_videos = [v for v in pool_videos if v.name in parsed_ranges]
        pool_weights = [eligible[v]["n_frames"] for v in pool_videos]
        if not pool_videos:
            print("❌ No eligible videos have valid ranges defined.")
            sys.exit(1)
        print(f"   Pool narrowed to {len(pool_videos)} video(s)")

    # ── 6. Sample loop ────────────────────────────────────────────
    new_rows: list[dict] = []
    attempts    = 0
    max_attempts = args.n * 100

    # Pre-build transformers (one per video)
    transformers: dict[str, PerspectiveTransformer] = {}
    if rois:
        for v in pool_videos:
            corners = find_roi_for_video(rois, v.name)
            if corners:
                transformers[v.name] = PerspectiveTransformer(
                    corners, target_size)

    print(f"\n🎲 Sampling {args.n} clips × {args.duration}s "
          f"(seed={args.seed}) ...\n")

    while len(new_rows) < args.n and attempts < max_attempts:
        attempts += 1

        # Weighted random pick
        video = random.choices(pool_videos, weights=pool_weights, k=1)[0]
        info  = eligible[video]

        start_sec = random.uniform(5, info["total_sec"] - args.duration - 5)
        end_sec   = start_sec + args.duration

        if too_close(start_sec, end_sec, existing_intervals,
                     video.name, min_gap_sec):
            continue

        # Check valid ranges (if loaded)
        if parsed_ranges:
            fps = eligible[video]["fps"]
            if not in_valid_range(video.name, start_sec, end_sec,
                                 fps, lookups, parsed_ranges):
                continue

        # Build clip filename
        tfx_tag   = "_transformed" if rois else ""
        clip_name = f"{video.stem}_s{int(start_sec)}{tfx_tag}.mp4"
        clip_path = output_dir / clip_name

        tf = transformers.get(video.name)

        print(f"  [{len(new_rows)+1}/{args.n}] {clip_name} …",
              end=" ", flush=True)
        ok = extract_clip(str(video), start_sec, args.duration,
                          str(clip_path), transformer=tf)
        if not ok:
            print("failed")
            continue

        print("✅")
        row = {
            "Clip_Filename":       clip_name,
            "Original_Video_Path": str(video),
            "Start_Time":          f"{start_sec:.2f}",
            "Duration":            f"{args.duration:.1f}",
        }
        new_rows.append(row)
        existing_intervals.append((video.name, start_sec, end_sec))

    if len(new_rows) < args.n:
        print(f"\n⚠️  Only found {len(new_rows)}/{args.n} non-overlapping "
              f"clips after {attempts} attempts.")

    if new_rows:
        append_to_manifest(manifest_path, new_rows)
    else:
        print("No clips sampled.")


if __name__ == "__main__":
    main()
