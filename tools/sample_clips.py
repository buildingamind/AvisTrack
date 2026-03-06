#!/usr/bin/env python3
"""
tools/sample_clips.py
──────────────────────
Sample short clips from raw videos and append records to the wave's manifest
file.  The tool checks all three existing manifests (train / val / test) to
avoid time-overlap with previously sampled clips.

Usage
-----
    python tools/sample_clips.py \\
        --config  configs/wave3_collective.yaml \\
        --split   train \\
        --n       20 \\
        --duration 3 \\
        --output-dir /media/woodlab/104-A/Wave3/01_Dataset_MOT_Format/train

Options
-------
    --config     AvisTrack YAML config  (for drive paths)
    --split      Which split to add to: train | val | test
    --n          Number of new clips to sample
    --duration   Clip duration in seconds (default 3)
    --output-dir Where to write the .mp4 clip files
    --seed       Random seed for reproducibility
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import cv2

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))
from avistrack.config.loader import load_config


SPLIT_TO_MANIFEST = {
    "train": "train_manifest",
    "val":   "val_manifest",
    "test":  "test_manifest",
}


# ── Manifest helpers ──────────────────────────────────────────────────────

def load_manifest(path: str | None) -> list[dict]:
    """Return list of dicts from a CSV manifest, or [] if path is None/missing."""
    if not path or not Path(path).exists():
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_all_sampled_intervals(cfg) -> list[tuple[str, float, float]]:
    """
    Load all already-sampled intervals from all three manifests.
    Returns a list of (original_video_path, start_sec, end_sec).
    """
    intervals = []
    for key in SPLIT_TO_MANIFEST.values():
        path = getattr(cfg.drive, key.replace("_manifest", "") + "_manifest", None)
        # schema stores them as train_manifest, val_manifest, test_manifest
        path = getattr(cfg.drive, key, None)
        for row in load_manifest(path):
            src  = row.get("Original_Video_Path", "")
            try:
                start    = float(row.get("Start_Time", 0))
                duration = float(row.get("Duration", 3))
                intervals.append((src, start, start + duration))
            except ValueError:
                pass
    return intervals


def overlaps(new_start: float, new_end: float,
             existing: list[tuple[str, float, float]],
             video_path: str) -> bool:
    """Return True if [new_start, new_end) overlaps any existing interval for this video."""
    for src, s, e in existing:
        if Path(src).name != Path(video_path).name:
            continue
        # Check overlap
        if new_start < e and new_end > s:
            return True
    return False


def append_to_manifest(manifest_path: str, rows: list[dict]) -> None:
    path   = Path(manifest_path)
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Clip_Filename", "Original_Video_Path", "Start_Time", "Duration"]
        )
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Appended {len(rows)} rows → {manifest_path}")


# ── Clip extraction ───────────────────────────────────────────────────────

def extract_clip(
    video_path: str,
    start_sec: float,
    duration: float,
    output_path: str,
) -> bool:
    """Extract [start_sec, start_sec+duration] from video_path → output_path."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ Cannot open: {video_path}")
        return False

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_sec * fps)
    n_frames    = int(duration  * fps)

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    while written < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        written += 1

    cap.release()
    out.release()
    return written > 0


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sample clips from raw videos.")
    parser.add_argument("--config",     required=True)
    parser.add_argument("--split",      required=True, choices=["train", "val", "test"])
    parser.add_argument("--n",          type=int, required=True, help="Number of new clips")
    parser.add_argument("--duration",   type=float, default=3.0, help="Clip duration in seconds")
    parser.add_argument("--output-dir", required=True, help="Where to write .mp4 files")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    cfg = load_config(args.config)

    raw_dir = cfg.drive.raw_videos
    if not raw_dir or not Path(raw_dir).exists():
        print(f"❌ raw_videos path not found: {raw_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine target manifest path
    manifest_key  = SPLIT_TO_MANIFEST[args.split]
    manifest_path = getattr(cfg.drive, manifest_key, None)
    if not manifest_path:
        print(f"❌ Manifest path for '{args.split}' not set in config.")
        sys.exit(1)

    # Find all source videos
    video_exts = {".mkv", ".mp4", ".avi", ".mov"}
    all_videos = [
        p for p in Path(raw_dir).rglob("*")
        if p.suffix.lower() in video_exts
    ]
    if not all_videos:
        print(f"❌ No videos found in {raw_dir}")
        sys.exit(1)

    # Build existing intervals to avoid
    existing_intervals = load_all_sampled_intervals(cfg)
    print(f"ℹ️  {len(existing_intervals)} intervals already sampled across all splits.")

    new_rows    = []
    attempts    = 0
    max_attempts = args.n * 50   # give up after this many tries

    while len(new_rows) < args.n and attempts < max_attempts:
        attempts += 1
        video = random.choice(all_videos)

        cap   = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            cap.release()
            continue
        fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        total_sec = n_frames / fps
        if total_sec < args.duration + 10:
            continue

        start_sec = random.uniform(5, total_sec - args.duration - 5)
        end_sec   = start_sec + args.duration

        if overlaps(start_sec, end_sec, existing_intervals, str(video)):
            continue

        # Build clip filename
        split_tag   = args.split.upper()
        clip_name   = f"{video.stem}_{split_tag}_s{int(start_sec)}.mp4"
        clip_path   = output_dir / clip_name

        print(f"  [{len(new_rows)+1}/{args.n}] Extracting {clip_name} …", end=" ", flush=True)
        ok = extract_clip(str(video), start_sec, args.duration, str(clip_path))
        if not ok:
            print("failed")
            continue

        print("✅")
        row = {
            "Clip_Filename":        clip_name,
            "Original_Video_Path":  str(video),
            "Start_Time":           f"{start_sec:.2f}",
            "Duration":             f"{args.duration:.1f}",
        }
        new_rows.append(row)
        # Register immediately so subsequent iterations don't re-use the window
        existing_intervals.append((str(video), start_sec, end_sec))

    if len(new_rows) < args.n:
        print(f"⚠️  Only found {len(new_rows)}/{args.n} non-overlapping clips "
              f"after {attempts} attempts.")

    if new_rows:
        append_to_manifest(manifest_path, new_rows)
    else:
        print("No clips sampled.")


if __name__ == "__main__":
    main()
