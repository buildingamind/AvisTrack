#!/usr/bin/env python3
"""
cli/run_batch.py
────────────────
Full-data batch processing: scan all raw videos, run the configured tracker,
write MOT-format results to the output directory.

Replaces yolo-tracking-lab/W3_COLL/src/batch_run.py
  • Reads everything from the YAML config (no hardcoded paths)
  • Supports checkpoint / resume (skips already-processed files)
  • Multi-process parallel execution (configurable)

Usage
-----
    python cli/run_batch.py --config configs/wave3_collective.yaml

    # Override parallelism:
    python cli/run_batch.py --config configs/wave3_collective.yaml --workers 4

    # Re-run even if output already exists:
    python cli/run_batch.py --config configs/wave3_collective.yaml --force
"""

import argparse
import multiprocessing
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from avistrack.config.loader import load_config
from avistrack.core.frame_source import FrameSource
from avistrack.core.transformer import PerspectiveTransformer

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov"}
MIN_VALID_SIZE   = 1024   # bytes – files smaller than this are treated as empty


# ── Single-video worker (runs in subprocess) ──────────────────────────────

def _process_video(args_tuple):
    """
    Process a single video and write the MOT txt file.
    Returns (video_name: str, status: str).
    """
    video_path, config_path, output_dir, force = args_tuple
    video_path  = Path(video_path)
    output_path = Path(output_dir) / (video_path.stem + ".txt")
    pid         = os.getpid()
    name        = video_path.name

    # ── Skip check ───────────────────────────────────────────────────────
    if not force and output_path.exists() and output_path.stat().st_size > MIN_VALID_SIZE:
        return name, "skipped"

    try:
        cfg = load_config(config_path)

        # ── Build tracker ─────────────────────────────────────────────
        from avistrack.backends.yolo.offline import YoloOfflineTracker
        tracker = YoloOfflineTracker(cfg)

        # ── Optional perspective transform ────────────────────────────
        transformer = None
        roi_path = cfg.drive.roi_file
        if roi_path and Path(roi_path).exists():
            try:
                ts = cfg.chamber.get("target_size", [640, 640])
                transformer = PerspectiveTransformer.from_roi_file(
                    roi_path,
                    video_path.name,
                    target_size=(ts[0], ts[1]),
                )
            except KeyError:
                pass   # video has no ROI – run on raw frame

        # ── Process frames ────────────────────────────────────────────
        rows = []
        t0   = time.perf_counter()

        with FrameSource.from_video(str(video_path)) as src:
            total = src.total_frames or 0
            for frame_idx, frame in src:
                if frame_idx % 500 == 0 and total:
                    pct = frame_idx / total * 100
                    print(f"[PID:{pid}] {name}  {frame_idx}/{total} ({pct:.0f}%)", end="\r")

                if transformer:
                    frame = transformer.transform(frame)

                dets = tracker.update(frame)
                for d in dets:
                    rows.append((
                        frame_idx, d.track_id,
                        d.x, d.y, d.w, d.h,
                        d.confidence, 1, 1.0,
                    ))

        elapsed = (time.perf_counter() - t0) / 60
        print(f"\n[PID:{pid}] ✅ {name}  {frame_idx} frames  {elapsed:.1f} min")

        # ── Write interpolated output ─────────────────────────────────
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tracker.flush_interpolation(rows, str(output_path))

        return name, "success"

    except Exception as e:
        return name, f"failed: {e}"


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch-process all raw videos.")
    parser.add_argument("--config",  required=True, help="Path to YAML config.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel processes (default 4). "
                             "Reduce if VRAM runs out.")
    parser.add_argument("--force",   action="store_true",
                        help="Re-process even if output file already exists.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    raw_dir    = cfg.drive.raw_videos
    output_dir = cfg.output.get("dir", "./outputs") if hasattr(cfg, "output") else "./outputs"
    # Resolve output dir from config dict
    try:
        import yaml
        with open(args.config) as f:
            raw_yaml = yaml.safe_load(f)
        output_dir = raw_yaml.get("output", {}).get("dir", "./outputs")
        # Resolve {root}
        if cfg.drive.root:
            output_dir = output_dir.replace("{root}", cfg.drive.root)
    except Exception:
        pass

    if not raw_dir or not Path(raw_dir).exists():
        print(f"❌ raw_videos path not found: {raw_dir}")
        sys.exit(1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Find videos
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(Path(raw_dir).rglob(f"*{ext}"))
    videos = sorted(videos)

    if not videos:
        print(f"❌ No videos found under {raw_dir}")
        sys.exit(1)

    n = len(videos)
    print(f"\n{'='*60}")
    print(f"🚀 AvisTrack batch run")
    print(f"   Config  : {args.config}")
    print(f"   Videos  : {n}")
    print(f"   Output  : {output_dir}")
    print(f"   Workers : {args.workers}")
    print(f"{'='*60}\n")

    task_args = [(str(v), args.config, output_dir, args.force) for v in videos]

    results = {"success": [], "skipped": [], "failed": []}

    with multiprocessing.Pool(processes=args.workers) as pool:
        for video_name, status in pool.imap_unordered(_process_video, task_args):
            key = "failed" if status.startswith("failed") else status
            results[key].append(video_name)
            if status.startswith("failed"):
                print(f"  ❌ {video_name}: {status}")

    print(f"\n{'='*60}")
    print(f"✅ Success : {len(results['success'])}")
    print(f"⏭️  Skipped : {len(results['skipped'])}")
    print(f"❌ Failed  : {len(results['failed'])}")
    if results["failed"]:
        for f in results["failed"]:
            print(f"     {f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
