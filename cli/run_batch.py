#!/usr/bin/env python3
import os; os.environ.setdefault("PYTHONUTF8", "1")
"""
cli/run_batch.py
────────────────
Full-data batch processing: scan all raw videos, run the configured tracker,
write results to the output directory.

  • Reads everything from the YAML config (no hardcoded paths)
  • Supports checkpoint / resume (skips already-processed files)
  • Per-video tqdm progress bar + log file

Usage
-----
    python -m cli.run_batch --config configs/w2_collective_prod.yaml

    # Re-run even if output already exists:
    python -m cli.run_batch --config configs/w2_collective_prod.yaml --force

    # Smoke test (first N frames per video):
    python -m cli.run_batch --config configs/w2_collective_prod.yaml --limit 500
"""

import argparse
import json
import logging
import multiprocessing
import os
import queue as _queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from avistrack.config.loader import load_config
from avistrack.core.frame_source import FrameSource
from avistrack.core.time_lookup import TimeLookup
from avistrack.core.transformer import PerspectiveTransformer

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov"}
MIN_VALID_SIZE   = 1024   # bytes – files smaller than this are treated as empty


# ── Helpers ───────────────────────────────────────────────────────────────

def _load_time_lookup(cfg, video_name: str) -> "TimeLookup | None":
    """
    Load a TimeLookup for *video_name* from cfg.drive.time_calibration.
    Returns None if missing/uncalibrated — unix_time column will be NaN.
    """
    cal_path = getattr(cfg.drive, "time_calibration", None)
    if not cal_path or not Path(cal_path).exists():
        return None
    try:
        with open(cal_path) as f:
            cal_data = json.load(f)
        tz_str = getattr(cfg.time, "timezone", "America/New_York") if hasattr(cfg, "time") else "America/New_York"
        return TimeLookup.from_calibration(cal_data, video_name, timezone_str=tz_str)
    except (KeyError, ValueError):
        return None


def _setup_log(log_path: Path) -> logging.Logger:
    log = logging.getLogger("run_batch")
    log.setLevel(logging.INFO)
    if not log.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        log.addHandler(fh)
    return log


# ── Single-video worker ───────────────────────────────────────────────────

def _process_video(args_tuple):
    """
    Process a single video. Returns (video_name, status, elapsed_min).
    """
    from tqdm import tqdm

    video_path, config_path, output_dir, force, limit = args_tuple
    video_path = Path(video_path)
    name       = video_path.name

    try:
        cfg = load_config(config_path)
        output_cfg = cfg.output if hasattr(cfg, "output") else {}
        output_format = (output_cfg.get("format", "mot")
                         if isinstance(output_cfg, dict)
                         else getattr(output_cfg, "format", "mot"))
        ext         = ".parquet" if output_format == "parquet" else ".txt"
        output_path = Path(output_dir) / (video_path.stem + ext)

        # ── Skip check ───────────────────────────────────────────────
        if not force and output_path.exists() and output_path.stat().st_size > MIN_VALID_SIZE:
            return name, "skipped", 0.0

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
                    roi_path, video_path.name, target_size=(ts[0], ts[1]),
                )
            except KeyError:
                pass

        # ── Process frames (batched inference + prefetch thread) ──────
        rows       = []
        t0         = time.perf_counter()
        batch_size = tracker.batch_size
        # prefetch queue holds decoded frames ahead of GPU work
        PREFETCH   = batch_size * 4
        frame_q: _queue.Queue = _queue.Queue(maxsize=PREFETCH)

        def _reader(src, transformer, limit, q):
            try:
                for fidx, frame in src:
                    if limit and fidx > limit:
                        break
                    if transformer:
                        frame = transformer.transform(frame)
                    q.put((fidx, frame))
            finally:
                q.put(None)  # sentinel

        def _flush_batch(buf_idxs, buf_frames):
            batch_dets = tracker.update_batch(buf_frames)
            for fidx, dets in zip(buf_idxs, batch_dets):
                for d in dets:
                    rows.append((fidx, d.track_id,
                                 d.x, d.y, d.w, d.h,
                                 d.confidence, 1, 1.0))

        with FrameSource.from_video(str(video_path)) as src:
            total = min(src.total_frames or 0, limit) if limit else (src.total_frames or 0)
            bar = tqdm(
                total=total,
                desc=name[:45],
                unit="fr",
                dynamic_ncols=True,
                leave=True,
            )
            reader = threading.Thread(
                target=_reader, args=(src, transformer, limit, frame_q), daemon=True
            )
            reader.start()

            buf_idxs:  list = []
            buf_frames: list = []
            while True:
                item = frame_q.get()
                if item is None:
                    break
                fidx, frame = item
                buf_idxs.append(fidx)
                buf_frames.append(frame)
                bar.update(1)
                if len(buf_frames) >= batch_size:
                    _flush_batch(buf_idxs, buf_frames)
                    buf_idxs, buf_frames = [], []

            if buf_frames:
                _flush_batch(buf_idxs, buf_frames)
            bar.close()
            reader.join()

        elapsed = (time.perf_counter() - t0) / 60

        # ── Interpolate + append unix_time ────────────────────────────
        df = tracker.flush_interpolation(rows)
        time_lookup = _load_time_lookup(cfg, name)
        df["unix_time"] = (
            df["frame"].map(time_lookup.frame_to_unix).astype("float64")
            if time_lookup else float("nan")
        )

        # ── Write output ──────────────────────────────────────────────
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_format == "parquet":
            df.to_parquet(output_path, engine="pyarrow", compression="zstd", index=False)
        else:
            df.drop(columns=["unix_time"], errors="ignore").to_csv(
                output_path, index=False, header=False, float_format="%.2f"
            )

        return name, "success", elapsed

    except Exception as e:
        return name, f"failed: {e}", 0.0


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch-process all raw videos.")
    parser.add_argument("--config",  required=True, help="Path to YAML config.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel processes (default 1). "
                             "Increase only if you have multiple GPUs.")
    parser.add_argument("--force",   action="store_true",
                        help="Re-process even if output file already exists.")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Stop each video after this many frames (smoke-test mode).")
    args = parser.parse_args()

    cfg = load_config(args.config)

    raw_dir    = cfg.drive.raw_videos
    output_dir = cfg.output.get("dir", "./outputs") if hasattr(cfg, "output") else "./outputs"
    try:
        import yaml
        with open(args.config) as f:
            raw_yaml = yaml.safe_load(f)
        output_dir = raw_yaml.get("output", {}).get("dir", "./outputs")
        if cfg.drive.root:
            output_dir = output_dir.replace("{root}", cfg.drive.root)
    except Exception:
        pass

    if not raw_dir or not Path(raw_dir).exists():
        print(f"ERROR: raw_videos path not found: {raw_dir}")
        sys.exit(1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Log file ──────────────────────────────────────────────────────
    log_path = Path(output_dir) / "batch_run.log"
    log      = _setup_log(log_path)

    # ── Find videos ───────────────────────────────────────────────────
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(Path(raw_dir).rglob(f"*{ext}"))
    videos = sorted(videos)

    if not videos:
        print(f"ERROR: No videos found under {raw_dir}")
        sys.exit(1)

    n = len(videos)
    header = (f"\n{'='*60}\n"
              f"AvisTrack batch run\n"
              f"   Config  : {args.config}\n"
              f"   Videos  : {n}\n"
              f"   Output  : {output_dir}\n"
              f"   Workers : {args.workers}\n"
              f"   Log     : {log_path}\n"
              f"{'='*60}")
    print(header)
    log.info(f"Batch start — {n} videos, output={output_dir}")

    task_args = [(str(v), args.config, output_dir, args.force, args.limit) for v in videos]
    results   = {"success": [], "skipped": [], "failed": []}

    from tqdm import tqdm as _tqdm
    overall = _tqdm(total=n, desc="Overall", unit="video", position=1, leave=True)

    with multiprocessing.Pool(processes=args.workers) as pool:
        for video_name, status, elapsed in pool.imap_unordered(_process_video, task_args):
            key = "failed" if status.startswith("failed") else status
            results[key].append(video_name)
            overall.update(1)
            if status == "success":
                overall.set_postfix_str(f"last: {video_name[:30]} ({elapsed:.1f}min)")
                log.info(f"SUCCESS  {elapsed:.1f}min  {video_name}")
            elif status == "skipped":
                log.info(f"SKIPPED  {video_name}")
            else:
                overall.write(f"  FAILED: {video_name}: {status}")
                log.error(f"FAILED   {video_name}: {status}")

    overall.close()

    summary = (f"\n{'='*60}\n"
               f"Success : {len(results['success'])}\n"
               f"Skipped : {len(results['skipped'])}\n"
               f"Failed  : {len(results['failed'])}")
    if results["failed"]:
        summary += "\n" + "\n".join(f"     {f}" for f in results["failed"])
    summary += f"\n{'='*60}"
    print(summary)
    log.info(f"Batch done — success={len(results['success'])}, "
             f"skipped={len(results['skipped'])}, failed={len(results['failed'])}")


if __name__ == "__main__":
    main()
