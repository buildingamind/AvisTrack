#!/usr/bin/env python3
import os; os.environ.setdefault("PYTHONUTF8", "1")
"""
cli/run_batch.py
────────────────
Full-data batch processing: scan all raw videos, run the configured tracker,
write results to the output directory.

Two input modes:

* **Legacy** (``--config tracker.yaml``) – existing single-file config
  drives video discovery, weights, and output dir. Untouched.
* **Workspace** (``--workspace-yaml --chamber-id --wave-id
  --experiment-name --tracker-config``) – chamber drive provides videos,
  workspace provides weights at ``models/{exp}/final/best.pt``, output
  dir is ``{batch-output-dir}/{batch-run-name}/`` and **must** receive a
  ``_meta.json`` lineage record before any video runs (refuses to start
  if one already exists, refuses to start if weights are missing).

Both modes:
  • Reads everything from the YAML config (no hardcoded paths)
  • Supports checkpoint / resume (skips already-processed files)
  • Per-video tqdm progress bar + log file

Usage
-----
    # Legacy mode
    python -m cli.run_batch --config configs/w2_collective_prod.yaml

    # Workspace mode — full lineage chain to recipe + experiment
    python -m cli.run_batch \\
        --workspace-yaml /media/ssd/avistrack/collective/workspace.yaml \\
        --chamber-id     collective_104A \\
        --wave-id        wave2 \\
        --experiment-name W2_collective_phase3 \\
        --tracker-config configs/tracker_yolo_offline.yaml \\
        --batch-output-dir /media/ssd/avistrack/collective/batch_outputs \\
        --batch-run-name  W2_104A_wave2_2026-05-01

    # Re-run even if output already exists (per-video skip override):
    ... --force

    # Smoke test (first N frames per video):
    ... --limit 500
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
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from avistrack.config.loader import load_config
from avistrack.core.frame_source import FrameSource
from avistrack.core.time_lookup import TimeLookup
from avistrack.core.transformer import PerspectiveTransformer
from avistrack import lineage as L

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov"}
MIN_VALID_SIZE   = 1024   # bytes – files smaller than this are treated as empty

# Name of the frozen tracker yaml the workspace path-resolver writes into
# the output dir. Workers load this concrete file (with chamber paths
# already substituted) so re-running on a different host with the same
# workspace + drive layout produces identical results.
FROZEN_TRACKER_FILENAME = "tracker_config.yaml"


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


# ── Workspace mode helpers (Step G) ───────────────────────────────────────

WORKSPACE_FLAGS = (
    "workspace_yaml", "sources_yaml", "chamber_id", "wave_id",
    "experiment_name", "tracker_config", "batch_output_dir", "batch_run_name",
)


def _is_workspace_args(args) -> bool:
    """True iff *any* workspace-mode flag is set on the parsed args."""
    return any(getattr(args, name, None) for name in WORKSPACE_FLAGS)


def _validate_workspace_args(args) -> None:
    """Enforce 'all-or-nothing' workspace flags + mutual exclusion with --config."""
    if args.config:
        raise SystemExit(
            "ERROR: --config cannot be combined with workspace-mode flags. "
            "Use --tracker-config in workspace mode."
        )
    required = {
        "--workspace-yaml":   args.workspace_yaml,
        "--chamber-id":       args.chamber_id,
        "--wave-id":          args.wave_id,
        "--experiment-name":  args.experiment_name,
        "--tracker-config":   args.tracker_config,
        "--batch-output-dir": args.batch_output_dir,
        "--batch-run-name":   args.batch_run_name,
    }
    missing = [name for name, val in required.items() if not val]
    if missing:
        raise SystemExit(
            f"ERROR: workspace mode requires {', '.join(missing)}.")


def _resolve_workspace_weights(workspace_root: Path, chamber_type: str,
                               experiment_name: str) -> Path:
    """``{workspace_root}/{chamber_type}/models/{exp}/final/best.pt``.

    Refuses to proceed if the file is missing — running batch inference
    against a non-existent experiment is always a user error.
    """
    weights = (Path(workspace_root) / chamber_type /
               "models" / experiment_name / "final" / "best.pt")
    if not weights.exists():
        raise SystemExit(
            f"ERROR: experiment {experiment_name!r} has no final weights at "
            f"{weights}. Did the training pipeline finish successfully?")
    return weights


def _freeze_tracker_config(
    tracker_config_yaml: Path,
    *,
    raw_videos: Path,
    roi_file:   Optional[Path],
    valid_ranges:     Optional[Path],
    time_calibration: Optional[Path],
    weights: Path,
    output_dir: Path,
    timezone:   str,
    target_size: Optional[list[int]],
    output_yaml_path: Path,
) -> Path:
    """Write a self-contained tracker yaml to ``output_yaml_path``.

    The frozen yaml has every drive path resolved against the chamber
    drive that is currently mounted, so the per-video worker subprocesses
    can simply ``load_config(frozen_yaml)`` without any workspace context.

    Returns the path written. Lifted out as a pure function so unit tests
    can verify the override semantics without spinning up a tracker.
    """
    import yaml  # local import — avoid module-load failure in cv2-less envs
    with open(tracker_config_yaml) as f:
        raw = yaml.safe_load(f) or {}

    drive = raw.setdefault("drive", {})
    drive["raw_videos"]       = str(raw_videos)
    drive["root"]             = str(raw_videos)  # legacy {root} expansion target
    if roi_file is not None:
        drive["roi_file"]     = str(roi_file)
    if valid_ranges is not None:
        drive["valid_ranges"] = str(valid_ranges)
    if time_calibration is not None:
        drive["time_calibration"] = str(time_calibration)

    model = raw.setdefault("model", {})
    model["weights"] = str(weights)

    chamber = raw.setdefault("chamber", {})
    if target_size is not None and "target_size" not in chamber:
        chamber["target_size"] = list(target_size)

    time_block = raw.setdefault("time", {})
    time_block.setdefault("timezone", timezone)

    output = raw.setdefault("output", {})
    output["dir"]    = str(output_dir)
    output.setdefault("format", "parquet")  # workspace mode defaults to parquet

    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml_path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
    return output_yaml_path


def _build_initial_batch_meta(
    *,
    batch_run_name:  str,
    experiment_name: str,
    chamber_type:    str,
    chamber_id:      str,
    wave_id:         str,
    drive_uuid:      str,
    weights:         Path,
    workspace_root:  Path,
    tracker_config:  Path,
    repo_root:       Optional[Path] = None,
) -> "L.BatchMeta":
    """Construct the BatchMeta written before any video runs."""
    return L.BatchMeta(
        batch_run_name=  batch_run_name,
        experiment_name= experiment_name,
        chamber_type=    chamber_type,
        chamber_id=      chamber_id,
        wave_id=         wave_id,
        drive_uuid=      drive_uuid,
        weights=         str(weights),
        workspace_root=  str(workspace_root),
        started_at=      L.now_iso(),
        git_sha=         L.git_sha(repo_root),
        git_dirty=       L.git_dirty(repo_root),
        tracker_config=  str(tracker_config) if tracker_config else None,
    )


def _resolve_workspace_inputs(args) -> dict:
    """
    Resolve every path workspace mode needs **before** any subprocess work.

    Returns a dict with keys:
      ctx, weights, videos, output_dir, frozen_tracker, batch_meta_path,
      batch_meta.

    Raises SystemExit on user errors (missing flags, missing weights,
    missing chamber drive, attempt to clobber an existing _meta.json).
    """
    _validate_workspace_args(args)

    workspace_yaml = Path(args.workspace_yaml)
    sources_yaml = (Path(args.sources_yaml) if args.sources_yaml
                    else workspace_yaml.with_name("sources.yaml"))
    if not sources_yaml.exists():
        raise SystemExit(f"ERROR: sources.yaml not found at {sources_yaml}.")

    from avistrack.workspace import load_context
    ctx = load_context(
        workspace_yaml=workspace_yaml,
        sources_yaml=sources_yaml,
        chamber_id=args.chamber_id,
        wave_id=args.wave_id,
        require_drive=True,
    )

    workspace_root = ctx.workspace_root
    weights = _resolve_workspace_weights(
        workspace_root, ctx.chamber_type, args.experiment_name)

    output_dir = Path(args.batch_output_dir) / args.batch_run_name
    batch_meta_path = output_dir / L.BATCH_META_FILENAME
    if batch_meta_path.exists() and not args.force_meta:
        raise SystemExit(
            f"ERROR: {batch_meta_path} already exists. Pick a different "
            f"--batch-run-name or pass --force-meta to overwrite a stale "
            f"meta after a deliberate restart.")

    videos = ctx.list_videos(modality=args.modality)
    if not videos:
        raise SystemExit(
            f"ERROR: no {args.modality.upper()} videos for chamber "
            f"{args.chamber_id}/{args.wave_id} under {ctx.wave_root}.")

    frozen_tracker = _freeze_tracker_config(
        tracker_config_yaml=Path(args.tracker_config),
        raw_videos=ctx.wave_root,
        roi_file=ctx.roi_file if ctx.metadata_dir.exists() else None,
        valid_ranges=ctx.valid_ranges_file if ctx.metadata_dir.exists() else None,
        time_calibration=ctx.time_calibration_file if ctx.metadata_dir.exists() else None,
        weights=weights,
        output_dir=output_dir,
        timezone=ctx.workspace.time.timezone,
        target_size=ctx.workspace.chamber.target_size,
        output_yaml_path=output_dir / FROZEN_TRACKER_FILENAME,
    )

    batch_meta = _build_initial_batch_meta(
        batch_run_name=  args.batch_run_name,
        experiment_name= args.experiment_name,
        chamber_type=    ctx.chamber_type,
        chamber_id=      ctx.chamber.chamber_id,
        wave_id=         ctx.wave.wave_id,
        drive_uuid=      ctx.chamber.drive_uuid,
        weights=         weights,
        workspace_root=  workspace_root,
        tracker_config=  frozen_tracker,
        repo_root=       Path(__file__).resolve().parent.parent,
    )

    return {
        "ctx":             ctx,
        "weights":         weights,
        "videos":          videos,
        "output_dir":      output_dir,
        "frozen_tracker":  frozen_tracker,
        "batch_meta_path": batch_meta_path,
        "batch_meta":      batch_meta,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def _run_pool(
    *,
    videos: list[Path],
    config_path: Path,
    output_dir: Path,
    workers: int,
    force: bool,
    limit: Optional[int],
    label: str,
) -> dict:
    """Execute the per-video tracker pool. Returns ``{success/skipped/failed: [...]}``.

    Lifted out of ``main()`` so legacy and workspace flows share one
    implementation. ``label`` only affects the progress banner.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "batch_run.log"
    log      = _setup_log(log_path)

    n = len(videos)
    print(f"\n{'='*60}\n"
          f"AvisTrack batch run [{label}]\n"
          f"   Config  : {config_path}\n"
          f"   Videos  : {n}\n"
          f"   Output  : {output_dir}\n"
          f"   Workers : {workers}\n"
          f"   Log     : {log_path}\n"
          f"{'='*60}")
    log.info(f"Batch start [{label}] — {n} videos, output={output_dir}")

    task_args = [(str(v), str(config_path), str(output_dir), force, limit) for v in videos]
    results   = {"success": [], "skipped": [], "failed": []}

    from tqdm import tqdm as _tqdm
    overall = _tqdm(total=n, desc="Overall", unit="video", position=1, leave=True)

    with multiprocessing.Pool(processes=workers) as pool:
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

    return results


def _run_legacy_batch(args) -> None:
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
        raise SystemExit(f"ERROR: raw_videos path not found: {raw_dir}")

    videos: list[Path] = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(Path(raw_dir).rglob(f"*{ext}"))
    videos = sorted(videos)
    if not videos:
        raise SystemExit(f"ERROR: No videos found under {raw_dir}")

    _run_pool(
        videos=videos,
        config_path=Path(args.config),
        output_dir=Path(output_dir),
        workers=args.workers, force=args.force, limit=args.limit,
        label="legacy",
    )


def _run_workspace_batch(args) -> None:
    inputs = _resolve_workspace_inputs(args)
    output_dir      = inputs["output_dir"]
    frozen_tracker  = inputs["frozen_tracker"]
    batch_meta      = inputs["batch_meta"]
    videos          = inputs["videos"]

    # _meta.json before any video runs — refuses to clobber unless --force-meta.
    L.write_batch_meta(output_dir, batch_meta, overwrite=args.force_meta)

    results = _run_pool(
        videos=videos,
        config_path=frozen_tracker,
        output_dir=output_dir,
        workers=args.workers, force=args.force, limit=args.limit,
        label=f"workspace · {args.experiment_name}",
    )

    L.finalize_batch_meta(
        output_dir,
        ended_at=L.now_iso(),
        n_videos=len(videos),
        n_succeeded=len(results["success"]),
        n_failed=len(results["failed"]),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch-process all raw videos.")
    # Legacy mode
    p.add_argument("--config",  default=None,
                   help="Path to legacy YAML config (legacy mode).")
    # Workspace mode
    p.add_argument("--workspace-yaml", default=None,
                   help="Path to {workspace_root}/{chamber_type}/workspace.yaml.")
    p.add_argument("--sources-yaml",   default=None,
                   help="Path to sources.yaml (default: sibling of workspace.yaml).")
    p.add_argument("--chamber-id",     default=None,
                   help="Chamber id from sources.yaml (workspace mode).")
    p.add_argument("--wave-id",        default=None,
                   help="Wave id from sources.yaml (workspace mode).")
    p.add_argument("--experiment-name", default=None,
                   help="Workspace experiment whose final/best.pt drives the run.")
    p.add_argument("--tracker-config",  default=None,
                   help="Tracker hyperparam yaml (workspace mode). The drive "
                        "section is overridden from chamber context.")
    p.add_argument("--batch-output-dir", default=None,
                   help="Parent dir for the batch run (workspace mode).")
    p.add_argument("--batch-run-name",   default=None,
                   help="Subdir under --batch-output-dir; collisions are refused.")
    p.add_argument("--modality", default="rgb", choices=["rgb", "ir"],
                   help="Filter source videos by modality (workspace mode).")
    p.add_argument("--force-meta", action="store_true",
                   help="Overwrite an existing _meta.json under "
                        "--batch-output-dir/--batch-run-name (workspace mode).")
    # Shared
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel processes (default 1).")
    p.add_argument("--force",   action="store_true",
                   help="Re-process even if a per-video output file already exists.")
    p.add_argument("--limit",   type=int, default=None,
                   help="Stop each video after this many frames (smoke test).")
    return p


def main():
    args = _build_arg_parser().parse_args()

    if _is_workspace_args(args):
        _run_workspace_batch(args)
    elif args.config:
        _run_legacy_batch(args)
    else:
        raise SystemExit(
            "ERROR: pass --config (legacy) or the workspace-mode flag set "
            "(--workspace-yaml + --chamber-id + --wave-id + "
            "--experiment-name + --tracker-config + --batch-output-dir + "
            "--batch-run-name).")


if __name__ == "__main__":
    main()
