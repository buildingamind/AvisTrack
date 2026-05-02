#!/usr/bin/env python3
"""
tools/extract_frames.py
───────────────────────
Extract a fixed number of evenly-spaced frames from each sampled clip, then
prune near-duplicate frames batch-wide via perceptual hashing (dHash). Each
kept frame is written as PNG and registered in a triage manifest with
``Triage_Status: pending``; ``review_triage.py`` is the next step.

Two modes
---------
**Workspace mode** (recommended, multi-chamber architecture):
    python tools/extract_frames.py \\
        --workspace-yaml /path/to/workspace.yaml \\
        --chamber-id vr_105A --wave-id wave3 \\
        --target-frames 100              # exact final count (recommended)

    # or, legacy fixed-density mode:
    python tools/extract_frames.py ... --frames-per-clip 5 --hash-threshold 5

  Reads clips from   ``{workspace}/clips/{chamber}/{wave}/*.mp4``
  Writes PNGs to     ``{workspace}/frames/{chamber}/{wave}/{clip_stem}_f{idx:06d}.png``
  Writes manifest to ``{workspace}/manifests/triage/{batch_id}.csv``

**Legacy --config mode** (single-drive layout, kept for old waves):
    python tools/extract_frames.py \\
        --config configs/VR/wave3_vr.yaml \\
        --split  train --frames-per-clip 5 --hash-threshold 5

  PNGs at ``01_Dataset_MOT_Format/<split>/images/<batch_id>/``
  Manifest at ``02_Global_Metadata/frame_manifests/<split>/<batch_id>.csv``

Notes
-----
* dHash with hash_size=8 → 64-bit fingerprint; Hamming distance ≤ 5 is the
  "near-duplicate" threshold.
* batch_id = ``<chamber>_<wave>_<YYYY-MM-DD>_batch<NN>`` where ``<NN>``
  auto-increments by inspecting existing batches.
"""

import argparse
import csv
import math
import random
import sys
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from avistrack.config.loader import load_config


SPLIT_TO_DIR = {"train": "train", "val": "val_tuning", "test": "test_golden"}

CSV_FIELDS = [
    "Frame_Filename",
    "Source_Clip",
    "Original_Video_Path",
    "Frame_Idx",
    "Timestamp",
    "Triage_Status",
]


# ── dHash ──────────────────────────────────────────────────────────────────

def dhash(img: np.ndarray, hash_size: int = 8) -> int:
    """64-bit perceptual hash (returned as an int) for fast Hamming compare."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(b)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# ── Helpers ────────────────────────────────────────────────────────────────

def derive_batch_id(images_root: Path, chamber: str, wave: str) -> str:
    """Legacy mode: scan existing batch sub-dirs under images/ for next NN."""
    today = date.today().isoformat()
    prefix = f"{chamber}_{wave}_{today}_batch"
    n = 1
    if images_root.exists():
        existing = [p.name for p in images_root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
        nums = []
        for name in existing:
            try:
                nums.append(int(name[len(prefix):]))
            except ValueError:
                pass
        if nums:
            n = max(nums) + 1
    return f"{prefix}{n:02d}"


def derive_batch_id_from_manifests(manifests_dir: Path, chamber: str, wave: str) -> str:
    """Workspace mode: scan existing manifest CSVs for next NN of today."""
    today = date.today().isoformat()
    prefix = f"{chamber}_{wave}_{today}_batch"
    n = 1
    if manifests_dir.exists():
        nums = []
        for p in manifests_dir.iterdir():
            if not (p.is_file() and p.suffix == ".csv" and p.stem.startswith(prefix)):
                continue
            try:
                nums.append(int(p.stem[len(prefix):]))
            except ValueError:
                pass
        if nums:
            n = max(nums) + 1
    return f"{prefix}{n:02d}"


def find_clips(clips_dir: Path) -> list[Path]:
    return sorted(p for p in clips_dir.glob("*.mp4") if p.is_file())


def evenly_spaced_indices(total: int, k: int) -> list[int]:
    if total <= 0 or k <= 0:
        return []
    if k >= total:
        return list(range(total))
    return [int(round(i * (total - 1) / (k - 1))) if k > 1 else total // 2 for i in range(k)]


def extract_clip_frames(clip_path: Path, k: int) -> list[tuple[int, np.ndarray, float]]:
    """Returns [(frame_idx, frame, timestamp_sec_within_clip), ...]"""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        cap.release()
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames < 1:
        cap.release()
        return []
    indices = evenly_spaced_indices(n_frames, k)
    out = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        out.append((idx, frame, idx / fps))
    cap.release()
    return out


# ── Main ───────────────────────────────────────────────────────────────────

def _resolve_paths_workspace(args):
    """Workspace mode: returns (clips_dir, frames_base, manifest_path, batch_id, mode_info).
    Frames are written flat into frames_base; clip_stem in the filename anchors
    provenance, manifests/triage/{batch_id}.csv carries the full row metadata."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from avistrack.workspace import load_context

    workspace_yaml = Path(args.workspace_yaml)
    sources_yaml = Path(args.sources_yaml) if args.sources_yaml else workspace_yaml.with_name("sources.yaml")
    if not sources_yaml.exists():
        print(f"❌ sources.yaml not found: {sources_yaml}")
        sys.exit(1)

    ctx = load_context(
        workspace_yaml=workspace_yaml,
        sources_yaml=sources_yaml,
        chamber_id=args.chamber_id,
        wave_id=args.wave_id,
        require_drive=False,
    )

    clips_dir     = ctx.clip_dir                                     # workspace/clips/{ch}/{wv}/
    frames_base   = ctx.frame_dir                                    # workspace/frames/{ch}/{wv}/
    manifests_dir = ctx.manifests_root / "triage"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    batch_id = derive_batch_id_from_manifests(
        manifests_dir, ctx.chamber.chamber_id, ctx.wave.wave_id
    )
    manifest_path = manifests_dir / f"{batch_id}.csv"

    return clips_dir, frames_base, manifest_path, batch_id, {"mode": "workspace"}


def _resolve_paths_config(args):
    """Legacy mode: original split-based layout under cfg.drive.dataset."""
    cfg = load_config(args.config)
    dataset_root = cfg.drive.dataset
    metadata_root = cfg.drive.metadata
    if not dataset_root or not metadata_root:
        print("❌ drive.dataset and drive.metadata must be set in config")
        sys.exit(1)

    clips_dir     = Path(dataset_root) / SPLIT_TO_DIR[args.split]
    images_root   = Path(dataset_root) / SPLIT_TO_DIR[args.split] / "images"
    manifests_dir = Path(metadata_root) / "frame_manifests" / args.split

    exp = cfg.experiment or "Unknown"
    chamber = args.chamber or (exp.split("_", 1)[1] if "_" in exp else exp)
    wave    = args.wave or exp

    batch_id = derive_batch_id(images_root, chamber, wave)
    frames_base = images_root / batch_id           # legacy: flat dir per batch
    frames_base.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / f"{batch_id}.csv"

    return clips_dir, frames_base, manifest_path, batch_id, {"mode": "config"}


def main():
    ap = argparse.ArgumentParser(
        description="Extract evenly-spaced frames from sampled clips with dHash dedup.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Mode selectors (mutually exclusive groupings, validated below)
    ap.add_argument("--workspace-yaml", default=None,
                    help="Workspace mode: path to {workspace_root}/{chamber_type}/workspace.yaml")
    ap.add_argument("--sources-yaml", default=None,
                    help="Workspace mode: sources.yaml path (default: sibling of workspace-yaml)")
    ap.add_argument("--chamber-id", default=None,
                    help="Workspace mode: chamber id from sources.yaml")
    ap.add_argument("--wave-id", default=None,
                    help="Workspace mode: wave id from sources.yaml")
    ap.add_argument("--config", default=None, help="Legacy mode: AvisTrack single-drive YAML config")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"],
                    help="Legacy mode only")
    ap.add_argument("--chamber", default=None,
                    help="Legacy mode: chamber id for batch naming (default: derived from experiment)")
    ap.add_argument("--wave", default=None,
                    help="Legacy mode: wave tag for batch naming (default: derived from experiment)")
    # Common flags
    ap.add_argument("--frames-per-clip", type=int, default=5,
                    help="Evenly-spaced frames extracted from each clip before dedup. "
                         "Ignored when --target-frames is given (auto-picked with 3x headroom).")
    ap.add_argument("--hash-threshold", type=int, default=5,
                    help="Hamming distance ≤ this → duplicate (default: 5; 0 disables dedup)")
    ap.add_argument("--target-frames", type=int, default=None,
                    help="Final frame count after dedup. Tool extracts ~3x candidates "
                         "from clips, dedups, then random-samples down to N. "
                         "If post-dedup pool < N, all are kept and a warning is printed.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for --target-frames sampling")
    args = ap.parse_args()

    workspace_mode = bool(args.workspace_yaml or args.chamber_id or args.wave_id)
    if workspace_mode:
        if args.config:
            print("❌ --config cannot be combined with workspace-mode flags "
                  "(--workspace-yaml / --chamber-id / --wave-id)")
            sys.exit(2)
        missing = [name for name, val in (
            ("--workspace-yaml", args.workspace_yaml),
            ("--chamber-id",     args.chamber_id),
            ("--wave-id",        args.wave_id),
        ) if not val]
        if missing:
            print(f"❌ workspace mode requires {', '.join(missing)}")
            sys.exit(2)
        clips_dir, frames_base, manifest_path, batch_id, mode_info = _resolve_paths_workspace(args)
    else:
        if not args.config:
            print("❌ Either --workspace-yaml (workspace mode) or --config (legacy) is required")
            sys.exit(2)
        clips_dir, frames_base, manifest_path, batch_id, mode_info = _resolve_paths_config(args)

    clips = find_clips(clips_dir)
    if not clips:
        print(f"❌ No .mp4 clips in {clips_dir}")
        sys.exit(1)
    frames_base.mkdir(parents=True, exist_ok=True)

    target = args.target_frames
    if target is not None and target <= 0:
        print("❌ --target-frames must be positive")
        sys.exit(2)

    # When --target-frames is given, oversample 3x so dedup has headroom.
    if target is not None:
        effective_fpc = max(args.frames_per_clip,
                            math.ceil(target * 3 / max(len(clips), 1)))
    else:
        effective_fpc = args.frames_per_clip

    print(f"🎞️  {len(clips)} clip(s) in {clips_dir}")
    print(f"📦 batch_id = {batch_id}  (mode={mode_info['mode']})")
    print(f"   → frames out:   {frames_base}/")
    print(f"   → manifest out: {manifest_path}")
    if target is not None:
        print(f"   → target-frames={target}, frames-per-clip={effective_fpc} (auto)")
    else:
        print(f"   → frames-per-clip={effective_fpc}")

    # Phase 1: extract all candidates into memory
    candidates: list[dict] = []
    for clip in clips:
        for frame_idx, frame, ts in extract_clip_frames(clip, effective_fpc):
            candidates.append({
                "clip": clip,
                "frame_idx": frame_idx,
                "frame": frame,
                "ts": ts,
                "hash": dhash(frame),
            })
    n_extracted = len(candidates)

    # Phase 2: batch-wide dedup (first-seen wins). hash_threshold=0 disables.
    if args.hash_threshold > 0:
        kept: list[dict] = []
        kept_hashes: list[int] = []
        for c in candidates:
            if any(hamming(c["hash"], kh) <= args.hash_threshold for kh in kept_hashes):
                continue
            kept.append(c)
            kept_hashes.append(c["hash"])
    else:
        kept = list(candidates)
    n_dedup = n_extracted - len(kept)

    # Phase 3: trim to target (random sample, seeded)
    n_trimmed = 0
    if target is not None:
        if len(kept) > target:
            rng = random.Random(args.seed)
            picked = rng.sample(kept, target)
            picked.sort(key=lambda c: (c["clip"].name, c["frame_idx"]))
            n_trimmed = len(kept) - target
            kept = picked
        elif len(kept) < target:
            print(f"⚠ only {len(kept)} frames after dedup, asked for {target}. "
                  f"To get more, lower --hash-threshold (currently {args.hash_threshold}) "
                  f"or sample more clips.")
    n_kept = len(kept)

    # Phase 4: write PNGs + manifest
    rows: list[dict] = []
    for c in kept:
        fname = f"{c['clip'].stem}_f{c['frame_idx']:06d}.png"
        cv2.imwrite(str(frames_base / fname), c["frame"])
        rows.append({
            "Frame_Filename": fname,
            "Source_Clip": c["clip"].name,
            "Original_Video_Path": str(c["clip"].resolve()),
            "Frame_Idx": str(c["frame_idx"]),
            "Timestamp": f"{c['ts']:.3f}",
            "Triage_Status": "pending",
        })

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    if target is not None:
        print(f"\n✅ Extracted {n_extracted}; dedup dropped {n_dedup}; "
              f"trimmed-to-target {n_trimmed}; kept {n_kept}.")
    else:
        print(f"\n✅ Extracted {n_extracted}; dedup dropped {n_dedup}; kept {n_kept}.")
    print(f"\nNext step:")
    if mode_info["mode"] == "workspace":
        print(f"  python tools/review_triage.py \\")
        print(f"      --workspace-yaml {args.workspace_yaml} \\")
        print(f"      --chamber-id {args.chamber_id} --wave-id {args.wave_id} \\")
        print(f"      --batch {batch_id}")
    else:
        print(f"  python tools/review_triage.py --batch {batch_id} --split {args.split} --config {args.config}")


if __name__ == "__main__":
    main()
