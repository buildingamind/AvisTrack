#!/usr/bin/env python3
"""
tools/extract_frames.py
───────────────────────
Extract a fixed number of evenly-spaced frames from each sampled clip, then
prune near-duplicate frames batch-wide via perceptual hashing (dHash).  Output
PNGs land in ``01_Dataset_MOT_Format/<split>/images/<batch_id>/`` and an initial
frame manifest is written to ``02_Global_Metadata/frame_manifests/<split>/<batch_id>.csv``
with ``Triage_Status: pending`` for every kept frame.

The triage tool (``review_triage.py``) is the next step — it flips rows to
``approved`` / ``rejected`` and physically moves rejects into a sibling
``_rejected/`` so that ``images/<batch_id>/*.png`` (top level only) stays
"ready for CVAT".

Usage
-----
    python tools/extract_frames.py \\
        --config configs/VR/wave3_vr.yaml \\
        --split  train \\
        --frames-per-clip 5 \\
        --hash-threshold 5

Notes
-----
* dHash with hash_size=8 produces a 64-bit fingerprint; Hamming distance ≤ 5
  is the standard "near-duplicate" threshold.
* batch_id default = ``<chamber>_<wave>_<YYYY-MM-DD>_batch<NN>``; ``<NN>`` auto-
  increments by inspecting existing batch dirs under ``images/``.
"""

import argparse
import csv
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

def main():
    ap = argparse.ArgumentParser(description="Extract evenly-spaced frames from sampled clips with dHash dedup.")
    ap.add_argument("--config", required=True, help="AvisTrack YAML config")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--frames-per-clip", type=int, default=5)
    ap.add_argument("--hash-threshold", type=int, default=5,
                    help="Hamming distance ≤ this → duplicate (default: 5)")
    ap.add_argument("--chamber", default=None,
                    help="Chamber id for batch naming (default: derived from experiment)")
    ap.add_argument("--wave", default=None,
                    help="Wave tag for batch naming (default: derived from experiment)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset_root = cfg.drive.dataset
    metadata_root = cfg.drive.metadata
    if not dataset_root or not metadata_root:
        print("❌ drive.dataset and drive.metadata must be set in config")
        sys.exit(1)

    clips_dir = Path(dataset_root) / SPLIT_TO_DIR[args.split]
    images_root = Path(dataset_root) / SPLIT_TO_DIR[args.split] / "images"
    manifests_dir = Path(metadata_root) / "frame_manifests" / args.split

    # Derive chamber/wave from experiment name "Wave3_VR" → ("VR", "Wave3_VR")
    exp = cfg.experiment or "Unknown"
    chamber = args.chamber or (exp.split("_", 1)[1] if "_" in exp else exp)
    wave = args.wave or exp

    batch_id = derive_batch_id(images_root, chamber, wave)
    out_dir = images_root / batch_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / f"{batch_id}.csv"

    clips = find_clips(clips_dir)
    if not clips:
        print(f"❌ No .mp4 clips in {clips_dir}")
        sys.exit(1)
    print(f"🎞️  {len(clips)} clip(s) in {clips_dir}")
    print(f"📦 batch_id = {batch_id}")
    print(f"   → frames out:  {out_dir}")
    print(f"   → manifest out: {manifest_path}")

    rows: list[dict] = []
    kept_hashes: list[int] = []
    n_extracted = n_dedup = n_kept = 0

    for clip in clips:
        frames = extract_clip_frames(clip, args.frames_per_clip)
        n_extracted += len(frames)
        for frame_idx, frame, ts in frames:
            h = dhash(frame)
            is_dup = any(hamming(h, kh) <= args.hash_threshold for kh in kept_hashes)
            if is_dup:
                n_dedup += 1
                continue
            kept_hashes.append(h)
            n_kept += 1
            fname = f"{clip.stem}_f{frame_idx:06d}.png"
            cv2.imwrite(str(out_dir / fname), frame)
            rows.append({
                "Frame_Filename": fname,
                "Source_Clip": clip.name,
                "Original_Video_Path": str(clip.resolve()),
                "Frame_Idx": str(frame_idx),
                "Timestamp": f"{ts:.3f}",
                "Triage_Status": "pending",
            })

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Extracted {n_extracted} frames; dedup dropped {n_dedup}; kept {n_kept}.")
    print(f"\nNext step:")
    print(f"  python tools/review_triage.py --batch {batch_id} --split {args.split} --config {args.config}")


if __name__ == "__main__":
    main()
