#!/usr/bin/env python3
"""
tools/curate_frames.py
──────────────────────
Phase 0 data curation: deduplicate frames from MOT-format clips and export
a YOLO-format dataset.

Subcommands
-----------
preview   Read label files only (no image decode).
          Supports multiple --clips-dir sources with optional --labels.
          Shows per-source IoU distribution + combined threshold simulation
          table so you can pick IOU_THRESHOLD before touching any images.

export    Read pre-extracted PNG frames, skip near-duplicate frames, and
          write a YOLO images/labels dataset with a train/val split.
          Original annotation folders are never modified.

Input format (MOT directory layout)
-------------------------------------
Multi-clip (W2):
    <clips-dir>/
      <clip_name>/
        gt/gt.txt          MOT: frame,id,x,y,w,h,conf,class,vis
        img1/frame_XXXXXX.png

Single-clip (W3):
    <clips-dir>/
      gt/gt.txt
      img1/frame_XXXXXX.png

Frame index: gt.txt is 1-based; filenames are 0-based.
  gt.txt frame N  →  img1/frame_{N-1:06d}.png

Filter method (export)
-----------------------
IoU (default, IOU_THRESHOLD=0.92):
  Keep frame when min IoU across shared tracks vs last kept frame < threshold.
  IoU=1.0 → box identical; IoU=0.0 → no overlap.
  Skip only when ALL animals have >=threshold overlap → nearly static frame.

Displacement (fallback):
  Keep frame when max center displacement >= threshold px.

Usage
-----
  # Preview — W2 and W3 separately + combined table
  python tools/curate_frames.py preview \\
      --clips-dir "E:/Wave2/01_Dataset_MOT_Format/train/annotations" \\
                  "E:/Wave3/01_Dataset_MOT_Format/train/clip_merge_train" \\
      --labels W2 W3

  # Export W2 only (IoU method, threshold=0.92)
  python tools/curate_frames.py export \\
      --clips-dir "E:/Wave2/01_Dataset_MOT_Format/train/annotations" \\
      --output-dir "E:/AvisTrack_datasets/yolo_W2_only_iou092" \\
      --iou-threshold 0.92 --val-split 0.15 --seed 42

  # Export W3 only
  python tools/curate_frames.py export \\
      --clips-dir "E:/Wave3/01_Dataset_MOT_Format/train/clip_merge_train" \\
      --output-dir "E:/AvisTrack_datasets/yolo_W3_only_iou092" \\
      --iou-threshold 0.92 --val-split 0.15 --seed 42
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np


# ── MOT helpers ───────────────────────────────────────────────────────────

def load_mot_labels(gt_txt: Path) -> dict[int, list[dict]]:
    """Load MOT gt.txt → {frame_idx (1-based): [{id,x,y,w,h}, ...]}"""
    frames: dict[int, list[dict]] = {}
    with open(gt_txt) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            frames.setdefault(frame, []).append({
                "id": int(parts[1]),
                "x":  float(parts[2]),
                "y":  float(parts[3]),
                "w":  float(parts[4]),
                "h":  float(parts[5]),
            })
    return frames


# ── IoU metric ────────────────────────────────────────────────────────────

def _box_iou(b1: dict, b2: dict) -> float:
    """IoU between two MOT boxes (x,y = top-left corner, pixels)."""
    ix = max(0.0, min(b1["x"] + b1["w"], b2["x"] + b2["w"]) - max(b1["x"], b2["x"]))
    iy = max(0.0, min(b1["y"] + b1["h"], b2["y"] + b2["h"]) - max(b1["y"], b2["y"]))
    inter = ix * iy
    union = b1["w"] * b1["h"] + b2["w"] * b2["h"] - inter
    return inter / union if union > 0 else 0.0


def compute_frame_min_ious(mot: dict[int, list[dict]]) -> list[float]:
    """
    Min IoU across shared tracks for each consecutive frame pair.
    Low value  → something moved a lot (likely to keep).
    High value → scene nearly static (likely to skip).
    """
    sorted_frames = sorted(mot.keys())
    min_ious: list[float] = []
    prev_boxes: dict[int, dict] = {}

    for frame_idx in sorted_frames:
        curr_boxes = {b["id"]: b for b in mot[frame_idx]}
        if prev_boxes:
            shared = set(prev_boxes) & set(curr_boxes)
            if shared:
                min_iou = min(_box_iou(prev_boxes[i], curr_boxes[i]) for i in shared)
                min_ious.append(float(min_iou))
        prev_boxes = curr_boxes

    return min_ious


def _should_keep_iou(
    frame_idx: int,
    mot: dict[int, list[dict]],
    prev_boxes: dict[int, dict],
    iou_threshold: float,
) -> tuple[bool, dict[int, dict]]:
    boxes = mot.get(frame_idx, [])
    if not boxes:
        return False, prev_boxes
    curr_boxes = {b["id"]: b for b in boxes}
    if not prev_boxes:
        return True, curr_boxes
    shared = set(prev_boxes) & set(curr_boxes)
    if not shared:
        return True, curr_boxes
    min_iou = min(_box_iou(prev_boxes[i], curr_boxes[i]) for i in shared)
    if min_iou < iou_threshold:
        return True, curr_boxes
    return False, prev_boxes


# ── displacement metric (fallback) ────────────────────────────────────────

def _should_keep_displacement(
    frame_idx: int,
    mot: dict[int, list[dict]],
    prev_centers: dict[int, tuple[float, float]],
    threshold: float,
) -> tuple[bool, dict[int, tuple[float, float]]]:
    boxes = mot.get(frame_idx, [])
    if not boxes:
        return False, prev_centers
    curr_centers = {b["id"]: (b["x"] + b["w"] / 2, b["y"] + b["h"] / 2) for b in boxes}
    if not prev_centers:
        return True, curr_centers
    shared = set(prev_centers) & set(curr_centers)
    if not shared:
        return True, curr_centers
    max_disp = max(
        np.hypot(curr_centers[i][0] - prev_centers[i][0],
                 curr_centers[i][1] - prev_centers[i][1])
        for i in shared
    )
    if max_disp >= threshold:
        return True, curr_centers
    return False, prev_centers


# ── clip discovery ────────────────────────────────────────────────────────

def _find_mot_clips(clips_dir: Path) -> list[tuple[Path, Path, str]]:
    """
    Return (img1_dir, gt_txt, clip_name) triples.
    Auto-detects single-clip vs multi-clip layout.
    """
    single_gt   = clips_dir / "gt" / "gt.txt"
    single_img1 = clips_dir / "img1"
    if single_gt.exists() and single_img1.is_dir():
        return [(single_img1, single_gt, clips_dir.name)]

    clips: list[tuple[Path, Path, str]] = []
    for clip_dir in sorted(clips_dir.iterdir()):
        if not clip_dir.is_dir():
            continue
        gt_txt   = clip_dir / "gt" / "gt.txt"
        img1_dir = clip_dir / "img1"
        if gt_txt.exists() and img1_dir.is_dir():
            clips.append((img1_dir, gt_txt, clip_dir.name))
    return clips


# ── preview ───────────────────────────────────────────────────────────────

IOU_SIM_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
RECOMMENDED_IOU    = 0.92


def _collect_source_stats(clips_dir: Path, label: str) -> dict:
    """
    Process one source directory. Returns per-clip rows + aggregate arrays.
    """
    clips = _find_mot_clips(clips_dir)
    if not clips:
        print(f"  ⚠  No MOT clips found in {clips_dir}")
        return {}

    clip_rows    = []
    all_min_ious = []

    for img1_dir, gt_txt, clip_name in clips:
        mot      = load_mot_labels(gt_txt)
        min_ious = compute_frame_min_ious(mot)
        n_imgs   = len(list(img1_dir.glob("frame_*.png")))
        all_min_ious.extend(min_ious)
        clip_rows.append({
            "name":       clip_name,
            "n_labeled":  len(mot),
            "n_images":   n_imgs,
            "n_trans":    len(min_ious),
            "iou_median": float(np.median(min_ious)) if min_ious else float("nan"),
            "iou_p25":    float(np.percentile(min_ious, 25)) if min_ious else float("nan"),
            "iou_p75":    float(np.percentile(min_ious, 75)) if min_ious else float("nan"),
        })

    return {
        "label":       label,
        "clips_dir":   clips_dir,
        "n_clips":     len(clips),
        "clip_rows":   clip_rows,
        "min_ious":    all_min_ious,
    }


def _print_source_section(src: dict) -> None:
    label    = src["label"]
    rows     = src["clip_rows"]
    arr      = np.array(src["min_ious"]) if src["min_ious"] else np.array([])
    n_clips  = src["n_clips"]

    print(f"\n{'='*68}")
    print(f"  SOURCE: {label}  ({n_clips} clips)  ->  {src['clips_dir']}")
    print(f"{'='*68}")

    # Per-clip table
    print(f"  {'Clip':<35}  {'Labeled':>7}  {'Images':>7}  {'Trans':>6}  {'IoU med':>8}")
    print(f"  {'-'*35}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*8}")
    for r in rows:
        med_s = f"{r['iou_median']:.3f}" if not np.isnan(r['iou_median']) else "  n/a"
        print(f"  {r['name']:<35}  {r['n_labeled']:>7}  {r['n_images']:>7}  "
              f"{r['n_trans']:>6}  {med_s:>8}")

    total_labeled = sum(r["n_labeled"] for r in rows)
    total_images  = sum(r["n_images"]  for r in rows)
    total_trans   = sum(r["n_trans"]   for r in rows)
    print(f"  {'TOTAL':<35}  {total_labeled:>7}  {total_images:>7}  {total_trans:>6}")

    if len(arr) == 0:
        return

    # Distribution
    print(f"\n  IoU distribution (min IoU per consecutive frame pair, N={len(arr)}):")
    pcts = [("Min",None),("P5",5),("P25",25),("Median",50),("P75",75),("P95",95),("Max",None)]
    vals = []
    for lbl, p in pcts:
        v = arr.min() if lbl=="Min" else arr.max() if lbl=="Max" else np.percentile(arr, p)
        vals.append((lbl, v))
    line = "  " + "  ".join(f"{lbl}={v:.3f}" for lbl, v in vals)
    print(line)


def _print_combined_sim(sources: list[dict]) -> None:
    """
    Combined threshold simulation: per-source kept count + total + train/val estimate.
    Note: simulation uses consecutive-frame IoU (approximation of export behaviour).
    """
    labels   = [s["label"] for s in sources]
    n_clips  = [s["n_clips"] for s in sources]
    arrs     = [np.array(s["min_ious"]) if s["min_ious"] else np.array([])
                for s in sources]

    print(f"\n{'='*72}")
    print(f"  COMBINED THRESHOLD SIMULATION  (method=IoU, keep if min_iou < threshold)")
    print(f"  * First frame of each clip always kept (+{sum(n_clips)} clips)")
    print(f"  * Simulation uses consecutive-frame IoU -- export counts may differ slightly")
    print(f"{'='*72}")

    # Header
    src_cols = "  ".join(f"{lb:>8}" for lb in labels)
    print(f"  {'Threshold':>10}  {src_cols}  {'Total':>7}  {'Train~85%':>10}  {'Val~15%':>8}")
    print(f"  {'-'*10}  {'  '.join(['-'*8]*len(labels))}  {'-'*7}  {'-'*10}  {'-'*8}")

    for t in IOU_SIM_THRESHOLDS:
        per_src_kept = []
        for arr, nc in zip(arrs, n_clips):
            trans_kept = int(np.sum(arr < t)) if len(arr) > 0 else 0
            per_src_kept.append(trans_kept + nc)

        total     = sum(per_src_kept)
        train_est = int(total * 0.85)
        val_est   = total - train_est
        note      = "  <- recommended" if abs(t - RECOMMENDED_IOU) < 1e-9 else ""

        src_vals  = "  ".join(f"{k:>8}" for k in per_src_kept)
        print(f"  {t:>10.2f}  {src_vals}  {total:>7}  {train_est:>10}  {val_est:>8}{note}")

    print(f"\n  Target: 800–2000 training frames for yolo11n / yolo26n on this task.")
    print(f"  Once you pick a threshold, run `export` for each source separately,")
    print(f"  then optionally create a combined data.yaml pointing to both.")


def cmd_preview(args):
    clips_dirs = [Path(d) for d in args.clips_dir]
    labels     = args.labels if args.labels else [f"SRC{i}" for i in range(len(clips_dirs))]

    if len(labels) != len(clips_dirs):
        print("Error: --labels count must match --clips-dir count.")
        sys.exit(1)

    sources = []
    for clips_dir, label in zip(clips_dirs, labels):
        print(f"\nScanning {label}: {clips_dir} …")
        src = _collect_source_stats(clips_dir, label)
        if src:
            sources.append(src)

    if not sources:
        print("No data found.")
        sys.exit(1)

    for src in sources:
        _print_source_section(src)

    if len(sources) > 1:
        _print_combined_sim(sources)
    else:
        # Single source: still print simulation for that source
        _print_combined_sim(sources)


# ── export helpers ────────────────────────────────────────────────────────

def _mot_to_yolo(boxes: list[dict], img_w: int, img_h: int) -> list[str]:
    lines = []
    for b in boxes:
        cx = max(0.0, min(1.0, (b["x"] + b["w"] / 2) / img_w))
        cy = max(0.0, min(1.0, (b["y"] + b["h"] / 2) / img_h))
        nw = max(0.0, min(1.0, b["w"] / img_w))
        nh = max(0.0, min(1.0, b["h"] / img_h))
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def _export_split(
    split_clips:   list[tuple[Path, Path, str]],
    split:         str,
    output_dir:    Path,
    filter_method: str,
    threshold:     float,
    iou_threshold: float,
) -> dict:
    total = kept = skipped = 0
    clip_stats = []

    for img1_dir, gt_txt, clip_name in split_clips:
        mot = load_mot_labels(gt_txt)
        if not mot:
            print(f"  ⚠  Empty labels: {gt_txt}, skipping.")
            continue

        first_frame_idx = min(mot.keys())
        probe_path      = img1_dir / f"frame_{first_frame_idx - 1:06d}.png"
        probe           = cv2.imread(str(probe_path))
        if probe is None:
            print(f"  ⚠  Cannot read probe image: {probe_path}, skipping {clip_name}.")
            continue
        img_h, img_w = probe.shape[:2]

        prev_centers: dict[int, tuple[float, float]] = {}
        prev_boxes:   dict[int, dict]                = {}
        clip_kept = 0

        print(f"  [{split}] {clip_name} …", end=" ", flush=True)

        for frame_idx in sorted(mot.keys()):
            total += 1

            if filter_method == "iou":
                keep, prev_boxes = _should_keep_iou(
                    frame_idx, mot, prev_boxes, iou_threshold)
            else:
                keep, prev_centers = _should_keep_displacement(
                    frame_idx, mot, prev_centers, threshold)

            if not keep:
                skipped += 1
                continue

            src_path = img1_dir / f"frame_{frame_idx - 1:06d}.png"
            frame    = cv2.imread(str(src_path))
            if frame is None:
                print(f"\n  ⚠  Missing image: {src_path}")
                skipped += 1
                continue

            stem     = f"{clip_name}_f{frame_idx:06d}"
            img_path = output_dir / "images" / split / f"{stem}.jpg"
            lbl_path = output_dir / "labels" / split / f"{stem}.txt"

            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            lbl_path.write_text("\n".join(_mot_to_yolo(mot[frame_idx], img_w, img_h)))

            kept      += 1
            clip_kept += 1

        print(f"{clip_kept} frames kept")
        clip_stats.append({"clip": clip_name, "kept": clip_kept})

    print(f"  -> {split}: {kept}/{total} kept ({skipped} skipped)\n")
    return {"clips": clip_stats, "total": total, "kept": kept, "skipped": skipped}


# ── frame-level split (for single-clip sources) ───────────────────────────

def _export_frame_level(
    clips:         list[tuple[Path, Path, str]],
    output_dir:    Path,
    filter_method: str,
    threshold:     float,
    iou_threshold: float,
    val_split:     float,
    seed:          int,
) -> dict:
    """
    Filter all kept frames across all clips, then shuffle and split at frame
    level.  Used automatically when there are too few clips for a clip-level
    split (e.g. W3 single-clip layout).
    """
    print("  (single-clip source: using frame-level train/val split)\n")

    # Pass 1 — collect every kept frame as (src_path, label_lines, stem)
    kept_frames: list[tuple[Path, list[str], str]] = []
    total = skipped = 0

    for img1_dir, gt_txt, clip_name in clips:
        mot = load_mot_labels(gt_txt)
        if not mot:
            continue

        first_frame_idx = min(mot.keys())
        probe_path      = img1_dir / f"frame_{first_frame_idx - 1:06d}.png"
        probe           = cv2.imread(str(probe_path))
        if probe is None:
            print(f"  ⚠  Cannot read probe image: {probe_path}, skipping {clip_name}.")
            continue
        img_h, img_w = probe.shape[:2]

        prev_centers: dict[int, tuple[float, float]] = {}
        prev_boxes:   dict[int, dict]                = {}

        print(f"  [scan] {clip_name} …", end=" ", flush=True)
        clip_kept = 0

        for frame_idx in sorted(mot.keys()):
            total += 1
            if filter_method == "iou":
                keep, prev_boxes = _should_keep_iou(
                    frame_idx, mot, prev_boxes, iou_threshold)
            else:
                keep, prev_centers = _should_keep_displacement(
                    frame_idx, mot, prev_centers, threshold)

            if not keep:
                skipped += 1
                continue

            src_path = img1_dir / f"frame_{frame_idx - 1:06d}.png"
            labels   = _mot_to_yolo(mot[frame_idx], img_w, img_h)
            stem     = f"{clip_name}_f{frame_idx:06d}"
            kept_frames.append((src_path, labels, stem))
            clip_kept += 1

        print(f"{clip_kept} frames kept")

    print(f"  Total kept: {len(kept_frames)} / {total}  ({skipped} skipped)")

    # Pass 2 — shuffle + split
    rng = random.Random(seed)
    rng.shuffle(kept_frames)
    n_val      = max(1, int(len(kept_frames) * val_split))
    val_frames = kept_frames[:n_val]
    trn_frames = kept_frames[n_val:]

    print(f"  Frame-level split: train={len(trn_frames)}  val={n_val}\n")

    split_stats = {}
    for split, frames in [("train", trn_frames), ("val", val_frames)]:
        written = 0
        for src_path, labels, stem in frames:
            img = cv2.imread(str(src_path))
            if img is None:
                continue
            img_path = output_dir / "images" / split / f"{stem}.jpg"
            lbl_path = output_dir / "labels" / split / f"{stem}.txt"
            cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            lbl_path.write_text("\n".join(labels))
            written += 1
        print(f"  -> {split}: {written} frames written")
        split_stats[split] = {"kept": written, "total": total, "skipped": skipped}

    return split_stats


# ── export ────────────────────────────────────────────────────────────────

def cmd_export(args):
    clips_dir     = Path(args.clips_dir)
    output_dir    = Path(args.output_dir)
    filter_method = args.filter_method
    threshold     = args.threshold
    iou_threshold = args.iou_threshold
    val_split     = args.val_split
    seed          = args.seed

    random.seed(seed)

    clips = _find_mot_clips(clips_dir)
    if not clips:
        print(f"No MOT clips found in {clips_dir}")
        sys.exit(1)

    filter_desc = (f"iou < {iou_threshold}" if filter_method == "iou"
                   else f"displacement >= {threshold}px")
    print(f"Found {len(clips)} clip(s) | method={filter_method} ({filter_desc}) | "
          f"val_split={val_split} | seed={seed}\n")

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    report = {
        "filter_method": filter_method,
        "threshold":     threshold,
        "iou_threshold": iou_threshold,
        "val_split":     val_split,
        "seed":          seed,
        "clips_dir":     str(clips_dir),
        "splits":        {},
        "totals":        {},
    }

    # Use frame-level split when there are too few clips for clip-level split
    n_val_clips = max(1, int(len(clips) * val_split))
    use_frame_split = (n_val_clips >= len(clips))

    if use_frame_split:
        split_stats = _export_frame_level(
            clips, output_dir, filter_method, threshold, iou_threshold,
            val_split, seed)
        for split in ("train", "val"):
            report["splits"][split] = []
            report["totals"][split] = split_stats.get(split, {})
    else:
        random.shuffle(clips)
        val_clips   = clips[:n_val_clips]
        train_clips = clips[n_val_clips:]
        print(f"  Train clips: {len(train_clips)}")
        print(f"  Val   clips: {len(val_clips)}\n")

        for split, split_clips in [("train", train_clips), ("val", val_clips)]:
            stats = _export_split(
                split_clips, split, output_dir,
                filter_method, threshold, iou_threshold)
            report["splits"][split] = stats["clips"]
            report["totals"][split] = {
                "total":   stats["total"],
                "kept":    stats["kept"],
                "skipped": stats["skipped"],
            }

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\nnc: 1\n"
        f"names: ['chick']\n"
    )

    report_path = output_dir / "curation_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    total_kept = sum(report["totals"][s]["kept"] for s in ("train", "val"))
    print(f"Total frames exported : {total_kept}")
    print(f"data.yaml             : {data_yaml}")
    print(f"curation_report.json  : {report_path}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Curate MOT-format clips into a deduplicated YOLO dataset.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # preview
    p_prev = sub.add_parser(
        "preview",
        help="IoU distribution + threshold simulation (no image decode).")
    p_prev.add_argument("--clips-dir", required=True, nargs="+",
        help="One or more root directories containing MOT clips.")
    p_prev.add_argument("--labels", nargs="+",
        help="Short labels for each source (e.g. W2 W3). Must match --clips-dir count.")

    # export
    p_exp = sub.add_parser(
        "export",
        help="Read PNG frames, deduplicate by IoU/displacement, export YOLO dataset.")
    p_exp.add_argument("--clips-dir",      required=True,
        help="Root directory containing MOT clips (one source per export run).")
    p_exp.add_argument("--output-dir",     required=True,
        help="Root output directory (created fresh; original data never modified).")
    p_exp.add_argument("--filter-method",  choices=["iou", "displacement"],
        default="iou",
        help="Frame filter method (default: iou).")
    p_exp.add_argument("--iou-threshold",  type=float, default=0.92,
        help="[iou] Keep frame when min IoU < this value (default: 0.92).")
    p_exp.add_argument("--threshold",      type=float, default=4.0,
        help="[displacement] Min max-displacement px to keep (default: 4.0).")
    p_exp.add_argument("--val-split",      type=float, default=0.15,
        help="Fraction of clips held out for validation (default: 0.15).")
    p_exp.add_argument("--seed",           type=int,   default=42,
        help="Random seed for train/val split (default: 42).")

    args = parser.parse_args()
    if args.cmd == "preview":
        cmd_preview(args)
    else:
        cmd_export(args)


if __name__ == "__main__":
    main()
