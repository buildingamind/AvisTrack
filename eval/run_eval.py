#!/usr/bin/env python3
"""
eval/run_eval.py
────────────────
Run multiple model weights on the test-set clips and output a side-by-side
performance report.

Usage
-----
    python eval/run_eval.py \\
        --config  configs/wave3_collective.yaml \\
        --weights /media/.../weights/v1.pt /media/.../weights/v2.pt \\
        --output  eval/reports/wave3_coll_$(date +%Y%m%d).csv

What it measures (per weight file)
-----------------------------------
    Precision   TP / (TP + FP)
    Recall      TP / (TP + FN)
    F1          harmonic mean
    ID-switches Number of times a track ID changes on the same animal
    Throughput  Average FPS on this machine

Ground-truth format
-------------------
For each test clip  <clip>.mp4  there must be a matching  <clip>.txt  in the
same folder, in MOT format (headerless):
    frame, id, x, y, w, h, conf, class, vis[, ext]

The Hungarian algorithm matches predicted IDs to GT IDs per frame.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent.parent))
from avistrack.config.loader import load_config
from avistrack.core.transformer import PerspectiveTransformer


# ── IoU helper ────────────────────────────────────────────────────────────

def _iou(a: list[float], b: list[float]) -> float:
    """IoU of two [x, y, w, h] boxes."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2]*a[3]) + (b[2]*b[3]) - inter
    return inter / union if union > 0 else 0.0


# ── Prediction runner ─────────────────────────────────────────────────────

def run_clip(clip_path: Path, weights: str, cfg) -> list[dict]:
    """
    Run a single weights file on a single clip.
    Returns a list of per-frame dicts: {frame, id, x, y, w, h, conf}
    """
    from ultralytics import YOLO
    from avistrack.backends.yolo.offline import YoloOfflineTracker

    # Temporarily override weights in a shallow copy of config
    class _Cfg:
        class model:
            backend = "yolo"
            mode    = "offline"
            weights = weights
            def get(self, k, d=None): return d
        class chamber:
            n_subjects = cfg.chamber.n_subjects
            def get(self, k, d=None): return d
        def get(self, k, d=None): return d

    mock_cfg       = _Cfg()
    mock_cfg.model.weights  = weights
    mock_cfg.chamber.n_subjects = cfg.chamber.n_subjects

    # Rebuild a proper config-like object
    import types
    c          = types.SimpleNamespace()
    c.model    = types.SimpleNamespace()
    c.model.backend     = "yolo"
    c.model.mode        = "offline"
    c.model.weights     = weights
    c.model.get         = lambda k, d=None: None
    c.chamber           = types.SimpleNamespace()
    c.chamber.n_subjects = cfg.chamber.n_subjects
    c.chamber.get        = lambda k, d=None: d
    c.get = lambda k, d=None: d

    tracker = YoloOfflineTracker(c)

    # Optional transformer
    transformer = None
    if cfg.drive.roi_file and Path(cfg.drive.roi_file).exists():
        try:
            transformer = PerspectiveTransformer.from_roi_file(
                cfg.drive.roi_file,
                clip_path.name,
                target_size=(640, 640),
            )
        except KeyError:
            pass   # no ROI for this clip – run without transform

    cap = cv2.VideoCapture(str(clip_path))
    rows = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if transformer:
            frame = transformer.transform(frame)
        dets = tracker.update(frame)
        for d in dets:
            rows.append({
                "frame": frame_idx,
                "id":    d.track_id,
                "x": d.x, "y": d.y, "w": d.w, "h": d.h,
                "conf": d.confidence,
            })
    cap.release()
    return rows


# ── Evaluation ────────────────────────────────────────────────────────────

def load_gt(gt_path: Path) -> dict[int, list[dict]]:
    """Load MOT-format GT file → {frame_idx: [{id, x, y, w, h}, ...]}"""
    gt: dict[int, list[dict]] = {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            gt.setdefault(frame, []).append({
                "id": int(parts[1]),
                "x": float(parts[2]),
                "y": float(parts[3]),
                "w": float(parts[4]),
                "h": float(parts[5]),
            })
    return gt


def evaluate(preds: list[dict], gt: dict[int, list[dict]], iou_thresh: float = 0.5) -> dict:
    """
    Per-frame Hungarian matching → aggregate TP/FP/FN and ID switches.
    """
    tp = fp = fn = id_switches = 0
    prev_match: dict[int, int] = {}   # gt_id → pred_id from previous frame

    all_frames = sorted(set(gt.keys()) | {r["frame"] for r in preds})
    pred_by_frame: dict[int, list[dict]] = {}
    for r in preds:
        pred_by_frame.setdefault(r["frame"], []).append(r)

    for frame in all_frames:
        gt_boxes   = gt.get(frame, [])
        pred_boxes = pred_by_frame.get(frame, [])

        if not gt_boxes and not pred_boxes:
            continue

        if not gt_boxes:
            fp += len(pred_boxes)
            continue

        if not pred_boxes:
            fn += len(gt_boxes)
            continue

        # Build cost matrix
        cost = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, p in enumerate(pred_boxes):
            for j, g in enumerate(gt_boxes):
                cost[i, j] = 1.0 - _iou(
                    [p["x"], p["y"], p["w"], p["h"]],
                    [g["x"], g["y"], g["w"], g["h"]],
                )

        r_ind, c_ind = linear_sum_assignment(cost)
        matched_gt   = set()
        matched_pred = set()

        for r, c in zip(r_ind, c_ind):
            if cost[r, c] <= (1.0 - iou_thresh):
                tp += 1
                matched_gt.add(c)
                matched_pred.add(r)
                # ID switch check
                gt_id   = gt_boxes[c]["id"]
                pred_id = pred_boxes[r]["id"]
                if gt_id in prev_match and prev_match[gt_id] != pred_id:
                    id_switches += 1
                prev_match[gt_id] = pred_id

        fp += len(pred_boxes) - len(matched_pred)
        fn += len(gt_boxes)   - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": round(precision, 4),
        "Recall":    round(recall,    4),
        "F1":        round(f1,        4),
        "ID_Switches": id_switches,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple YOLO weights on test clips.")
    parser.add_argument("--config",  required=True)
    parser.add_argument("--weights", nargs="+", required=True, help="One or more .pt paths.")
    parser.add_argument("--output",  default="eval/reports/eval_result.csv")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_dir = cfg.drive.dataset
    if not dataset_dir:
        print("❌ drive.dataset not set in config.")
        sys.exit(1)

    test_dir = Path(dataset_dir) / "test_golden"
    clips    = sorted(test_dir.glob("*.mp4"))

    if not clips:
        print(f"❌ No .mp4 clips found in {test_dir}")
        sys.exit(1)

    print(f"📋 {len(clips)} test clips  ×  {len(args.weights)} weight files\n")

    report_rows = []

    for weights_path in args.weights:
        weights_name = Path(weights_path).name
        all_preds: list[dict] = []
        all_gt:    dict[int, list[dict]] = {}
        offset = 0

        t0 = time.perf_counter()
        for clip in clips:
            gt_file = clip.with_suffix(".txt")
            if not gt_file.exists():
                print(f"  ⚠️  No GT file for {clip.name}, skipping.")
                continue

            preds = run_clip(clip, weights_path, cfg)
            gt    = load_gt(gt_file)

            # Offset frames so all clips merge into one timeline
            for r in preds:
                r["frame"] += offset
                all_preds.append(r)
            for frame, boxes in gt.items():
                all_gt[frame + offset] = boxes

            n_frames = max((r["frame"] for r in preds), default=0) - offset
            offset  += n_frames + 1

        elapsed = time.perf_counter() - t0
        total_frames = offset
        fps = total_frames / elapsed if elapsed > 0 else 0.0

        metrics = evaluate(all_preds, all_gt, iou_thresh=args.iou_threshold)
        metrics["Weights"]    = weights_name
        metrics["FPS"]        = round(fps, 1)
        metrics["N_clips"]    = len(clips)

        report_rows.append(metrics)

        print(f"  {weights_name}")
        print(f"    Precision={metrics['Precision']:.3f}  "
              f"Recall={metrics['Recall']:.3f}  "
              f"F1={metrics['F1']:.3f}  "
              f"ID-sw={metrics['ID_Switches']}  "
              f"FPS={metrics['FPS']}\n")

    # Write CSV report
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Weights", "Precision", "Recall", "F1",
                  "ID_Switches", "FPS", "TP", "FP", "FN", "N_clips"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"📊 Report saved → {out}")


if __name__ == "__main__":
    main()
