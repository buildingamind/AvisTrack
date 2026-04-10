#!/usr/bin/env python3
"""
eval/infer_clips.py
───────────────────
Batch YOLO inference on all test clips.  Saves raw per-frame detections so
tracking methods can be compared without re-running the model each time.

Usage:
    python eval/infer_clips.py \
        --weights   "E:/.../aug_minimal/weights/best.pt" \
        --clips-dir "E:/Wave2/01_Dataset_MOT_Format/test_golden" \
        --output-dir "E:/Wave2/04_Evaluation/W2_Collective" \
        [--conf 0.05] [--imgsz 640] [--device 0]

Output per clip:
    <output-dir>/detections/<clip_name>/dets.csv
    Columns: frame_idx(0-based), x1, y1, x2, y2, conf
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def infer_clip(model: YOLO, img_dir: Path, conf: float, imgsz: int) -> list[dict]:
    """Run inference on sorted frames in img_dir. Returns list of det dicts."""
    frames = sorted(img_dir.glob("frame_*.png"))
    if not frames:
        frames = sorted(img_dir.glob("frame_*.jpg"))

    rows = []
    for idx, img_path in enumerate(frames):
        results = model.predict(
            str(img_path), conf=conf, imgsz=imgsz,
            verbose=False, device=model.device,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            rows.append({
                "frame_idx": idx,
                "x1": round(float(x1), 2), "y1": round(float(y1), 2),
                "x2": round(float(x2), 2), "y2": round(float(y2), 2),
                "conf": round(float(c), 4),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",     required=True)
    ap.add_argument("--clips-dir",   required=True)
    ap.add_argument("--output-dir",  required=True)
    ap.add_argument("--conf",   type=float, default=0.05,
                    help="Detection confidence threshold (low to catch all boxes)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    clips_root = Path(args.clips_dir) / "annotations"
    out_root   = Path(args.output_dir) / "detections"
    clips      = sorted(d for d in clips_root.iterdir() if d.is_dir())

    if not clips:
        print(f"No clips found in {clips_root}")
        sys.exit(1)

    # Normalise device: '0' → 'cuda:0', 'cpu' stays as-is
    device = args.device
    if device.isdigit():
        device = f"cuda:{device}"

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)
    model.to(device)

    print(f"Running inference on {len(clips)} clips  (conf={args.conf})\n")

    for clip_dir in clips:
        img_dir = clip_dir / "img1"
        if not img_dir.exists():
            print(f"  [skip] {clip_dir.name}  — no img1/")
            continue

        out_dir = out_root / clip_dir.name
        out_csv = out_dir / "dets.csv"
        if out_csv.exists():
            print(f"  [skip] {clip_dir.name}  — already done")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        n_frames = len(list(img_dir.glob("frame_*.png")))
        print(f"  {clip_dir.name}  ({n_frames} frames) ...", end=" ", flush=True)

        rows = infer_clip(model, img_dir, args.conf, args.imgsz)

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_idx","x1","y1","x2","y2","conf"])
            writer.writeheader()
            writer.writerows(rows)

        print(f"{len(rows)} detections saved")

    print(f"\nDone. Detections → {out_root}")


if __name__ == "__main__":
    main()
