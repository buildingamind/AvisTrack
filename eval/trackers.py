#!/usr/bin/env python3
"""
eval/trackers.py
────────────────
Four tracking methods, all using top-N (=9) selection.

Methods:
    top9_hungarian  — Top-9 + Hungarian IoU matching (simple baseline)
    top9_kalman     — Top-9 + Kalman velocity model + Hungarian matching
    top9_interp     — top9_hungarian + linear interpolation for track gaps
    bytetrack       — Ultralytics built-in ByteTrack (reruns inference)

Usage:
    python eval/trackers.py \
        --detections-dir "E:/Wave2/04_Evaluation/W2_Collective/detections" \
        --clips-dir      "E:/Wave2/01_Dataset_MOT_Format/test_golden" \
        --output-dir     "E:/Wave2/04_Evaluation/W2_Collective/tracks" \
        --weights        "E:/.../aug_minimal/weights/best.pt" \
        [--methods top9_hungarian top9_kalman top9_interp bytetrack] \
        [--n-tracks 9]

Output per method per clip:
    <output-dir>/<method>/<clip_name>/gt.txt  (MOT format, 1-based frames)
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

N_TRACKS = 9  # number of animals — always 9


# ── IoU helpers ───────────────────────────────────────────────────────────────

def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) x1y1x2y2 → (N,M) IoU matrix."""
    ax1, ay1, ax2, ay2 = a[:,0], a[:,1], a[:,2], a[:,3]
    bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]
    ix1 = np.maximum(ax1[:,None], bx1[None,:])
    iy1 = np.maximum(ay1[:,None], by1[None,:])
    ix2 = np.minimum(ax2[:,None], bx2[None,:])
    iy2 = np.minimum(ay2[:,None], by2[None,:])
    inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    union = area_a[:,None] + area_b[None,:] - inter
    return np.where(union > 0, inter/union, 0.0)


def _top_n(dets: np.ndarray, n: int) -> np.ndarray:
    """Keep top-n rows by confidence (col 4)."""
    if len(dets) <= n:
        return dets
    idx = np.argpartition(dets[:,4], -n)[-n:]
    return dets[idx]


# ── Kalman filter for one box ─────────────────────────────────────────────────

class KalmanBoxTracker:
    """Constant-velocity Kalman filter.  State: [cx,cy,w,h, vcx,vcy,vw,vh]."""

    _F = np.eye(8); _F[:4, 4:] = np.eye(4)
    _H = np.eye(4, 8)
    _Q = np.diag([1.,1.,10.,10., 0.1,0.1,1.,1.])
    _R = np.diag([1.,1.,10.,10.])

    def __init__(self, box: np.ndarray):
        cx = (box[0]+box[2])/2; cy = (box[1]+box[3])/2
        w  =  box[2]-box[0];    h  =  box[3]-box[1]
        self.x = np.array([cx,cy,w,h, 0.,0.,0.,0.])
        self.P = np.diag([10.,10.,20.,20., 100.,100.,100.,100.])
        self.age = 0            # frames since last update

    def predict(self) -> np.ndarray:
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self._Q
        self.age += 1
        return self._to_xyxy()

    def update(self, box: np.ndarray):
        cx = (box[0]+box[2])/2; cy = (box[1]+box[3])/2
        w  =  box[2]-box[0];    h  =  box[3]-box[1]
        z = np.array([cx,cy,w,h])
        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + self._R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self._H) @ self.P
        self.age = 0

    def _to_xyxy(self) -> np.ndarray:
        cx,cy,w,h = self.x[:4]
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2])


# ── Tracker classes ───────────────────────────────────────────────────────────

class Top9HungarianTracker:
    """Top-9 selection + Hungarian IoU matching per frame."""

    def __init__(self, n=N_TRACKS, min_iou=0.3):
        self.n = n; self.min_iou = min_iou
        self.tracks: dict[int, np.ndarray] = {}
        self.next_id = 1

    def reset(self):
        self.tracks = {}; self.next_id = 1

    def update(self, dets: np.ndarray) -> dict[int, np.ndarray]:
        dets = _top_n(dets, self.n)
        if len(dets) == 0:
            return {}

        det_boxes = dets[:, :4]

        if not self.tracks:
            for box in det_boxes:
                self.tracks[self.next_id] = box.copy()
                self.next_id += 1
            return dict(self.tracks)

        tids = list(self.tracks)
        t_boxes = np.stack([self.tracks[tid] for tid in tids])
        iou = _iou_matrix(t_boxes, det_boxes)           # (n_tracks, n_dets)
        rows, cols = linear_sum_assignment(-iou)

        assigned_dets = set()
        new_tracks = {}
        for r, c in zip(rows, cols):
            if iou[r, c] >= self.min_iou:
                tid = tids[r]
                new_tracks[tid] = det_boxes[c].copy()
                assigned_dets.add(c)

        for c in range(len(det_boxes)):
            if c not in assigned_dets:
                new_tracks[self.next_id] = det_boxes[c].copy()
                self.next_id += 1

        self.tracks = new_tracks
        return dict(self.tracks)


class Top9KalmanTracker:
    """Top-9 selection + Kalman prediction + Hungarian matching."""

    def __init__(self, n=N_TRACKS, min_iou=0.2, max_age=5):
        self.n = n; self.min_iou = min_iou; self.max_age = max_age
        self.kf: dict[int, KalmanBoxTracker] = {}
        self.next_id = 1

    def reset(self):
        self.kf = {}; self.next_id = 1

    def update(self, dets: np.ndarray) -> dict[int, np.ndarray]:
        dets = _top_n(dets, self.n)
        det_boxes = dets[:, :4] if len(dets) else np.zeros((0,4))

        # Predict existing tracks; cull stale ones
        preds = {}
        for tid, kf in list(self.kf.items()):
            pb = kf.predict()
            if kf.age > self.max_age:
                del self.kf[tid]
            else:
                preds[tid] = pb

        if not preds:
            for box in det_boxes:
                self.kf[self.next_id] = KalmanBoxTracker(box)
                self.next_id += 1
            return {tid: kf._to_xyxy() for tid, kf in self.kf.items()}

        if len(det_boxes) == 0:
            return {tid: pb for tid, pb in preds.items()}

        tids = list(preds)
        t_boxes = np.stack([preds[tid] for tid in tids])
        iou = _iou_matrix(t_boxes, det_boxes)
        rows, cols = linear_sum_assignment(-iou)

        assigned_dets = set()
        for r, c in zip(rows, cols):
            if iou[r, c] >= self.min_iou:
                self.kf[tids[r]].update(det_boxes[c])
                assigned_dets.add(c)

        for c in range(len(det_boxes)):
            if c not in assigned_dets:
                self.kf[self.next_id] = KalmanBoxTracker(det_boxes[c])
                self.next_id += 1

        return {tid: kf._to_xyxy() for tid, kf in self.kf.items()}


def _interpolate_tracks(raw: dict[int, dict[int, np.ndarray]],
                        max_gap: int = 10) -> dict[int, dict[int, np.ndarray]]:
    """
    Linear interpolation for missing frames.
    raw: {track_id: {frame_idx: box_xyxy}}
    Returns same structure with gaps filled.
    """
    result = {tid: dict(frames) for tid, frames in raw.items()}
    for tid, frames in result.items():
        f_sorted = sorted(frames)
        for i in range(len(f_sorted)-1):
            f0, f1 = f_sorted[i], f_sorted[i+1]
            gap = f1 - f0 - 1
            if 0 < gap <= max_gap:
                b0, b1 = frames[f0], frames[f1]
                for k in range(1, gap+1):
                    t = k / (gap+1)
                    frames[f0+k] = b0*(1-t) + b1*t
    return result


class Top9InterpTracker:
    """top9_hungarian + linear interpolation for short track gaps."""

    def __init__(self, n=N_TRACKS, min_iou=0.3, max_gap=10):
        self.n = n; self.min_iou = min_iou; self.max_gap = max_gap

    def run_clip(self, all_dets: dict[int, np.ndarray]) -> dict[int, dict[int, np.ndarray]]:
        """
        all_dets: {frame_idx: np.ndarray (M,5)}
        Returns {track_id: {frame_idx: box_xyxy}} after interpolation.
        """
        base = Top9HungarianTracker(self.n, self.min_iou)
        raw: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
        for frame in sorted(all_dets):
            result = base.update(all_dets[frame])
            for tid, box in result.items():
                raw[tid][frame] = box
        return _interpolate_tracks(dict(raw), self.max_gap)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_dets(csv_path: Path) -> dict[int, np.ndarray]:
    """Load detections CSV → {frame_idx: (M,5) array [x1,y1,x2,y2,conf]}."""
    by_frame: dict[int, list] = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            fi = int(row["frame_idx"])
            by_frame[fi].append([
                float(row["x1"]), float(row["y1"]),
                float(row["x2"]), float(row["y2"]),
                float(row["conf"]),
            ])
    return {f: np.array(v) for f, v in by_frame.items()}


def save_mot(tracks_by_frame: dict[int, list[tuple]], out_path: Path):
    """
    Save tracks as MOT format.
    tracks_by_frame: {frame_idx(0-based): [(track_id, x1,y1,x2,y2), ...]}
    MOT output uses 1-based frame index and (x,y,w,h) format.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for fi in sorted(tracks_by_frame):
        mot_frame = fi + 1   # 0-based → 1-based
        for tid, x1, y1, x2, y2 in tracks_by_frame[fi]:
            w = x2-x1; h = y2-y1
            lines.append(f"{mot_frame},{int(tid)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},-1,-1,-1\n")
    with open(out_path, "w") as f:
        f.writelines(lines)


def _dict_to_frames(tid_frame_box: dict[int, dict[int, np.ndarray]]) -> dict[int, list[tuple]]:
    """Convert {tid:{frame:box}} → {frame:[(tid,x1,y1,x2,y2),...]}."""
    out: dict[int, list] = defaultdict(list)
    for tid, frames in tid_frame_box.items():
        for fi, box in frames.items():
            out[fi].append((tid, *box.tolist()))
    return dict(out)


# ── Per-method runners ────────────────────────────────────────────────────────

def _run_custom(tracker_cls, all_dets, **kwargs):
    tracker = tracker_cls(**kwargs)
    by_frame: dict[int, list[tuple]] = {}
    for fi in sorted(all_dets):
        result = tracker.update(all_dets[fi])
        by_frame[fi] = [(tid, *box.tolist()) for tid, box in result.items()]
    return by_frame


def _run_interp(all_dets):
    t = Top9InterpTracker()
    tid_frame = t.run_clip(all_dets)
    return _dict_to_frames(tid_frame)


def _run_bytetrack(weights: str, clip_dir: Path) -> dict[int, list[tuple]]:
    """Run Ultralytics model.track() with ByteTrack, keep top-N most-seen tracks."""
    from ultralytics import YOLO
    model = YOLO(weights)
    img_dir = clip_dir / "img1"
    frames = sorted(img_dir.glob("frame_*.png"))

    raw_tracks: dict[int, list] = defaultdict(list)  # frame -> [(id,x1,y1,x2,y2)]
    model.predictor = None  # reset state

    for fi, img_path in enumerate(frames):
        res = model.track(str(img_path), tracker="bytetrack.yaml",
                          persist=True, verbose=False, conf=0.05)[0]
        if res.boxes.id is None:
            continue
        ids   = res.boxes.id.int().cpu().numpy()
        xyxy  = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        # Keep top-N by confidence if more than N detected
        if len(ids) > N_TRACKS:
            keep = np.argsort(confs)[::-1][:N_TRACKS]
            ids, xyxy = ids[keep], xyxy[keep]
        for tid, (x1,y1,x2,y2) in zip(ids, xyxy):
            raw_tracks[fi].append((int(tid), x1, y1, x2, y2))

    # Keep the N_TRACKS most-frequently seen track IDs (prune spurious)
    id_count: dict[int, int] = defaultdict(int)
    for dets in raw_tracks.values():
        for tid, *_ in dets:
            id_count[tid] += 1
    keep_ids = set(sorted(id_count, key=id_count.get, reverse=True)[:N_TRACKS])

    # Remap to 1..N
    id_map = {old: new+1 for new, old in enumerate(sorted(keep_ids))}
    out: dict[int, list[tuple]] = {}
    for fi, dets in raw_tracks.items():
        kept = [(id_map[tid], x1,y1,x2,y2) for tid,x1,y1,x2,y2 in dets if tid in keep_ids]
        if kept:
            out[fi] = kept
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_METHODS = ["top9_hungarian", "top9_kalman", "top9_interp", "bytetrack"]


def run_all(detections_dir: Path, clips_dir: Path, output_dir: Path,
            weights: str, methods: list[str]):
    clips = sorted(detections_dir.iterdir())
    print(f"Tracking {len(clips)} clips  ×  {len(methods)} methods\n")

    for clip_dir in clips:
        dets_csv = clip_dir / "dets.csv"
        if not dets_csv.exists():
            continue

        clip_name = clip_dir.name
        orig_clip_dir = clips_dir / "annotations" / clip_name
        print(f"  {clip_name}")

        all_dets = load_dets(dets_csv) if any(m != "bytetrack" for m in methods) else {}

        for method in methods:
            out_path = output_dir / method / clip_name / "gt.txt"
            if out_path.exists():
                print(f"    [{method}] skip (exists)")
                continue

            if method == "top9_hungarian":
                by_frame = _run_custom(Top9HungarianTracker, all_dets)
            elif method == "top9_kalman":
                by_frame = _run_custom(Top9KalmanTracker, all_dets)
            elif method == "top9_interp":
                by_frame = _run_interp(all_dets)
            elif method == "bytetrack":
                by_frame = _run_bytetrack(weights, orig_clip_dir)
            else:
                print(f"    [{method}] unknown — skipped")
                continue

            save_mot(by_frame, out_path)
            n_det = sum(len(v) for v in by_frame.values())
            print(f"    [{method}] {n_det} track-detections → {out_path.parent}")

    print(f"\nTracks → {output_dir}")


def main():
    global N_TRACKS
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections-dir", required=True)
    ap.add_argument("--clips-dir",      required=True)
    ap.add_argument("--output-dir",     required=True)
    ap.add_argument("--weights",        default=None,
                    help="Required only for bytetrack method")
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS,
                    choices=ALL_METHODS)
    ap.add_argument("--n-tracks", type=int, default=N_TRACKS)
    args = ap.parse_args()

    if "bytetrack" in args.methods and not args.weights:
        print("--weights required for bytetrack method"); sys.exit(1)

    N_TRACKS = args.n_tracks

    run_all(
        Path(args.detections_dir),
        Path(args.clips_dir),
        Path(args.output_dir),
        args.weights or "",
        args.methods,
    )


if __name__ == "__main__":
    main()
