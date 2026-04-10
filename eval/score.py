#!/usr/bin/env python3
"""
eval/score.py
─────────────
Compute HOTA, IDF1, MOTA for all tracking methods against ground truth.

Usage:
    python eval/score.py \
        --tracks-dir "E:/Wave2/04_Evaluation/W2_Collective/tracks" \
        --gt-dir     "E:/Wave2/01_Dataset_MOT_Format/test_golden" \
        --output     "E:/Wave2/04_Evaluation/W2_Collective/reports/scores.json"

Output: JSON with per-clip and overall scores per method.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


# ── MOT I/O ───────────────────────────────────────────────────────────────────

def load_mot(txt_path: Path) -> dict[int, list[tuple]]:
    """
    Load MOT format file.
    Returns {frame(1-based): [(id, x1, y1, x2, y2), ...]}  (converted to xyxy)
    """
    by_frame: dict[int, list] = defaultdict(list)
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split(",")
            frame = int(p[0])
            tid   = int(p[1])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            by_frame[frame].append((tid, x, y, x+w, y+h))
    return dict(by_frame)


# ── IoU helpers ───────────────────────────────────────────────────────────────

def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax1,ay1,ax2,ay2 = a[:,0],a[:,1],a[:,2],a[:,3]
    bx1,by1,bx2,by2 = b[:,0],b[:,1],b[:,2],b[:,3]
    ix1 = np.maximum(ax1[:,None], bx1); iy1 = np.maximum(ay1[:,None], by1)
    ix2 = np.minimum(ax2[:,None], bx2); iy2 = np.minimum(ay2[:,None], by2)
    inter = np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)
    aa = (ax2-ax1)*(ay2-ay1); ab = (bx2-bx1)*(by2-by1)
    union = aa[:,None]+ab[None,:]-inter
    return np.where(union>0, inter/union, 0.0)


# ── HOTA (custom implementation) ─────────────────────────────────────────────

def compute_hota(gt: dict[int, list[tuple]], pred: dict[int, list[tuple]],
                 alphas=None) -> dict:
    """
    Compute HOTA averaged over IoU thresholds alpha ∈ [0.05 .. 0.95].
    gt / pred: {frame(1-based): [(id, x1,y1,x2,y2), ...]}
    """
    if alphas is None:
        alphas = np.arange(0.05, 0.955, 0.05)   # 19 thresholds

    all_frames = sorted(set(gt) | set(pred))
    det_a_list, ass_a_list, hota_list = [], [], []

    for alpha in alphas:
        tp = fp = fn = 0
        gt_pred_by_frame: dict[int, dict] = {}   # frame -> {gt_id: pred_id}
        pred_gt_by_frame: dict[int, dict] = {}   # frame -> {pred_id: gt_id}

        for frame in all_frames:
            g_dets = gt.get(frame,   [])
            p_dets = pred.get(frame, [])

            if not g_dets and not p_dets:
                continue
            if not g_dets:
                fp += len(p_dets); continue
            if not p_dets:
                fn += len(g_dets); continue

            g_boxes = np.array([[d[1],d[2],d[3],d[4]] for d in g_dets])
            p_boxes = np.array([[d[1],d[2],d[3],d[4]] for d in p_dets])
            g_ids   = [d[0] for d in g_dets]
            p_ids   = [d[0] for d in p_dets]

            iou = _iou_matrix(g_boxes, p_boxes)
            rows, cols = linear_sum_assignment(-iou)

            g2p = {}; p2g = {}
            m_g = set(); m_p = set()
            for r, c in zip(rows, cols):
                if iou[r, c] >= alpha:
                    g2p[g_ids[r]] = p_ids[c]
                    p2g[p_ids[c]] = g_ids[r]
                    m_g.add(r); m_p.add(c); tp += 1
            fp += len(p_dets) - len(m_p)
            fn += len(g_dets) - len(m_g)
            if g2p:
                gt_pred_by_frame[frame] = g2p
                pred_gt_by_frame[frame] = p2g

        det_a = tp / (tp+fp+fn) if (tp+fp+fn) > 0 else 0.0

        if tp == 0:
            det_a_list.append(det_a); ass_a_list.append(0.0); hota_list.append(0.0)
            continue

        # Association counts per unique (gt_id, pred_id) pair
        pair_tpa: dict[tuple, int] = defaultdict(int)
        pred_matches: dict[int, dict[int, int]] = defaultdict(dict)  # pred_id->{frame:gt_id}
        gt_matches:   dict[int, dict[int, int]] = defaultdict(dict)  # gt_id->{frame:pred_id}

        for frame, g2p in gt_pred_by_frame.items():
            for gid, pid in g2p.items():
                pair_tpa[(gid, pid)] += 1
                pred_matches[pid][frame] = gid
                gt_matches[gid][frame]   = pid

        ass_sum = 0.0
        for (gid, pid), tpa in pair_tpa.items():
            fpa = sum(1 for g in pred_matches[pid].values() if g != gid)
            fna = sum(1 for p in gt_matches[gid].values()   if p != pid)
            ass_sum += tpa * tpa / (tpa+fpa+fna)

        ass_a = ass_sum / tp
        det_a_list.append(det_a)
        ass_a_list.append(ass_a)
        hota_list.append((det_a * ass_a) ** 0.5)

    return {
        "HOTA": round(float(np.mean(hota_list)), 4),
        "DetA": round(float(np.mean(det_a_list)), 4),
        "AssA": round(float(np.mean(ass_a_list)), 4),
    }


# ── IDF1 / MOTA via motmetrics (fallback: manual) ────────────────────────────

def _try_motmetrics(gt: dict, pred: dict) -> dict | None:
    try:
        import motmetrics as mm
    except ImportError:
        return None

    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(gt) | set(pred))
    for frame in all_frames:
        g = gt.get(frame, [])
        p = pred.get(frame, [])
        g_ids  = [d[0] for d in g]
        p_ids  = [d[0] for d in p]
        if not g and not p:
            continue
        if g and p:
            g_boxes = np.array([[d[1],d[2],d[3],d[4]] for d in g])
            p_boxes = np.array([[d[1],d[2],d[3],d[4]] for d in p])
            dists = 1.0 - _iou_matrix(g_boxes, p_boxes)
        else:
            dists = np.empty((len(g_ids), len(p_ids)))
        acc.update(g_ids, p_ids, dists)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["idf1","mota","num_switches"], name="s")
    row = summary.iloc[0]
    return {
        "IDF1": round(float(row["idf1"]), 4),
        "MOTA": round(float(row["mota"]), 4),
        "IDSW": int(row["num_switches"]),
    }


def _manual_mota_idf1(gt: dict, pred: dict) -> dict:
    """Fallback if motmetrics not available."""
    tp = fp = fn = idsw = 0
    prev: dict[int, int] = {}
    all_frames = sorted(set(gt) | set(pred))
    for frame in all_frames:
        g = gt.get(frame, []);  p = pred.get(frame, [])
        if not g and not p: continue
        if not g: fp += len(p); continue
        if not p: fn += len(g); continue
        g_boxes = np.array([[d[1],d[2],d[3],d[4]] for d in g])
        p_boxes = np.array([[d[1],d[2],d[3],d[4]] for d in p])
        iou = _iou_matrix(g_boxes, p_boxes)
        rows, cols = linear_sum_assignment(-iou)
        m_g = set(); m_p = set()
        for r, c in zip(rows, cols):
            if iou[r, c] >= 0.5:
                gid = g[r][0]; pid = p[c][0]
                if gid in prev and prev[gid] != pid: idsw += 1
                prev[gid] = pid
                tp += 1; m_g.add(r); m_p.add(c)
        fp += len(p)-len(m_p); fn += len(g)-len(m_g)

    gt_count = sum(len(v) for v in gt.values())
    mota = 1-(fp+fn+idsw)/gt_count if gt_count else 0.0
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    idf1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return {"IDF1": round(idf1,4), "MOTA": round(mota,4), "IDSW": idsw}


def compute_all_metrics(gt: dict, pred: dict) -> dict:
    hota_scores = compute_hota(gt, pred)
    extra = _try_motmetrics(gt, pred) or _manual_mota_idf1(gt, pred)
    return {**hota_scores, **extra}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks-dir", required=True)
    ap.add_argument("--gt-dir",     required=True)
    ap.add_argument("--output",     required=True)
    args = ap.parse_args()

    tracks_root = Path(args.tracks_dir)
    gt_root     = Path(args.gt_dir) / "annotations"
    out_path    = Path(args.output)

    methods = sorted(d.name for d in tracks_root.iterdir() if d.is_dir())
    if not methods:
        print("No tracking methods found."); sys.exit(1)

    print(f"Methods: {methods}")

    report: dict[str, dict] = {}

    for method in methods:
        method_dir = tracks_root / method
        clips = sorted(d.name for d in method_dir.iterdir() if d.is_dir())
        print(f"\n  [{method}]  {len(clips)} clips")

        per_clip: dict[str, dict] = {}
        # Accumulated GT / pred across all clips for overall score
        all_gt: dict[int, list] = {}
        all_pred: dict[int, list] = {}
        offset = 0

        for clip_name in clips:
            pred_txt = method_dir / clip_name / "gt.txt"
            gt_txt   = gt_root / clip_name / "gt" / "gt.txt"
            if not pred_txt.exists() or not gt_txt.exists():
                print(f"    [skip] {clip_name}")
                continue

            gt   = load_mot(gt_txt)
            pred = load_mot(pred_txt)

            metrics = compute_all_metrics(gt, pred)
            per_clip[clip_name] = metrics
            print(f"    {clip_name[-30:]:30s}  "
                  f"HOTA={metrics['HOTA']:.3f}  IDF1={metrics['IDF1']:.3f}  "
                  f"MOTA={metrics['MOTA']:.3f}  IDSW={metrics['IDSW']}")

            # Merge into global timeline
            max_frame = max(max(gt, default=0), max(pred, default=0))
            for f, v in gt.items():
                all_gt[f+offset]   = v
            for f, v in pred.items():
                all_pred[f+offset] = v
            offset += max_frame + 1

        overall = compute_all_metrics(all_gt, all_pred) if all_gt else {}
        print(f"    {'OVERALL':30s}  "
              f"HOTA={overall.get('HOTA',0):.3f}  IDF1={overall.get('IDF1',0):.3f}  "
              f"MOTA={overall.get('MOTA',0):.3f}  IDSW={overall.get('IDSW',0)}")
        report[method] = {"overall": overall, "clips": per_clip}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nScores → {out_path}")


if __name__ == "__main__":
    main()
