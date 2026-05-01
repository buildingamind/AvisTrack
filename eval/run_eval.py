#!/usr/bin/env python3
"""
eval/run_eval.py
────────────────
Two-mode evaluator.

**Mode A** (legacy clip-MOT tracking eval) — unchanged from before Step E.
Runs multiple weights against per-clip MOT-format ground truth and
reports Precision / Recall / F1 / ID-switches / FPS::

    python eval/run_eval.py \\
        --config  configs/wave3_collective.yaml \\
        --weights /media/.../weights/v1.pt /media/.../weights/v2.pt \\
        --output  eval/reports/wave3_coll_$(date +%Y%m%d).csv

**Mode B** (workspace-aware YOLO val) — added in Step E. Reads the
experiment's ``meta.json`` to find the dataset, runs ultralytics
``model.val()`` against ``datasets/{name}/data.yaml``, aggregates by
clip via ``manifest.csv``, and writes lineage-tagged outputs to
``models/{exp}/eval/{dataset_name}/``::

    python eval/run_eval.py \\
        --workspace-yaml /media/wkspc/collective/workspace.yaml \\
        --experiment-name W2_collective_phase1
        # weights default to {exp}/final/best.pt
        # dataset_name defaults to whatever meta.json says

Modes are mutually exclusive — pick one.

Mode A internals
----------------
    Precision   TP / (TP + FP)
    Recall      TP / (TP + FN)
    F1          harmonic mean
    ID-switches Number of times a track ID changes on the same animal
    Throughput  Average FPS on this machine

Ground-truth format (mode A only)
---------------------------------
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

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from avistrack import lineage as L
from avistrack.config.loader import load_config, load_workspace


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
    import cv2
    from ultralytics import YOLO
    from avistrack.backends.yolo.offline import YoloOfflineTracker
    from avistrack.core.transformer import PerspectiveTransformer

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
    from scipy.optimize import linear_sum_assignment
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


# ── Mode A: legacy clip-MOT tracking eval ────────────────────────────────

def run_mode_a(args) -> int:
    cfg = load_config(args.config)
    dataset_dir = cfg.drive.dataset
    if not dataset_dir:
        print("❌ drive.dataset not set in config.")
        return 1

    test_dir = Path(dataset_dir) / "test_golden"
    clips    = sorted(test_dir.glob("*.mp4"))

    if not clips:
        print(f"❌ No .mp4 clips found in {test_dir}")
        return 1

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
    return 0


# ── Mode B: workspace-aware YOLO val ─────────────────────────────────────

def _resolve_workspace_yaml(workspace_yaml: str, workspace_root) -> Path:
    if "{workspace_root}" in workspace_yaml:
        if workspace_root is None:
            raise SystemExit(
                "workspace_yaml has '{workspace_root}'; pass --workspace-root."
            )
        workspace_yaml = workspace_yaml.replace("{workspace_root}", str(workspace_root))
    return Path(workspace_yaml).expanduser().resolve()


def aggregate_per_clip(manifest_csv: Path, split: str = "test") -> list[dict]:
    """Group manifest rows by ``clip_stem`` for the requested split.

    Returns one row per clip with frame counts and chamber/wave provenance.
    Real per-clip mAP would require ultralytics-side bookkeeping that
    isn't exposed cleanly today; this is the lineage-friendly proxy.
    """
    if not manifest_csv.exists():
        return []
    by_clip: dict[str, dict] = {}
    with open(manifest_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") != split:
                continue
            stem = row.get("clip_stem", "")
            if not stem:
                continue
            entry = by_clip.setdefault(stem, {
                "clip_stem":  stem,
                "chamber_id": row.get("chamber_id", ""),
                "wave_id":    row.get("wave_id", ""),
                "split":      split,
                "n_frames":   0,
            })
            entry["n_frames"] += 1
    return sorted(by_clip.values(), key=lambda r: r["clip_stem"])


def run_mode_b(args) -> int:
    import yaml as _yaml

    workspace_yaml = _resolve_workspace_yaml(
        args.workspace_yaml, args.workspace_root
    )
    if not workspace_yaml.exists():
        raise SystemExit(f"workspace yaml not found: {workspace_yaml}")
    workspace = load_workspace(workspace_yaml)

    models_root = Path(workspace.workspace.models)
    exp_dir = models_root / args.experiment_name
    if not (exp_dir / "meta.json").exists():
        raise SystemExit(
            f"meta.json not found at {exp_dir}. Run training first or check "
            f"--experiment-name."
        )
    meta = L.read_meta(exp_dir)

    dataset_name = args.dataset_name or meta.dataset_name
    datasets_root = Path(workspace.workspace.dataset)
    dataset_dir = datasets_root / dataset_name
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"dataset data.yaml not found: {data_yaml}")

    if args.weights:
        if len(args.weights) != 1:
            raise SystemExit(
                "mode B accepts at most one --weights value. "
                "(Pass none to default to {exp}/final/best.pt.)"
            )
        weights = Path(args.weights[0]).expanduser().resolve()
    else:
        weights = exp_dir / "final" / "best.pt"
        if not weights.exists():
            raise SystemExit(
                f"default weights {weights} not found. Either finish "
                f"training or pass --weights."
            )

    out_dir = exp_dir / "eval" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📋 Workspace eval")
    print(f"   experiment   : {args.experiment_name}")
    print(f"   dataset      : {dataset_name}")
    print(f"   data.yaml    : {data_yaml}")
    print(f"   weights      : {weights}")
    print(f"   output       : {out_dir}")
    print()

    # Lazy import: ultralytics is heavy and not needed for mode A or for
    # tests that only exercise the path-resolution / aggregation helpers.
    from ultralytics import YOLO

    model = YOLO(str(weights))
    results = model.val(
        data=str(data_yaml), split=args.split,
        imgsz=args.imgsz, batch=args.batch, device=args.device,
        save_json=False, verbose=True,
    )

    # ultralytics 8.x: results.box.{map,map50,mp,mr}
    box = getattr(results, "box", None)
    summary = {
        "experiment_name": args.experiment_name,
        "dataset_name":    dataset_name,
        "split":           args.split,
        "weights":         str(weights),
        "mAP50-95":        float(getattr(box, "map",   0.0)) if box else 0.0,
        "mAP50":           float(getattr(box, "map50", 0.0)) if box else 0.0,
        "precision":       float(getattr(box, "mp",    0.0)) if box else 0.0,
        "recall":          float(getattr(box, "mr",    0.0)) if box else 0.0,
    }
    p, r = summary["precision"], summary["recall"]
    summary["F1"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    summary_csv = out_dir / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    per_clip_csv = out_dir / "per_clip_results.csv"
    rows = aggregate_per_clip(dataset_dir / "manifest.csv", split=args.split)
    if rows:
        with open(per_clip_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    eval_cfg = out_dir / "eval_config.yaml"
    eval_cfg.write_text(_yaml.safe_dump({
        "experiment_name": args.experiment_name,
        "dataset_name":    dataset_name,
        "split":           args.split,
        "weights":         str(weights),
        "git_sha":         L.git_sha(),
        "git_dirty":       L.git_dirty(),
        "started_at":      L.now_iso(),
        "data_yaml":       str(data_yaml),
    }, sort_keys=False))

    print(f"\n✅ Eval done.")
    print(f"   {summary_csv}")
    if rows:
        print(f"   {per_clip_csv}  ({len(rows)} clips)")
    print(f"   {eval_cfg}")
    return 0


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO weights — legacy clip-MOT (mode A) "
                    "or workspace-aware YOLO val (mode B).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Mode A
    parser.add_argument("--config", default=None,
                        help="[mode A] Legacy AvisTrack config yaml.")
    parser.add_argument("--weights", nargs="+", default=None,
                        help="[mode A] One or more .pt paths. "
                             "[mode B] Optional override (default: {exp}/final/best.pt).")
    parser.add_argument("--output", default="eval/reports/eval_result.csv",
                        help="[mode A] CSV report path.")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="[mode A] IoU threshold for matching.")
    # Mode B
    parser.add_argument("--workspace-yaml", default=None,
                        help="[mode B] Path (or {workspace_root} placeholder) to workspace.yaml.")
    parser.add_argument("--workspace-root", default=None,
                        help="[mode B] Resolves {workspace_root} in --workspace-yaml.")
    parser.add_argument("--experiment-name", default=None,
                        help="[mode B] Experiment to evaluate.")
    parser.add_argument("--dataset-name", default=None,
                        help="[mode B] Override the dataset (defaults to meta.json).")
    parser.add_argument("--split", default="test",
                        help="[mode B] Dataset split to evaluate (default: test).")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="[mode B] Image size for ultralytics val.")
    parser.add_argument("--batch", type=int, default=16,
                        help="[mode B] Batch size for ultralytics val.")
    parser.add_argument("--device", default=0,
                        help="[mode B] Device passed to ultralytics val.")
    args = parser.parse_args()

    mode_a = bool(args.config)
    mode_b = bool(args.workspace_yaml or args.experiment_name)
    if mode_a and mode_b:
        parser.error("--config (mode A) is mutually exclusive with "
                     "--workspace-yaml/--experiment-name (mode B).")
    if not mode_a and not mode_b:
        parser.error("Pass either --config (mode A) or --workspace-yaml + "
                     "--experiment-name (mode B).")

    if mode_a:
        if not args.weights:
            parser.error("mode A requires --weights.")
        sys.exit(run_mode_a(args))
    else:
        if not (args.workspace_yaml and args.experiment_name):
            parser.error("mode B requires --workspace-yaml and --experiment-name.")
        sys.exit(run_mode_b(args))


if __name__ == "__main__":
    main()
