#!/usr/bin/env python3
"""
eval/select_champion.py
───────────────────────
Bake-off evaluator + champion selector (workspace-aware).

Evaluates every trained run under ``models/{exp}/phase{N}/`` on the dataset's
TEST split via ultralytics ``model.val()``, then:

  * writes ``phase{N}/test_eval.csv`` — one row per run, sorted by mAP50-95::

        run_name, mAP50-95, mAP50, P, R, F1, params, gpu_ms_per_img, gpu_fps, weights

  * selects the champion (``avistrack.champion.select_champion`` — highest
    mAP50-95, with a speed tiebreak within ``--tiebreak-pts``),
  * copies it to ``final/best.pt``, exports ``final/best.onnx``,
  * writes ``final/champion.meta.json`` (lineage-tagged),
  * updates ``meta.json`` (``final_weights``) and ``models/index.csv``.

This reproduces the pre-w4 model-selection convention. The training half is
``train/run_train.py``; this is the post-training eval+select half.

Usage
-----
    python eval/select_champion.py \\
        --workspace-yaml F:\\BAM\\avistrack_workspace\\plus\\workspace.yaml \\
        --experiment-name pre-w5_plus_2026-07-13
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Windows consoles default to cp1252; the emoji in status prints below would
# raise UnicodeEncodeError. Make stdout/stderr utf-8 (replace on failure).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from avistrack import lineage as L                      # noqa: E402
from avistrack.config.loader import load_workspace      # noqa: E402
from avistrack.champion import select_champion          # noqa: E402

TEST_EVAL_FIELDS = ["run_name", "mAP50-95", "mAP50", "P", "R", "F1",
                    "params", "gpu_ms_per_img", "gpu_fps", "weights"]
METRIC_KEYS = ("run_name", "mAP50-95", "mAP50", "P", "R", "F1",
               "params", "gpu_ms_per_img", "gpu_fps")


def _resolve_workspace_yaml(workspace_yaml: str, workspace_root) -> Path:
    if "{workspace_root}" in workspace_yaml:
        if workspace_root is None:
            raise SystemExit(
                "workspace_yaml has '{workspace_root}'; pass --workspace-root.")
        workspace_yaml = workspace_yaml.replace("{workspace_root}", str(workspace_root))
    return Path(workspace_yaml).expanduser().resolve()


def evaluate_run(weights: Path, data_yaml: Path, split: str,
                 imgsz: int, batch: int, device) -> dict:
    """Ultralytics val for one weights file on ``split`` → metrics dict."""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    res = model.val(data=str(data_yaml), split=split, imgsz=imgsz,
                    batch=batch, device=device, save_json=False, verbose=False)
    box = getattr(res, "box", None)
    mp = float(getattr(box, "mp", 0.0)) if box else 0.0
    mr = float(getattr(box, "mr", 0.0)) if box else 0.0
    f1 = (2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0
    params = sum(p.numel() for p in model.model.parameters())
    speed = getattr(res, "speed", {}) or {}
    inf_ms = float(speed.get("inference", 0.0))
    fps = (1000.0 / inf_ms) if inf_ms > 0 else 0.0
    return {
        "mAP50-95": float(getattr(box, "map", 0.0)) if box else 0.0,
        "mAP50":    float(getattr(box, "map50", 0.0)) if box else 0.0,
        "P": mp, "R": mr, "F1": f1,
        "params": int(params),
        "gpu_ms_per_img": inf_ms,
        "gpu_fps": fps,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace-yaml", required=True)
    ap.add_argument("--workspace-root", default=None,
                    help="Resolves {workspace_root} in --workspace-yaml.")
    ap.add_argument("--experiment-name", required=True)
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=0)
    ap.add_argument("--metric", default="mAP50-95")
    ap.add_argument("--tiebreak-pts", type=float, default=0.01)
    ap.add_argument("--no-onnx", action="store_true",
                    help="Skip the ONNX export of the champion.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing final/ directory.")
    args = ap.parse_args()

    ws_path = _resolve_workspace_yaml(args.workspace_yaml, args.workspace_root)
    if not ws_path.exists():
        raise SystemExit(f"workspace yaml not found: {ws_path}")
    workspace = load_workspace(ws_path)
    models_root = Path(workspace.workspace.models)
    exp_dir = models_root / args.experiment_name
    if not (exp_dir / "meta.json").exists():
        raise SystemExit(f"meta.json not found at {exp_dir}. Train first?")
    meta = L.read_meta(exp_dir)

    dataset_dir = Path(workspace.workspace.dataset) / meta.dataset_name
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"dataset data.yaml not found: {data_yaml}")

    phase_dir = exp_dir / f"phase{args.phase}"
    if not phase_dir.is_dir():
        raise SystemExit(f"phase dir not found: {phase_dir}")

    run_dirs = sorted(p for p in phase_dir.iterdir()
                      if p.is_dir() and (p / "weights" / "best.pt").exists())
    if not run_dirs:
        raise SystemExit(f"no trained runs (weights/best.pt) under {phase_dir}")

    print(f"📋 Champion selection")
    print(f"   experiment : {args.experiment_name}")
    print(f"   dataset    : {meta.dataset_name}")
    print(f"   split      : {args.split}")
    print(f"   runs       : {len(run_dirs)} -> {[p.name for p in run_dirs]}\n")

    rows = []
    for rd in run_dirs:
        w = rd / "weights" / "best.pt"
        print(f"  [eval] {rd.name} ...")
        m = evaluate_run(w, data_yaml, args.split, args.imgsz, args.batch, args.device)
        m["run_name"] = rd.name
        m["weights"] = str(w.resolve())
        rows.append(m)

    rows.sort(key=lambda r: r["mAP50-95"], reverse=True)

    test_eval_csv = phase_dir / "test_eval.csv"
    with open(test_eval_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=TEST_EVAL_FIELDS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in TEST_EVAL_FIELDS})

    print(f"\n🏁 test_eval.csv ({args.split} split, by mAP50-95):")
    for i, r in enumerate(rows, 1):
        print(f"   {i}. {r['run_name']:8s}  mAP50-95={r['mAP50-95']:.4f}  "
              f"mAP50={r['mAP50']:.4f}  P={r['P']:.3f} R={r['R']:.3f}  "
              f"fps={r['gpu_fps']:.1f}")

    champ, tiebreak_applied = select_champion(rows, args.metric, args.tiebreak_pts)

    final_dir = exp_dir / "final"
    if final_dir.exists() and not args.force:
        raise SystemExit(f"{final_dir} already exists; pass --force to overwrite.")
    final_dir.mkdir(parents=True, exist_ok=True)
    champ_weights = Path(champ["weights"])
    shutil.copy2(champ_weights, final_dir / "best.pt")

    if not args.no_onnx:
        from ultralytics import YOLO
        print(f"\n📦 exporting champion {champ['run_name']} → ONNX ...")
        onnx_src = YOLO(str(champ_weights)).export(
            format="onnx", imgsz=args.imgsz, device=args.device)
        shutil.copy2(str(onnx_src), final_dir / "best.onnx")

    champ_meta = {
        "champion_run":     champ["run_name"],
        "selection_metric": args.metric,
        "tiebreak_pts":     args.tiebreak_pts,
        "tiebreak_applied": tiebreak_applied,
        "metrics":          {k: champ[k] for k in METRIC_KEYS},
        "source_weights":   str(champ_weights),
        "phase":            f"phase{args.phase}",
        "split":            args.split,
        "experiment_name":  meta.experiment_name,
        "dataset_name":     meta.dataset_name,
        "git_sha":          L.git_sha(REPO_ROOT),
        "git_dirty":        L.git_dirty(REPO_ROOT),
        "evaluated_at":     L.now_iso(),
    }
    (final_dir / "champion.meta.json").write_text(json.dumps(champ_meta, indent=2))

    L.update_meta(exp_dir, final_weights=str(final_dir / "best.pt"))
    L.append_index(models_root, L.read_meta(exp_dir),
                   final_weights=final_dir / "best.pt")

    print(f"\n👑 champion: {champ['run_name']}  "
          f"(mAP50-95={champ['mAP50-95']:.4f}, tiebreak_applied={tiebreak_applied})")
    print(f"   → {final_dir / 'best.pt'}")
    if not args.no_onnx:
        print(f"   → {final_dir / 'best.onnx'}")
    print(f"   → {final_dir / 'champion.meta.json'}")
    print(f"   → {test_eval_csv}")


if __name__ == "__main__":
    main()
