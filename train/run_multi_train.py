#!/usr/bin/env python3
"""
train/run_multi_train.py
────────────────────────
Train multiple YOLO candidates back-to-back, evaluate each on the same test
set, score them with a configurable weighted formula, and copy the winner
into a ``champion/`` directory for downstream deployment.

Architecture
------------
This is a thin orchestrator around two existing tools:
  * ``train/run_train.py``   — handles the training loop (skip-completed,
                                multi-candidate) — we synthesize an experiment
                                YAML if needed and shell out to it.
  * ``eval/run_eval.py``      — runs each best.pt on golden test clips and
                                emits Precision / Recall / F1 / ID_Switches /
                                FPS per weights file.

Config (``configs/VR/multi_train.yaml``)
----------------------------------------
    output_root: /path/to/03_Model_Training            # per-candidate dirs land here
    data: /home/.../avistrack_training_cache/VR_combined_prod/dataset.yaml
    eval_config: configs/VR/wave3_vr.yaml              # for run_eval.py (test_golden + ROI)
    defaults:
      epochs: 200
      device: 0
      patience: 20
    candidates:
      - name: yolo11n_640
        model: yolo11n.pt
        imgsz: 640
        batch: 32
      - name: yolo11n_512
        model: yolo11n.pt
        imgsz: 512
        batch: 48
      - name: yolo11s_640
        model: yolo11s.pt
        imgsz: 640
        batch: 16
        epochs: 150
    score_weights:                  # weights for leaderboard scoring (sum need not = 1)
      F1: 0.5
      FPS: 0.3
      Recall: 0.2
      ID_Switches: -0.1            # negative weight = penalty

Usage
-----
    python train/run_multi_train.py \\
        --config configs/VR/multi_train.yaml

    # Train only, skip eval (faster for quick iteration)
    python train/run_multi_train.py \\
        --config configs/VR/multi_train.yaml --skip-eval

    # Eval only (re-score after editing weights, no retrain)
    python train/run_multi_train.py \\
        --config configs/VR/multi_train.yaml --skip-train
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


METRIC_FIELDS = ["F1", "Recall", "Precision", "FPS", "ID_Switches", "TP", "FP", "FN"]


def load_cfg(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k in ("output_root", "data", "candidates"):
        if k not in cfg:
            print(f"❌ multi_train config missing required field: {k}")
            sys.exit(1)
    return cfg


def write_experiment_yaml(cfg: dict, dest: Path) -> None:
    """Translate multi_train config → run_train.py compatible experiment YAML."""
    exp = {
        "output_root": cfg["output_root"],
        "data": cfg["data"],
        "defaults": cfg.get("defaults", {}),
        "runs": cfg["candidates"],
    }
    with open(dest, "w") as f:
        yaml.safe_dump(exp, f, sort_keys=False)


def candidate_weights_path(output_root: str, name: str) -> Path:
    return Path(output_root) / name / "weights" / "best.pt"


def normalize(values: list[float]) -> list[float]:
    """Min-max normalize to [0, 1] for fair weighting across heterogeneous metrics."""
    if not values:
        return values
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def compute_score(rows: list[dict], weights: dict) -> list[dict]:
    """Add 'Score' column. Negative weights mean lower-is-better metrics (e.g. ID_Switches)."""
    if not rows:
        return rows
    metric_arrays = {m: [float(r.get(m, 0)) for r in rows] for m in weights}
    norm = {m: normalize(arr) for m, arr in metric_arrays.items()}
    for i, r in enumerate(rows):
        score = 0.0
        for m, w in weights.items():
            v = norm[m][i]
            if w < 0:
                v = 1.0 - v  # invert so lower-is-better contributes positively
            score += abs(w) * v if w < 0 else w * v
        r["Score"] = round(score, 4)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Multi-candidate YOLO training + scoring orchestrator.")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-train even if best.pt exists (passed through to run_train).")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    output_root = Path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    # ── Train ────────────────────────────────────────────────────────
    if not args.skip_train:
        exp_yaml = output_root / "_synthesized_experiment.yaml"
        write_experiment_yaml(cfg, exp_yaml)
        cmd = [
            sys.executable, str(Path(__file__).parent / "run_train.py"),
            "--experiment", str(exp_yaml),
        ]
        if args.force:
            cmd.append("--force")
        print(f"🏋️  Training: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("⚠️  Some training runs failed (see above). Continuing to eval with what's available.")

    # ── Eval ─────────────────────────────────────────────────────────
    if args.skip_eval:
        print("⏩ Skipping eval (--skip-eval).")
        return

    eval_cfg = cfg.get("eval_config")
    if not eval_cfg:
        print("❌ No eval_config in multi_train YAML — cannot evaluate.")
        sys.exit(1)

    weights_paths = []
    for cand in cfg["candidates"]:
        wp = candidate_weights_path(cfg["output_root"], cand["name"])
        if wp.exists():
            weights_paths.append(str(wp))
        else:
            print(f"⚠️  best.pt missing for {cand['name']} — skipping in eval.")
    if not weights_paths:
        print("❌ No weights to evaluate.")
        sys.exit(1)

    raw_csv = output_root / "_eval_raw.csv"
    cmd = [
        sys.executable, str(Path(__file__).parent.parent / "eval" / "run_eval.py"),
        "--config", eval_cfg,
        "--output", str(raw_csv),
        "--weights", *weights_paths,
    ]
    print(f"📊 Eval: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("⚠️  Eval returned non-zero.")
        sys.exit(1)

    # ── Leaderboard ──────────────────────────────────────────────────
    with open(raw_csv) as f:
        rows = list(csv.DictReader(f))

    weights_map = cfg.get("score_weights") or {"F1": 0.5, "FPS": 0.3, "Recall": 0.2}
    rows = compute_score(rows, weights_map)
    rows.sort(key=lambda r: r.get("Score", 0), reverse=True)

    fields = ["Weights", "Score"] + [m for m in weights_map] + \
             [m for m in METRIC_FIELDS if m not in weights_map]
    leaderboard = output_root / "leaderboard.csv"
    with open(leaderboard, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in fields} for r in rows)

    print(f"\n🏆 Leaderboard:")
    for i, r in enumerate(rows, 1):
        print(f"   {i}. {r['Weights']:30s}  Score={r.get('Score', 0):.4f}  "
              f"F1={r.get('F1', '?')}  FPS={r.get('FPS', '?')}")
    print(f"\n   Saved: {leaderboard}")

    # ── Copy champion ────────────────────────────────────────────────
    if not rows:
        return
    champ_weights_name = rows[0]["Weights"]                      # e.g. best.pt
    champ_candidate = next(
        (c for c in cfg["candidates"]
         if candidate_weights_path(cfg["output_root"], c["name"]).name == champ_weights_name
         and candidate_weights_path(cfg["output_root"], c["name"]).exists()),
        None,
    )
    # Fall back: match by full path containing candidate name
    if champ_candidate is None:
        for c in cfg["candidates"]:
            wp = candidate_weights_path(cfg["output_root"], c["name"])
            if wp.exists() and wp.name == champ_weights_name:
                champ_candidate = c
                break
    if champ_candidate is None:
        # Final fallback: take the first candidate whose best.pt exists in score order
        for r in rows:
            for c in cfg["candidates"]:
                wp = candidate_weights_path(cfg["output_root"], c["name"])
                if wp.exists() and wp.name == r["Weights"]:
                    champ_candidate = c
                    break
            if champ_candidate:
                break
    if champ_candidate is None:
        print("⚠️  Could not resolve champion candidate — skipping copy.")
        return

    champ_dir = output_root / "champion"
    champ_dir.mkdir(exist_ok=True)
    src = candidate_weights_path(cfg["output_root"], champ_candidate["name"])
    shutil.copy2(src, champ_dir / "best.pt")
    meta = {
        "champion_candidate": champ_candidate["name"],
        "champion_score": rows[0].get("Score", 0),
        "selected_at": datetime.now().isoformat(timespec="seconds"),
        "leaderboard": rows,
        "score_weights": weights_map,
        "source_weights": str(src),
        "training_data": cfg["data"],
    }
    with open(champ_dir / "champion.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n👑 Champion: {champ_candidate['name']} → {champ_dir / 'best.pt'}")
    print(f"   Meta: {champ_dir / 'champion.meta.json'}")


if __name__ == "__main__":
    main()
