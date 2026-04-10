#!/usr/bin/env python3
"""
train/run_pipeline.py
─────────────────────
Full W2 Collective training pipeline: Phase 1 → 2 → 3

  Phase 1 : architecture scan  (yolo11n vs yolo26n, 15 epochs, ~1 hr)
  Phase 2 : strategy scan      (5 runs, 100 epochs, ~12 hr)
  Phase 3 : augmentation tuning (12 runs, 100 epochs, ~14 hr)

Between phases the pipeline automatically:
  - picks the Phase 1 winner and patches Phase 2 scratch-run models
  - picks Phase 2 top-2 by mAP50-95 and fills in the Phase 3 model paths

Usage
-----
    # Run full pipeline (Phase 1 → 2 → 3)
    python train/run_pipeline.py

    # Skip completed phases (e.g. Phase 1 already done)
    python train/run_pipeline.py --start-phase 2

    # Dry-run to verify config without training
    python train/run_pipeline.py --dry-run
"""

import argparse
import csv
import sys
from pathlib import Path

import yaml

PYTHON = sys.executable

PHASE1_YAML        = "train/experiments/W2_collective_phase1.yaml"
PHASE2_YAML        = "train/experiments/W2_collective_phase2.yaml"
PHASE3_YAML        = "train/experiments/W2_collective_phase3.yaml"
PHASE3_FILLED_YAML = "train/experiments/W2_collective_phase3_filled.yaml"

W2_DATA   = "E:/Wave2/03_Model_Training/Datasets/YOLO/W2_iou096_v1/data.yaml"
W2W3_DATA = "E:/Wave2/03_Model_Training/Datasets/YOLO/W2W3_combined_iou096_092_v1/data.yaml"

# ── Helpers ───────────────────────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def run_phase(yaml_path: str, dry_run: bool) -> int:
    import subprocess
    cmd = [PYTHON, "train/run_train.py", "--experiment", yaml_path]
    if dry_run:
        cmd.append("--dry-run")
    return subprocess.run(cmd).returncode


def read_best_map95(results_csv: Path) -> float:
    """Return the best mAP50-95 value from a YOLO results.csv."""
    best = 0.0
    with open(results_csv, newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue  # header row
            try:
                val = float(row[8].strip())   # metrics/mAP50-95(B)
                if val > best:
                    best = val
            except (IndexError, ValueError):
                continue
    return best


# ── Phase 1: pick winner ──────────────────────────────────────────────────

PHASE1_CANDIDATES = ["yolo8n", "yolo8s", "yolo11n", "yolo11s", "yolo26n", "yolo26s"]


def phase1_winner(output_root: str) -> tuple[str, str]:
    """Return (run_name, model_file) for the Phase 1 winner by best mAP50-95."""
    root = Path(output_root)
    # Map run name → model file (used for patching Phase 2 scratch runs)
    model_map = {
        "yolo8n":  "yolov8n.pt",
        "yolo8s":  "yolov8s.pt",
        "yolo11n": "yolo11n.pt",
        "yolo11s": "yolo11s.pt",
        "yolo26n": "yolo26n.pt",
        "yolo26s": "yolo26s.pt",
    }
    scores = {}
    for name in PHASE1_CANDIDATES:
        csv_path = root / name / "results.csv"
        if csv_path.exists():
            scores[name] = read_best_map95(csv_path)

    if not scores:
        print("[pipeline] No Phase 1 results found; defaulting to yolo11n")
        return "yolo11n", "yolo11n.pt"

    print("[pipeline] Phase 1 results:")
    winner = max(scores, key=scores.__getitem__)
    for name in PHASE1_CANDIDATES:
        if name in scores:
            tag = " <- winner" if name == winner else ""
            print(f"  {name:10s}  mAP50-95 = {scores[name]:.4f}{tag}")
        else:
            print(f"  {name:10s}  (no results)")
    return winner, model_map[winner]


def patch_phase2_scratch_models(phase2_yaml: str, winner_model: str) -> None:
    """Replace yolo11n.pt → winner model in scratch runs if different."""
    exp = load_yaml(phase2_yaml)
    changed = False
    for run in exp.get("runs", []):
        if run.get("model") == "yolo11n.pt" and run.get("model") != winner_model:
            run["model"] = winner_model
            changed = True
    if changed:
        save_yaml(exp, phase2_yaml)
        print(f"[pipeline] Phase 2: patched scratch runs to use {winner_model}")


# ── Phase 2: pick top-2 ───────────────────────────────────────────────────

def phase2_top2(output_root: str, runs: list) -> list:
    """Return top-2 run dicts sorted by best mAP50-95."""
    root = Path(output_root)
    scored = []
    print("[pipeline] Phase 2 results:")
    for run in runs:
        name = run["name"]
        csv_path = root / name / "results.csv"
        if csv_path.exists():
            score = read_best_map95(csv_path)
            scored.append((score, run))
            print(f"  {name:25s}  mAP50-95 = {score:.4f}")
        else:
            print(f"  {name:25s}  (no results — skipped or failed)")

    scored.sort(key=lambda x: x[0], reverse=True)
    top2 = [r for _, r in scored[:2]]

    if len(top2) < 2:
        raise RuntimeError(
            f"Need at least 2 Phase 2 results; only found {len(top2)}.")
    print(f"[pipeline] Top-2: {top2[0]['name']}, {top2[1]['name']}")
    return top2


# ── Phase 3: fill template ────────────────────────────────────────────────

def _data_for_run(run: dict) -> str:
    """Infer data path from the run name (combined vs W2-only)."""
    if "combined" in run["name"]:
        return W2W3_DATA
    return W2_DATA


def build_phase3_filled(template_path: str, winner1: dict, winner2: dict,
                        phase2_root: str, out_path: str) -> None:
    """Write a filled Phase 3 yaml with winner best.pt paths substituted."""
    exp = load_yaml(template_path)

    w1_pt = str(Path(phase2_root) / winner1["name"] / "weights" / "best.pt")
    w2_pt = str(Path(phase2_root) / winner2["name"] / "weights" / "best.pt")

    # Use winner1's data domain for Phase 3 top-level default
    exp["data"] = _data_for_run(winner1)

    for run in exp.get("runs", []):
        model = run.get("model", "")
        if "winner1" in model or "FILL_IN_winner1" in model:
            run["model"] = w1_pt
        elif "winner2" in model or "FILL_IN_winner2" in model:
            run["model"] = w2_pt

    save_yaml(exp, out_path)
    print(f"[pipeline] Phase 3 yaml written → {out_path}")
    print(f"  winner1 ({winner1['name']}): {w1_pt}")
    print(f"  winner2 ({winner2['name']}): {w2_pt}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print training calls without running them.")
    parser.add_argument("--start-phase", type=int, default=1, choices=[1, 2, 3],
                        help="Skip earlier phases (assumes they are complete).")
    args = parser.parse_args()

    phase1_exp = load_yaml(PHASE1_YAML)
    phase2_exp = load_yaml(PHASE2_YAML)

    phase1_root = phase1_exp["output_root"]
    phase2_root = phase2_exp["output_root"]

    # ── Phase 1 ──────────────────────────────────────────────────────────
    if args.start_phase <= 1:
        print("\n=== Phase 1: Architecture Scan ===")
        rc = run_phase(PHASE1_YAML, args.dry_run)
        if rc != 0:
            print("[pipeline] Phase 1 failed — aborting.")
            sys.exit(rc)

    # ── Pick Phase 1 winner & patch Phase 2 ──────────────────────────────
    print()
    winner_arch, winner_model = phase1_winner(phase1_root)
    print(f"[pipeline] Phase 1 winner: {winner_arch} ({winner_model})")
    if not args.dry_run:
        patch_phase2_scratch_models(PHASE2_YAML, winner_model)
        # Reload after potential patch
        phase2_exp = load_yaml(PHASE2_YAML)

    # ── Phase 2 ──────────────────────────────────────────────────────────
    if args.start_phase <= 2:
        print("\n=== Phase 2: Training Strategy ===")
        rc = run_phase(PHASE2_YAML, args.dry_run)
        if rc != 0:
            print("[pipeline] Phase 2 failed — aborting.")
            sys.exit(rc)

    # ── Pick Phase 2 top-2 & build Phase 3 yaml ──────────────────────────
    print()
    if args.dry_run:
        print("[pipeline] Dry-run: skipping Phase 2 top-2 selection and Phase 3 yaml build.")
        print("=== Pipeline dry-run complete ===")
        return

    top2 = phase2_top2(phase2_root, phase2_exp["runs"])
    build_phase3_filled(PHASE3_YAML, top2[0], top2[1], phase2_root,
                        PHASE3_FILLED_YAML)

    # ── Phase 3 ──────────────────────────────────────────────────────────
    if args.start_phase <= 3:
        print("\n=== Phase 3: Augmentation Tuning ===")
        rc = run_phase(PHASE3_FILLED_YAML, args.dry_run)
        if rc != 0:
            print("[pipeline] Phase 3 failed.")
            sys.exit(rc)

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()
