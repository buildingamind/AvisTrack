#!/usr/bin/env python3
"""
train/run_train.py
──────────────────
Batch YOLO training runner for multi-phase W2 Collective experiment plans.

Reads an experiment YAML, iterates over the defined runs, skips already-
completed ones (best.pt exists), and optionally evaluates each run
immediately after training.

Usage
-----
    # Dry run — print what would be launched
    python train/run_train.py \\
        --experiment train/experiments/W2_collective_phase1.yaml \\
        --dry-run

    # Run all (skip already-completed)
    python train/run_train.py \\
        --experiment train/experiments/W2_collective_phase1.yaml

    # Run only one candidate
    python train/run_train.py \\
        --experiment train/experiments/W2_collective_phase2.yaml \\
        --only w2_scratch

    # Force re-run even if best.pt exists
    python train/run_train.py \\
        --experiment train/experiments/W2_collective_phase1.yaml \\
        --force

Experiment YAML schema
----------------------
    output_root: /path/to/PhaseN_Dir
    data: /path/to/data.yaml        # optional top-level default
    defaults:                        # applied to every run unless overridden
      epochs: 15
      imgsz:  640
      batch:  16
      device: 0
      patience: 10
      exist_ok: true
    runs:
      - name: yolo11n
        model: yolo11n.pt
      - name: w3_finetune
        model: /path/to/best.pt
        data:  /path/to/other_data.yaml   # per-run override
        lr0:   0.001
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


# ── Experiment loading ────────────────────────────────────────────────────

def load_experiment(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_run_args(
    defaults: dict,
    run: dict,
    output_root: str,
    top_level_data: str | None,
) -> dict:
    """
    Merge defaults with per-run overrides.
    Per-run keys always win.  'data' falls back to top_level_data if not
    specified in the run itself.
    """
    args = {**defaults}
    args.update({k: v for k, v in run.items() if k != "name"})
    args["project"] = str(Path(output_root))
    args["name"]    = run["name"]
    if "data" not in args:
        if top_level_data:
            args["data"] = top_level_data
        else:
            raise ValueError(
                f"Run '{run['name']}' has no 'data' and no top-level data is defined.")
    return args


def is_completed(output_root: str, run_name: str) -> bool:
    """Return True if best.pt already exists for this run."""
    return (Path(output_root) / run_name / "weights" / "best.pt").exists()


# ── Training ──────────────────────────────────────────────────────────────

def run_training(train_args: dict, dry_run: bool) -> bool:
    """Launch a single YOLO training run.  Returns True on success."""
    model_init = train_args.pop("model")

    if dry_run:
        print(f"    YOLO({model_init!r}).train(")
        for k, v in sorted(train_args.items()):
            print(f"      {k}={v!r}")
        print("    )")
        return True

    from ultralytics import YOLO
    model = YOLO(model_init)
    results = model.train(**train_args)
    return results is not None


# ── Post-training eval ────────────────────────────────────────────────────

def run_eval(output_root: str, run_name: str, eval_config: str) -> None:
    weights  = str(Path(output_root) / run_name / "weights" / "best.pt")
    out_csv  = f"eval/reports/{run_name}_eval.csv"
    cmd = [
        sys.executable, "eval/run_eval.py",
        "--config",  eval_config,
        "--weights", weights,
        "--output",  out_csv,
    ]
    print(f"    eval → {out_csv}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"    ⚠  Eval returned non-zero for {run_name}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch YOLO training runner for multi-phase experiments.")
    parser.add_argument("--experiment", required=True,
        help="Path to experiment YAML config.")
    parser.add_argument("--dry-run", action="store_true",
        help="Print training calls without actually running them.")
    parser.add_argument("--only", default=None,
        help="Run only the candidate with this name.")
    parser.add_argument("--force", action="store_true",
        help="Re-run even if best.pt already exists.")
    parser.add_argument("--eval", action="store_true",
        help="Run eval/run_eval.py after each completed training run.")
    parser.add_argument("--eval-config", default=None,
        help="AvisTrack YAML config for evaluation (required with --eval).")
    args = parser.parse_args()

    if args.eval and not args.eval_config:
        parser.error("--eval requires --eval-config")

    exp = load_experiment(args.experiment)

    output_root    = exp["output_root"]
    top_level_data = exp.get("data")
    defaults       = exp.get("defaults", {})
    runs           = exp.get("runs", [])

    if not runs:
        print("No runs defined in experiment config.")
        sys.exit(1)

    if args.only:
        runs = [r for r in runs if r["name"] == args.only]
        if not runs:
            print(f"No run named '{args.only}' found in {args.experiment}")
            sys.exit(1)

    print(f"Experiment : {args.experiment}")
    print(f"Output root: {output_root}")
    print(f"Runs       : {len(runs)}")
    if args.dry_run:
        print("Mode       : DRY RUN")
    print()

    completed_names: list[str] = []
    skipped_names:   list[str] = []
    failed_names:    list[str] = []

    for run in runs:
        name = run["name"]

        if not args.force and is_completed(output_root, name):
            print(f"  [skip] {name}  (best.pt exists)")
            skipped_names.append(name)
            continue

        print(f"  [run ] {name}")
        try:
            train_args = build_run_args(
                defaults.copy(), run.copy(), output_root, top_level_data)
        except ValueError as e:
            print(f"  [ERR] Config error: {e}")
            failed_names.append(name)
            continue

        ok = run_training(train_args, dry_run=args.dry_run)

        if ok:
            completed_names.append(name)
            print(f"  [done] {name}\n")
            if args.eval and not args.dry_run:
                run_eval(output_root, name, args.eval_config)
        else:
            failed_names.append(name)
            print(f"  [FAIL] {name} failed\n")

    # Summary
    print("-" * 56)
    print(f"Completed : {len(completed_names)}   "
          f"Skipped : {len(skipped_names)}   "
          f"Failed : {len(failed_names)}")
    if completed_names:
        print(f"  done    : {', '.join(completed_names)}")
    if skipped_names:
        print(f"  skipped : {', '.join(skipped_names)}")
    if failed_names:
        print(f"  failed  : {', '.join(failed_names)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
