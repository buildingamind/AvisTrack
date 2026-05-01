#!/usr/bin/env python3
"""
train/run_train.py
──────────────────
Batch YOLO training runner.

Two experiment YAML schemas are supported:

* **Workspace mode** (Step E, recommended) – the YAML declares
  ``chamber_type``, ``workspace_yaml``, ``experiment_name``,
  ``dataset_name`` and ``phase``. Output and data paths are resolved
  against the workspace; ``meta.json`` + ``snapshots/`` are written on
  first launch and lineage is appended to ``models/index.csv`` when
  training ends. See ``avistrack.config.schema.ExperimentConfig``.

* **Legacy mode** – the YAML carries absolute ``output_root`` /
  ``data:`` paths. Preserved unchanged so historical
  ``train/experiments/W2_collective_phase{1,2,3}.yaml`` still work
  during the migration.

Usage
-----
    # Dry run — print what would be launched
    python train/run_train.py --experiment train/experiments/templates/phase1_template.yaml --dry-run

    # Run all (skip already-completed)
    python train/run_train.py --experiment <yaml>

    # Run only one candidate
    python train/run_train.py --experiment <yaml> --only yolo11n

    # Force re-run even if best.pt exists
    python train/run_train.py --experiment <yaml> --force

Workspace experiment YAML schema
--------------------------------
    chamber_type: collective
    workspace_yaml: "{workspace_root}/collective/workspace.yaml"
    experiment_name: W2_collective_phase1
    dataset_name: full_v1
    phase: 1
    defaults:
      epochs: 300
      imgsz: 640
      batch: 16
      device: 0
      patience: 20
    runs:
      - { name: yolo11n, model: yolo11n.pt }
      - { name: ft_phase1_winner, model: phase1/yolo11n/weights/best.pt, lr0: 0.001 }
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from avistrack import lineage as L
from avistrack.config.loader import load_experiment, load_workspace
from avistrack.config.schema import ExperimentConfig, WorkspaceConfig


# ── Schema detection ─────────────────────────────────────────────────────

def _is_workspace_schema(raw: dict) -> bool:
    """A workspace-mode YAML must carry chamber_type AND workspace_yaml."""
    return "chamber_type" in raw and "workspace_yaml" in raw


def _read_raw(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ── Workspace-mode helpers ───────────────────────────────────────────────

def _resolve_workspace(exp: ExperimentConfig,
                       workspace_root: Optional[str]) -> tuple[WorkspaceConfig, Path, Path, Path]:
    """Return (workspace_cfg, exp_dir, phase_dir, data_yaml)."""
    workspace_yaml = exp.workspace_yaml
    if "{workspace_root}" in workspace_yaml:
        if workspace_root is None:
            raise SystemExit(
                "experiment.workspace_yaml still contains '{workspace_root}'. "
                "Pass --workspace-root or set AVISTRACK_WORKSPACE_ROOT."
            )
        workspace_yaml = workspace_yaml.replace("{workspace_root}", str(workspace_root))

    ws_path = Path(workspace_yaml).expanduser().resolve()
    if not ws_path.exists():
        raise SystemExit(f"workspace yaml not found: {ws_path}")

    workspace = load_workspace(ws_path)
    if workspace.chamber_type != exp.chamber_type:
        raise SystemExit(
            f"chamber_type mismatch: experiment says {exp.chamber_type!r}, "
            f"workspace.yaml at {ws_path} says {workspace.chamber_type!r}"
        )

    workspace_chamber_dir = Path(workspace.workspace.root)
    datasets_root = Path(workspace.workspace.dataset)
    models_root   = Path(workspace.workspace.models)
    dataset_dir   = datasets_root / exp.dataset_name
    data_yaml     = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(
            f"data.yaml not found at {data_yaml}. Build the dataset first:\n"
            f"  python tools/build_dataset.py --workspace-yaml {ws_path} "
            f"--recipe <recipe.yaml>"
        )

    exp_dir   = models_root / exp.experiment_name
    phase_dir = exp_dir / f"phase{exp.phase}"
    return workspace, exp_dir, phase_dir, data_yaml


def _ensure_meta_and_snapshot(
    exp_dir: Path,
    exp: ExperimentConfig,
    experiment_yaml: Path,
    data_yaml: Path,
    workspace_root: Path,
) -> L.ExperimentMeta:
    """First launch: write meta.json + take_snapshot. Subsequent: validate match."""
    dataset_dir = data_yaml.parent
    recipe_yaml = dataset_dir / "recipe.yaml"
    if not recipe_yaml.exists():
        raise SystemExit(
            f"recipe.yaml not found at {recipe_yaml}. The dataset directory "
            f"appears to be malformed."
        )
    recipe_hash = L.hash_recipe(recipe_yaml)

    meta_path = exp_dir / L.META_FILENAME
    if meta_path.exists():
        existing = L.read_meta(exp_dir)
        if existing.experiment_name != exp.experiment_name:
            raise SystemExit(
                f"meta.json experiment_name mismatch: stored "
                f"{existing.experiment_name!r}, yaml says {exp.experiment_name!r}"
            )
        if existing.dataset_name != exp.dataset_name:
            raise SystemExit(
                f"experiment {exp.experiment_name!r} was first launched with "
                f"dataset {existing.dataset_name!r}; cannot now switch to "
                f"{exp.dataset_name!r}. Use a new experiment_name."
            )
        if existing.recipe_hash != recipe_hash:
            raise SystemExit(
                f"recipe content for dataset {exp.dataset_name!r} has changed "
                f"(hash {existing.recipe_hash} → {recipe_hash}). Rebuild the "
                f"dataset under a different name to start a new experiment."
            )
        return existing

    meta = L.ExperimentMeta(
        experiment_name=exp.experiment_name,
        chamber_type=exp.chamber_type,
        dataset_name=exp.dataset_name,
        recipe_hash=recipe_hash,
        git_sha=L.git_sha(REPO_ROOT),
        git_dirty=L.git_dirty(REPO_ROOT),
        started_at=L.now_iso(),
        workspace_root=str(workspace_root),
    )
    L.write_meta(exp_dir, meta)
    L.take_snapshot(exp_dir, experiment_yaml, recipe_yaml, data_yaml,
                    repo_root=REPO_ROOT)
    return meta


def _resolve_run_model(model_field: str, exp_dir: Path) -> str:
    """Resolve a run's ``model`` field.

    Rules:
      * ``yolov8n.pt`` (no path separator) → ultralytics registry name, passthrough
      * relative path with separator → tried under ``exp_dir/`` first
        (``phase1/yolo11n/weights/best.pt`` → ``{exp_dir}/phase1/...``)
      * absolute path → passthrough
    Falls back to passthrough if neither relative-to-exp_dir nor absolute exists,
    so dry-runs and pre-training inspections don't error out.
    """
    if "/" not in model_field and "\\" not in model_field:
        return model_field
    p = Path(model_field)
    if p.is_absolute():
        return str(p)
    candidate = exp_dir / model_field
    if candidate.exists():
        return str(candidate)
    return str(candidate)   # let YOLO surface the error if missing


# ── Run building ─────────────────────────────────────────────────────────

def build_run_args(
    defaults: dict,
    run: dict,
    output_root: str,
    top_level_data: Optional[str],
) -> dict:
    """Merge defaults with per-run overrides; per-run keys win.

    ``data`` falls back to ``top_level_data`` if not specified per-run.
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


def is_completed(output_root: str | Path, run_name: str) -> bool:
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

    from ultralytics import YOLO   # imported lazily so dry-runs don't need it
    model = YOLO(model_init)
    results = model.train(**train_args)
    return results is not None


# ── Post-training eval ────────────────────────────────────────────────────

def run_eval(output_root: str | Path, run_name: str, eval_config: str) -> None:
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


# ── Workspace-mode driver ────────────────────────────────────────────────

def run_workspace_experiment(
    experiment_yaml: Path,
    *,
    only: Optional[str],
    force: bool,
    dry_run: bool,
    do_eval: bool,
    eval_config: Optional[str],
    workspace_root: Optional[str],
) -> int:
    exp = load_experiment(experiment_yaml, workspace_root=workspace_root)
    workspace, exp_dir, phase_dir, data_yaml = _resolve_workspace(exp, workspace_root)

    if not dry_run:
        exp_dir.mkdir(parents=True, exist_ok=True)
    ws_root = Path(workspace.workspace.root).parent

    if not dry_run:
        meta = _ensure_meta_and_snapshot(exp_dir, exp, experiment_yaml,
                                         data_yaml, ws_root)
    else:
        meta = None

    runs = list(exp.runs)
    if only:
        runs = [r for r in runs if r.name == only]
        if not runs:
            print(f"No run named '{only}' found in {experiment_yaml}")
            return 1

    print(f"Experiment   : {exp.experiment_name}  (phase {exp.phase})")
    print(f"Workspace    : {ws_root}")
    print(f"Dataset      : {exp.dataset_name}  ({data_yaml})")
    print(f"Output       : {phase_dir}")
    print(f"Runs         : {len(runs)}")
    if dry_run:
        print(f"Mode         : DRY RUN")
    print()

    completed: list[str] = []
    skipped:   list[str] = []
    failed:    list[str] = []

    defaults = exp.defaults.model_dump()

    for run in runs:
        name = run.name
        if not force and is_completed(phase_dir, name):
            print(f"  [skip] {name}  (best.pt exists)")
            skipped.append(name)
            continue

        print(f"  [run ] {name}")
        run_dict = {"name": name, "model": _resolve_run_model(run.model, exp_dir)}
        if run.model_extra:
            run_dict.update(run.model_extra)

        try:
            train_args = build_run_args(
                defaults.copy(), run_dict, str(phase_dir), str(data_yaml))
        except ValueError as e:
            print(f"  [ERR] Config error: {e}")
            failed.append(name)
            continue

        ok = run_training(train_args, dry_run=dry_run)
        if ok:
            completed.append(name)
            print(f"  [done] {name}\n")
            if do_eval and not dry_run:
                run_eval(phase_dir, name, eval_config)
        else:
            failed.append(name)
            print(f"  [FAIL] {name} failed\n")

    if not dry_run and meta is not None:
        L.update_meta(exp_dir, ended_at=L.now_iso())

    _print_summary(completed, skipped, failed)
    return 1 if failed else 0


# ── Legacy-mode driver (preserved verbatim) ──────────────────────────────

def run_legacy_experiment(
    experiment_yaml: str,
    *,
    only: Optional[str],
    force: bool,
    dry_run: bool,
    do_eval: bool,
    eval_config: Optional[str],
) -> int:
    with open(experiment_yaml) as f:
        exp = yaml.safe_load(f)

    output_root    = exp["output_root"]
    top_level_data = exp.get("data")
    defaults       = exp.get("defaults", {})
    runs           = exp.get("runs", [])

    if not runs:
        print("No runs defined in experiment config.")
        return 1

    if only:
        runs = [r for r in runs if r["name"] == only]
        if not runs:
            print(f"No run named '{only}' found in {experiment_yaml}")
            return 1

    print(f"Experiment : {experiment_yaml}")
    print(f"Output root: {output_root}")
    print(f"Runs       : {len(runs)}")
    if dry_run:
        print("Mode       : DRY RUN")
    print()

    completed: list[str] = []
    skipped:   list[str] = []
    failed:    list[str] = []

    for run in runs:
        name = run["name"]

        if not force and is_completed(output_root, name):
            print(f"  [skip] {name}  (best.pt exists)")
            skipped.append(name)
            continue

        print(f"  [run ] {name}")
        try:
            train_args = build_run_args(
                defaults.copy(), run.copy(), output_root, top_level_data)
        except ValueError as e:
            print(f"  [ERR] Config error: {e}")
            failed.append(name)
            continue

        ok = run_training(train_args, dry_run=dry_run)
        if ok:
            completed.append(name)
            print(f"  [done] {name}\n")
            if do_eval and not dry_run:
                run_eval(output_root, name, eval_config)
        else:
            failed.append(name)
            print(f"  [FAIL] {name} failed\n")

    _print_summary(completed, skipped, failed)
    return 1 if failed else 0


def _print_summary(completed: list[str], skipped: list[str], failed: list[str]) -> None:
    print("-" * 56)
    print(f"Completed : {len(completed)}   "
          f"Skipped : {len(skipped)}   "
          f"Failed : {len(failed)}")
    if completed:
        print(f"  done    : {', '.join(completed)}")
    if skipped:
        print(f"  skipped : {', '.join(skipped)}")
    if failed:
        print(f"  failed  : {', '.join(failed)}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch YOLO training runner (workspace + legacy modes).")
    parser.add_argument("--experiment", required=True,
        help="Path to experiment YAML config.")
    parser.add_argument("--workspace-root", default=None,
        help="Override the workspace root (only honoured by workspace-mode YAMLs).")
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

    raw = _read_raw(args.experiment)
    if _is_workspace_schema(raw):
        rc = run_workspace_experiment(
            Path(args.experiment),
            only=args.only,
            force=args.force,
            dry_run=args.dry_run,
            do_eval=args.eval,
            eval_config=args.eval_config,
            workspace_root=args.workspace_root,
        )
    else:
        rc = run_legacy_experiment(
            args.experiment,
            only=args.only,
            force=args.force,
            dry_run=args.dry_run,
            do_eval=args.eval,
            eval_config=args.eval_config,
        )
    sys.exit(rc)


if __name__ == "__main__":
    main()
