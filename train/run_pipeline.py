#!/usr/bin/env python3
"""
train/run_pipeline.py
─────────────────────
Three-phase training pipeline driver (workspace-aware, Step E schema).

  Phase 1 : architecture scan
  Phase 2 : training strategy scan (uses phase 1 winner as a starting point)
  Phase 3 : augmentation tuning (uses phase 2 top-2 as starting points)

After each phase the driver writes:
  * ``{exp}/phase{N}/leaderboard.csv``   — runs sorted by mAP50-95
  * ``{exp}/phase{N}/winner.json``       — top-1 with score

After phase 3 finishes:
  * ``{exp}/final/best.pt``              — copy of winning weights
  * ``{exp}/final/source.json``          — which phase/run it came from
  * ``{exp}/meta.json``                  — ``ended_at`` + ``final_weights`` set
  * ``{models_root}/index.csv``          — appended/updated

Usage
-----
    # All three phases, single experiment
    python train/run_pipeline.py \
        --phase1 train/experiments/templates/phase1_template.yaml \
        --phase2 train/experiments/templates/phase2_template.yaml \
        --phase3 train/experiments/templates/phase3_template.yaml \
        --workspace-root /media/ssd/avistrack

    # Skip ahead (Phase 1 already done)
    python train/run_pipeline.py ... --start-phase 2

    # Dry-run (no training, just show the call graph)
    python train/run_pipeline.py ... --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from avistrack import lineage as L
from avistrack.config.loader import load_experiment, load_workspace
from avistrack.config.schema import ExperimentConfig

PYTHON = sys.executable


# ── Helpers ───────────────────────────────────────────────────────────────

def _load_phase_yaml(path: str, workspace_root: Optional[str]) -> ExperimentConfig:
    return load_experiment(path, workspace_root=workspace_root)


def _resolve_phase_dirs(exp: ExperimentConfig,
                        workspace_root: Optional[str]) -> tuple[Path, Path]:
    """Return (exp_dir, phase_dir)."""
    workspace_yaml = exp.workspace_yaml
    if "{workspace_root}" in workspace_yaml:
        if workspace_root is None:
            raise SystemExit(
                "phase yaml workspace_yaml has '{workspace_root}'; "
                "pass --workspace-root."
            )
        workspace_yaml = workspace_yaml.replace("{workspace_root}", str(workspace_root))
    workspace = load_workspace(Path(workspace_yaml))
    models_root = Path(workspace.workspace.models)
    exp_dir = models_root / exp.experiment_name
    phase_dir = exp_dir / f"phase{exp.phase}"
    return exp_dir, phase_dir


def _run_train(exp_yaml: Path, workspace_root: Optional[str], dry_run: bool) -> int:
    cmd = [PYTHON, str(REPO_ROOT / "train" / "run_train.py"),
           "--experiment", str(exp_yaml)]
    if workspace_root:
        cmd += ["--workspace-root", str(workspace_root)]
    if dry_run:
        cmd.append("--dry-run")
    return subprocess.run(cmd).returncode


# ── results.csv parsing ──────────────────────────────────────────────────

def _read_run_results(run_dir: Path) -> dict:
    """Return best metrics row from ``run_dir/results.csv`` or empty dict."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {}
    best_row: dict = {}
    best_score = -1.0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v for k, v in row.items()}
            try:
                score = float(row.get("metrics/mAP50-95(B)") or
                              row.get("metrics/mAP50-95") or
                              row.get("metrics/mAP_0.5:0.95(B)") or 0.0)
            except ValueError:
                continue
            if score > best_score:
                best_score = score
                best_row = row
    if best_row:
        best_row["best_mAP50-95"] = best_score
    return best_row


def _phase_runs(phase_dir: Path) -> list[Path]:
    """Each subdir of phase_dir whose ``results.csv`` exists."""
    return sorted(p for p in phase_dir.iterdir()
                  if p.is_dir() and (p / "results.csv").exists())


# ── leaderboard + winner ─────────────────────────────────────────────────

LEADERBOARD_FIELDS = [
    "name", "best_mAP50-95", "best_mAP50",
    "best_precision", "best_recall", "best_f1",
    "epochs_run", "weights_path",
]


def _write_leaderboard(phase_dir: Path,
                       exp: ExperimentConfig) -> Optional[dict]:
    """Write leaderboard.csv + winner.json. Returns winner dict or None."""
    rows = []
    for run_dir in _phase_runs(phase_dir):
        metrics = _read_run_results(run_dir)
        if not metrics:
            continue
        rows.append({
            "name":          run_dir.name,
            "best_mAP50-95": float(metrics.get("best_mAP50-95", 0.0)),
            "best_mAP50":    metrics.get("metrics/mAP50(B)") or
                             metrics.get("metrics/mAP50") or "",
            "best_precision": metrics.get("metrics/precision(B)") or
                              metrics.get("metrics/precision") or "",
            "best_recall":    metrics.get("metrics/recall(B)") or
                              metrics.get("metrics/recall") or "",
            "best_f1":        metrics.get("metrics/F1(B)") or
                              metrics.get("metrics/F1") or "",
            "epochs_run":    metrics.get("epoch", ""),
            "weights_path":  str(run_dir / "weights" / "best.pt"),
        })
    if not rows:
        print(f"  [pipeline] No results found under {phase_dir}.")
        return None

    rows.sort(key=lambda r: r["best_mAP50-95"], reverse=True)
    leaderboard_csv = phase_dir / "leaderboard.csv"
    with open(leaderboard_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in LEADERBOARD_FIELDS})

    winner = rows[0]
    winner_json = phase_dir / "winner.json"
    winner_json.write_text(json.dumps({
        "phase": exp.phase,
        "run_name": winner["name"],
        "score": winner["best_mAP50-95"],
        "weights_path": winner["weights_path"],
        "selected_at": L.now_iso(),
    }, indent=2))

    print(f"\n  [pipeline] Phase {exp.phase} leaderboard:")
    for i, r in enumerate(rows, 1):
        tag = " <- winner" if i == 1 else ""
        print(f"    {i}. {r['name']:30s}  mAP50-95 = {r['best_mAP50-95']:.4f}{tag}")
    print(f"  [pipeline] Saved {leaderboard_csv} + {winner_json.name}")
    return winner


def _phase_top_n(phase_dir: Path, n: int) -> list[dict]:
    """Return top-N runs from leaderboard.csv (after _write_leaderboard ran)."""
    leaderboard_csv = phase_dir / "leaderboard.csv"
    if not leaderboard_csv.exists():
        return []
    with open(leaderboard_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    rows = rows[:n]
    out = []
    for r in rows:
        out.append({"name": r["name"],
                    "score": float(r["best_mAP50-95"]),
                    "weights_path": r["weights_path"]})
    return out


# ── Phase 2 / Phase 3 placeholder substitution ───────────────────────────

def _patch_phase2_winner(phase2_yaml: Path, phase1_winner: dict,
                         tmp_dir: Path) -> Path:
    """Substitute phase 1 winner into any phase2 run that uses 'yolo11s.pt'
    or 'PHASE1_WINNER' as the model. Writes a temp filled yaml.
    """
    raw = yaml.safe_load(phase2_yaml.read_text())
    winner_path = f"phase1/{phase1_winner['name']}/weights/best.pt"
    for run in raw.get("runs", []) or []:
        m = run.get("model", "")
        if m == "PHASE1_WINNER" or m.endswith("yolo11s.pt") or m.endswith("yolo11n.pt"):
            # Only substitute the placeholder/winner-token; leave actual paths alone.
            if m == "PHASE1_WINNER":
                run["model"] = winner_path
    out = tmp_dir / "phase2_filled.yaml"
    out.write_text(yaml.safe_dump(raw, sort_keys=False))
    return out


def _patch_phase3_winners(phase3_yaml: Path, phase2_top2: list[dict],
                          tmp_dir: Path) -> Path:
    """Replace WINNER1 / WINNER2 tokens in phase3 yaml with actual run names."""
    if len(phase2_top2) < 1:
        raise SystemExit("Phase 2 produced no runs — cannot fill phase 3.")
    if len(phase2_top2) < 2:
        # Allow single-winner pipelines: WINNER2 is replaced with WINNER1.
        phase2_top2 = phase2_top2 + [phase2_top2[0]]

    text = phase3_yaml.read_text()
    text = text.replace("WINNER1", phase2_top2[0]["name"])
    text = text.replace("WINNER2", phase2_top2[1]["name"])
    out = tmp_dir / "phase3_filled.yaml"
    out.write_text(text)
    return out


# ── Final winner copy + index update ─────────────────────────────────────

def _copy_final_winner(exp_dir: Path, phase: int, winner: dict) -> Path:
    final_dir = exp_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    src = Path(winner["weights_path"])
    dst = final_dir / "best.pt"
    shutil.copy2(src, dst)
    (final_dir / "source.json").write_text(json.dumps({
        "phase": phase,
        "run_name": winner["name"],
        "score": winner["score"] if "score" in winner else winner.get("best_mAP50-95"),
        "source_weights": str(src),
        "copied_at": L.now_iso(),
    }, indent=2))
    return dst


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase1", required=True, type=Path,
                        help="Path to Phase 1 experiment YAML.")
    parser.add_argument("--phase2", required=True, type=Path,
                        help="Path to Phase 2 experiment YAML.")
    parser.add_argument("--phase3", required=True, type=Path,
                        help="Path to Phase 3 experiment YAML.")
    parser.add_argument("--workspace-root", default=None,
                        help="Workspace root for {workspace_root} placeholder resolution.")
    parser.add_argument("--start-phase", type=int, default=1, choices=[1, 2, 3],
                        help="Skip earlier phases (assume they are complete).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print training calls without running them.")
    args = parser.parse_args()

    phase1_exp = _load_phase_yaml(args.phase1, args.workspace_root)
    phase2_exp = _load_phase_yaml(args.phase2, args.workspace_root)
    phase3_exp = _load_phase_yaml(args.phase3, args.workspace_root)

    # Sanity: all three phases must agree on experiment_name + chamber_type.
    for ex, name in [(phase2_exp, "phase2"), (phase3_exp, "phase3")]:
        if ex.experiment_name != phase1_exp.experiment_name:
            raise SystemExit(
                f"{name}.experiment_name {ex.experiment_name!r} does not match "
                f"phase1 {phase1_exp.experiment_name!r}; all three phases of one "
                f"pipeline must share the experiment_name."
            )

    exp_dir, _ = _resolve_phase_dirs(phase1_exp, args.workspace_root)
    _, phase1_dir = _resolve_phase_dirs(phase1_exp, args.workspace_root)
    _, phase2_dir = _resolve_phase_dirs(phase2_exp, args.workspace_root)
    _, phase3_dir = _resolve_phase_dirs(phase3_exp, args.workspace_root)
    models_root = exp_dir.parent

    tmp_dir = Path(tempfile.mkdtemp(prefix="avistrack_pipeline_"))

    try:
        # ── Phase 1 ──────────────────────────────────────────────────────
        if args.start_phase <= 1:
            print("\n=== Phase 1 ===")
            rc = _run_train(args.phase1, args.workspace_root, args.dry_run)
            if rc != 0:
                print("[pipeline] Phase 1 failed — aborting.")
                sys.exit(rc)

        if args.dry_run:
            print("\n[pipeline] Dry-run: skipping leaderboard / phase 2/3.")
            return

        winner1 = _write_leaderboard(phase1_dir, phase1_exp)
        if not winner1:
            sys.exit("[pipeline] No phase 1 winner could be selected.")

        # ── Phase 2 ──────────────────────────────────────────────────────
        if args.start_phase <= 2:
            phase2_filled = _patch_phase2_winner(args.phase2, winner1, tmp_dir)
            print(f"\n=== Phase 2 (winner1={winner1['name']}) ===")
            rc = _run_train(phase2_filled, args.workspace_root, args.dry_run)
            if rc != 0:
                print("[pipeline] Phase 2 failed — aborting.")
                sys.exit(rc)

        winner2 = _write_leaderboard(phase2_dir, phase2_exp)
        if not winner2:
            sys.exit("[pipeline] No phase 2 winner could be selected.")
        top2 = _phase_top_n(phase2_dir, 2)

        # ── Phase 3 ──────────────────────────────────────────────────────
        if args.start_phase <= 3:
            phase3_filled = _patch_phase3_winners(args.phase3, top2, tmp_dir)
            print(f"\n=== Phase 3 (top-2={[r['name'] for r in top2]}) ===")
            rc = _run_train(phase3_filled, args.workspace_root, args.dry_run)
            if rc != 0:
                print("[pipeline] Phase 3 failed.")
                sys.exit(rc)

        final_winner = _write_leaderboard(phase3_dir, phase3_exp)
        if not final_winner:
            sys.exit("[pipeline] No phase 3 winner could be selected.")

        # ── Final + lineage index ────────────────────────────────────────
        final_pt = _copy_final_winner(exp_dir, phase=3, winner=final_winner)
        L.update_meta(exp_dir, ended_at=L.now_iso(),
                      final_weights=str(final_pt))
        meta = L.read_meta(exp_dir)
        L.append_index(models_root, meta, final_weights=final_pt)
        print(f"\n=== Pipeline complete — final/best.pt → {final_pt} ===")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
