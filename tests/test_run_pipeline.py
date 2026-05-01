"""
tests/test_run_pipeline.py – Step E4 helpers in train/run_pipeline.py.

The real pipeline shells out to ``run_train.py`` which in turn requires
ultralytics. These tests bypass the subprocess by exercising the
leaderboard / winner / placeholder-substitution helpers directly with
hand-crafted ``results.csv`` files. The actual subprocess wiring is left
to the manual end-to-end smoke test in the plan file.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from train import run_pipeline as RP


# ── Helpers ─────────────────────────────────────────────────────────────

def _make_results_csv(run_dir: Path, *, map5095: float, map50: float = 0.9) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "weights").mkdir(exist_ok=True)
    (run_dir / "weights" / "best.pt").write_text("(fake)")
    with open(run_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train/box_loss",
                    "metrics/precision(B)", "metrics/recall(B)",
                    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
                    "metrics/F1(B)"])
        # Two epochs so we exercise "best across rows" picking.
        w.writerow([1, 0.5, 0.7, 0.7, max(0.0, map50 - 0.05),
                    max(0.0, map5095 - 0.05), 0.7])
        w.writerow([2, 0.4, 0.8, 0.8, map50, map5095, 0.8])


class _StubExp:
    """Minimal stand-in for ExperimentConfig in helpers that need .phase."""
    def __init__(self, phase: int):
        self.phase = phase


# ── _read_run_results ───────────────────────────────────────────────────

def test_read_run_results_picks_best(tmp_path: Path):
    run_dir = tmp_path / "yolo11n"
    _make_results_csv(run_dir, map5095=0.42)
    out = RP._read_run_results(run_dir)
    assert pytest.approx(out["best_mAP50-95"], abs=1e-9) == 0.42


def test_read_run_results_missing(tmp_path: Path):
    out = RP._read_run_results(tmp_path / "no_such_run")
    assert out == {}


# ── _write_leaderboard ──────────────────────────────────────────────────

def test_write_leaderboard_orders_descending(tmp_path: Path):
    phase = tmp_path / "phase1"
    _make_results_csv(phase / "yolo8n",  map5095=0.40)
    _make_results_csv(phase / "yolo11n", map5095=0.55)
    _make_results_csv(phase / "yolo11s", map5095=0.50)

    winner = RP._write_leaderboard(phase, _StubExp(phase=1))

    assert winner is not None
    assert winner["name"] == "yolo11n"
    assert pytest.approx(winner["best_mAP50-95"], abs=1e-9) == 0.55

    rows = list(csv.DictReader((phase / "leaderboard.csv").open()))
    names = [r["name"] for r in rows]
    assert names == ["yolo11n", "yolo11s", "yolo8n"]

    winner_json = json.loads((phase / "winner.json").read_text())
    assert winner_json["run_name"] == "yolo11n"
    assert winner_json["phase"] == 1


def test_write_leaderboard_skips_runs_without_results(tmp_path: Path):
    phase = tmp_path / "phase1"
    _make_results_csv(phase / "yolo8n",  map5095=0.30)
    # Empty subdir — no results.csv → should be skipped silently.
    (phase / "empty_run").mkdir(parents=True)

    winner = RP._write_leaderboard(phase, _StubExp(phase=1))
    assert winner is not None
    assert winner["name"] == "yolo8n"
    rows = list(csv.DictReader((phase / "leaderboard.csv").open()))
    assert len(rows) == 1


def test_write_leaderboard_returns_none_when_empty(tmp_path: Path):
    phase = tmp_path / "phase1"
    phase.mkdir()
    assert RP._write_leaderboard(phase, _StubExp(phase=1)) is None


# ── _phase_top_n ────────────────────────────────────────────────────────

def test_phase_top_n(tmp_path: Path):
    phase = tmp_path / "phase2"
    _make_results_csv(phase / "a", map5095=0.30)
    _make_results_csv(phase / "b", map5095=0.45)
    _make_results_csv(phase / "c", map5095=0.40)
    RP._write_leaderboard(phase, _StubExp(phase=2))

    top2 = RP._phase_top_n(phase, 2)
    assert [r["name"] for r in top2] == ["b", "c"]
    assert pytest.approx(top2[0]["score"], abs=1e-9) == 0.45


# ── Placeholder substitution ────────────────────────────────────────────

def test_patch_phase2_winner_replaces_token(tmp_path: Path):
    yaml_path = tmp_path / "phase2.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "chamber_type": "collective",
        "workspace_yaml": "x", "experiment_name": "exp", "dataset_name": "d",
        "phase": 2,
        "runs": [
            {"name": "scratch", "model": "PHASE1_WINNER"},
            {"name": "external", "model": "/external/best.pt"},
            {"name": "literal", "model": "yolo11n.pt"},   # untouched
        ],
    }))
    out = RP._patch_phase2_winner(
        yaml_path, {"name": "yolo11n"}, tmp_path,
    )
    raw = yaml.safe_load(out.read_text())
    runs = {r["name"]: r["model"] for r in raw["runs"]}
    assert runs["scratch"]  == "phase1/yolo11n/weights/best.pt"
    assert runs["external"] == "/external/best.pt"
    assert runs["literal"]  == "yolo11n.pt"


def test_patch_phase3_winners_substitutes_both(tmp_path: Path):
    yaml_path = tmp_path / "phase3.yaml"
    yaml_path.write_text(
        "experiment_name: exp\n"
        "runs:\n"
        "  - name:  WINNER1_aug_default\n"
        "    model: phase2/WINNER1/weights/best.pt\n"
        "  - name:  WINNER2_aug_default\n"
        "    model: phase2/WINNER2/weights/best.pt\n"
    )
    top2 = [{"name": "w2_scratch"}, {"name": "combined_finetune"}]
    out = RP._patch_phase3_winners(yaml_path, top2, tmp_path)
    text = out.read_text()
    assert "WINNER1" not in text
    assert "WINNER2" not in text
    assert "phase2/w2_scratch/weights/best.pt" in text
    assert "phase2/combined_finetune/weights/best.pt" in text
    assert "w2_scratch_aug_default" in text


def test_patch_phase3_winners_single_winner(tmp_path: Path):
    """If phase 2 produced only one winner, WINNER2 falls back to WINNER1."""
    yaml_path = tmp_path / "phase3.yaml"
    yaml_path.write_text(
        "runs:\n"
        "  - name:  WINNER1_a\n"
        "    model: phase2/WINNER1/weights/best.pt\n"
        "  - name:  WINNER2_a\n"
        "    model: phase2/WINNER2/weights/best.pt\n"
    )
    out = RP._patch_phase3_winners(yaml_path, [{"name": "only_one"}], tmp_path)
    text = out.read_text()
    assert text.count("phase2/only_one/weights/best.pt") == 2


# ── _copy_final_winner ──────────────────────────────────────────────────

def test_copy_final_winner_writes_source_json(tmp_path: Path):
    exp_dir = tmp_path / "exp"
    src = tmp_path / "src" / "weights" / "best.pt"
    src.parent.mkdir(parents=True)
    src.write_text("(fake)")
    winner = {"name": "ww", "weights_path": str(src), "score": 0.7}
    dst = RP._copy_final_winner(exp_dir, phase=3, winner=winner)

    assert dst.exists()
    src_meta = json.loads((exp_dir / "final" / "source.json").read_text())
    assert src_meta["phase"] == 3
    assert src_meta["run_name"] == "ww"
    assert src_meta["score"] == 0.7
