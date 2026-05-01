"""
tests/test_run_eval_workspace.py – Step E5.

Exercises the path-resolution + per-clip aggregation helpers in mode B
of ``eval/run_eval.py``. The full ``run_mode_b`` driver is not invoked
here because it imports ultralytics — a manual smoke test on a real
workspace covers that path. We instead pin down the boundary that mode B
relies on: workspace lookup, dataset-name fallback, and the manifest.csv
aggregator that produces ``per_clip_results.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from avistrack import lineage as L
from eval import run_eval as RE


# ── _resolve_workspace_yaml ─────────────────────────────────────────────

def test_resolve_workspace_yaml_with_placeholder(tmp_path: Path):
    out = RE._resolve_workspace_yaml(
        "{workspace_root}/collective/workspace.yaml",
        workspace_root=str(tmp_path),
    )
    assert out == (tmp_path / "collective" / "workspace.yaml").resolve()


def test_resolve_workspace_yaml_missing_root_raises(tmp_path: Path):
    with pytest.raises(SystemExit, match=r"\{workspace_root\}"):
        RE._resolve_workspace_yaml(
            "{workspace_root}/x.yaml", workspace_root=None
        )


def test_resolve_workspace_yaml_concrete_passthrough(tmp_path: Path):
    p = tmp_path / "workspace.yaml"
    p.write_text("dummy")
    out = RE._resolve_workspace_yaml(str(p), workspace_root=None)
    assert out == p.resolve()


# ── aggregate_per_clip ──────────────────────────────────────────────────

def _write_manifest(path: Path, rows: list[dict]) -> None:
    import csv
    fields = ["split", "chamber_id", "wave_id", "clip_stem",
              "frame_stem", "image_link", "label_link"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def test_aggregate_per_clip_groups_and_counts(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"split": "test", "chamber_id": "c1", "wave_id": "w2",
         "clip_stem": "clipA", "frame_stem": "f1"},
        {"split": "test", "chamber_id": "c1", "wave_id": "w2",
         "clip_stem": "clipA", "frame_stem": "f2"},
        {"split": "test", "chamber_id": "c2", "wave_id": "w2",
         "clip_stem": "clipB", "frame_stem": "f1"},
        {"split": "train", "chamber_id": "c1", "wave_id": "w2",
         "clip_stem": "clipC", "frame_stem": "f1"},   # different split
    ])
    rows = RE.aggregate_per_clip(manifest, split="test")
    assert len(rows) == 2
    by_stem = {r["clip_stem"]: r for r in rows}
    assert by_stem["clipA"]["n_frames"] == 2
    assert by_stem["clipA"]["chamber_id"] == "c1"
    assert by_stem["clipB"]["n_frames"] == 1
    assert by_stem["clipB"]["chamber_id"] == "c2"
    # train-only clip not counted
    assert "clipC" not in by_stem


def test_aggregate_per_clip_missing_manifest(tmp_path: Path):
    rows = RE.aggregate_per_clip(tmp_path / "no_such_manifest.csv")
    assert rows == []


def test_aggregate_per_clip_other_split(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"split": "val", "chamber_id": "c1", "wave_id": "w2",
         "clip_stem": "v1", "frame_stem": "f1"},
        {"split": "val", "chamber_id": "c1", "wave_id": "w2",
         "clip_stem": "v1", "frame_stem": "f2"},
    ])
    rows = RE.aggregate_per_clip(manifest, split="val")
    assert len(rows) == 1
    assert rows[0]["n_frames"] == 2
    assert rows[0]["split"] == "val"


# ── Mode dispatch (smoke, no ultralytics) ───────────────────────────────

def test_main_rejects_both_modes_together(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--config", "x.yaml",
        "--workspace-yaml", "y.yaml",
        "--experiment-name", "exp",
    ])
    with pytest.raises(SystemExit):
        RE.main()
    err = capsys.readouterr().err
    assert "mutually exclusive" in err


def test_main_requires_one_mode(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_eval.py"])
    with pytest.raises(SystemExit):
        RE.main()
    err = capsys.readouterr().err
    assert "either --config" in err


def test_mode_b_missing_meta_raises(tmp_path: Path):
    """Mode B paths up to ultralytics import: workspace + meta lookup."""
    ws = tmp_path / "collective"
    ws.mkdir()
    (ws / "datasets").mkdir()
    (ws / "models").mkdir()
    (ws / "workspace.yaml").write_text(yaml.safe_dump({
        "chamber_type": "collective",
        "workspace": {
            "root":     str(ws),
            "dataset":  str(ws / "datasets"),
            "models":   str(ws / "models"),
        },
        "chamber":   {"n_subjects": 9},
    }))

    class Args:
        workspace_yaml  = str(ws / "workspace.yaml")
        workspace_root  = None
        experiment_name = "doesnt_exist"
        dataset_name    = None
        weights         = None
        split           = "test"
        imgsz           = 640
        batch           = 16
        device          = "cpu"

    with pytest.raises(SystemExit, match="meta.json not found"):
        RE.run_mode_b(Args)
