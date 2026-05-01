"""
tests/test_run_train_workspace.py – Step E3.

Exercise the workspace-mode driver in ``train/run_train.py`` without
needing ultralytics or torch. ``run_training`` is monkey-patched away so
we can check what arguments YOLO would have been called with, and that
``meta.json`` / ``snapshots/`` are created exactly as the lineage
contract demands.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from avistrack import lineage as L
from train import run_train as RT


# ── Fake workspace builder ──────────────────────────────────────────────

def _make_workspace(tmp_path: Path, *, dataset: str = "full_v1",
                    chamber_type: str = "collective") -> tuple[Path, Path]:
    """Create a minimal on-disk workspace with one built dataset.

    Returns (workspace_root, workspace_yaml_path).
    """
    ws_root = tmp_path / "wkspc"
    chamber_dir = ws_root / chamber_type
    (chamber_dir / "clips").mkdir(parents=True)
    (chamber_dir / "annotations").mkdir(parents=True)
    (chamber_dir / "manifests").mkdir(parents=True)
    (chamber_dir / "datasets" / dataset).mkdir(parents=True)
    (chamber_dir / "models").mkdir(parents=True)

    workspace_yaml = chamber_dir / "workspace.yaml"
    workspace_yaml.write_text(yaml.safe_dump({
        "chamber_type": chamber_type,
        "workspace": {
            "root":        str(chamber_dir),
            "clips":       str(chamber_dir / "clips"),
            "annotations": str(chamber_dir / "annotations"),
            "manifests":   str(chamber_dir / "manifests"),
            "dataset":     str(chamber_dir / "datasets"),
            "models":      str(chamber_dir / "models"),
        },
        "chamber": {"n_subjects": 9, "fps": 30, "target_size": [640, 640]},
    }))

    ds_dir = chamber_dir / "datasets" / dataset
    (ds_dir / "data.yaml").write_text(yaml.safe_dump({
        "path": str(ds_dir), "nc": 1, "names": ["chick"],
        "train": "images/train", "val": "images/val", "test": "images/test",
    }))
    (ds_dir / "recipe.yaml").write_text(yaml.safe_dump({
        "name": dataset, "chamber_type": chamber_type,
    }))
    (ds_dir / "manifest.csv").write_text(
        "split,chamber_id,wave_id,clip_stem,frame_stem,image_link,label_link\n"
    )
    return ws_root, workspace_yaml


def _make_experiment_yaml(tmp_path: Path, ws_yaml: Path, *,
                          experiment_name: str = "W2_test",
                          dataset_name: str = "full_v1",
                          phase: int = 1,
                          runs: list[dict] | None = None,
                          chamber_type: str = "collective") -> Path:
    if runs is None:
        runs = [{"name": "yolo8n", "model": "yolov8n.pt"}]
    yaml_path = tmp_path / f"{experiment_name}_phase{phase}.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "chamber_type": chamber_type,
        "workspace_yaml": str(ws_yaml),
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "phase": phase,
        "defaults": {"epochs": 2, "imgsz": 320, "batch": 4, "device": "cpu",
                     "patience": 5, "exist_ok": True},
        "runs": runs,
    }))
    return yaml_path


# ── Monkey-patch helper ─────────────────────────────────────────────────

@pytest.fixture
def captured_runs(monkeypatch):
    """Replace run_training so we capture train_args without ultralytics."""
    captured: list[dict] = []

    def fake_run_training(train_args, dry_run):
        captured.append(dict(train_args))
        # Synthesise a best.pt so subsequent runs see "completed".
        out = Path(train_args["project"]) / train_args["name"] / "weights"
        out.mkdir(parents=True, exist_ok=True)
        (out / "best.pt").write_text("(fake)")
        return True

    monkeypatch.setattr(RT, "run_training", fake_run_training)
    return captured


# ── Tests ───────────────────────────────────────────────────────────────

def test_schema_detection_workspace_vs_legacy(tmp_path: Path):
    ws_yaml = tmp_path / "ws.yaml"
    ws_yaml.write_text("chamber_type: collective\nworkspace_yaml: x\n")
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text("output_root: /x\ndata: /y\nruns: []\n")
    assert RT._is_workspace_schema(yaml.safe_load(ws_yaml.read_text())) is True
    assert RT._is_workspace_schema(yaml.safe_load(legacy.read_text())) is False


def test_workspace_run_creates_meta_and_snapshot(tmp_path: Path, captured_runs):
    ws_root, ws_yaml = _make_workspace(tmp_path)
    exp_yaml = _make_experiment_yaml(tmp_path, ws_yaml)

    rc = RT.run_workspace_experiment(
        exp_yaml, only=None, force=False, dry_run=False,
        do_eval=False, eval_config=None, workspace_root=None,
    )
    assert rc == 0

    exp_dir = ws_root / "collective" / "models" / "W2_test"
    meta = L.read_meta(exp_dir)
    assert meta.experiment_name == "W2_test"
    assert meta.dataset_name == "full_v1"
    assert meta.recipe_hash != ""
    assert meta.started_at != ""
    assert meta.ended_at is not None
    snap = exp_dir / "snapshots"
    assert (snap / "experiment.yaml").exists()
    assert (snap / "recipe.yaml").exists()
    assert (snap / "data.yaml").exists()


def test_workspace_run_resolves_paths(tmp_path: Path, captured_runs):
    ws_root, ws_yaml = _make_workspace(tmp_path)
    exp_yaml = _make_experiment_yaml(tmp_path, ws_yaml, runs=[
        {"name": "yolo11n", "model": "yolo11n.pt", "lr0": 0.001},
    ])

    RT.run_workspace_experiment(
        exp_yaml, only=None, force=False, dry_run=False,
        do_eval=False, eval_config=None, workspace_root=None,
    )

    assert len(captured_runs) == 1
    args = captured_runs[0]
    assert args["name"]    == "yolo11n"
    assert args["model"]   == "yolo11n.pt"     # registry name passthrough
    assert args["lr0"]     == 0.001            # extra field preserved
    assert args["epochs"]  == 2
    # data + project are workspace-derived
    assert args["data"].endswith("/datasets/full_v1/data.yaml")
    assert args["project"].endswith("/models/W2_test/phase1")


def test_workspace_run_resolves_relative_phase_path(tmp_path: Path, captured_runs):
    ws_root, ws_yaml = _make_workspace(tmp_path)
    # Phase 1 already finished — drop a fake best.pt where phase 2 will look.
    exp_dir = ws_root / "collective" / "models" / "W2_test"
    phase1_w = exp_dir / "phase1" / "yolo11n" / "weights" / "best.pt"
    phase1_w.parent.mkdir(parents=True)
    phase1_w.write_text("(fake)")

    # Phase 2 references it via a workspace-relative path.
    exp_yaml = _make_experiment_yaml(tmp_path, ws_yaml, phase=2,
        runs=[{"name": "ft_phase1",
               "model": "phase1/yolo11n/weights/best.pt"}])

    RT.run_workspace_experiment(
        exp_yaml, only=None, force=False, dry_run=False,
        do_eval=False, eval_config=None, workspace_root=None,
    )
    assert captured_runs[0]["model"] == str(phase1_w)


def test_workspace_run_skips_completed(tmp_path: Path, captured_runs):
    ws_root, ws_yaml = _make_workspace(tmp_path)
    exp_yaml = _make_experiment_yaml(tmp_path, ws_yaml, runs=[
        {"name": "yolo8n", "model": "yolov8n.pt"},
        {"name": "yolo8s", "model": "yolov8s.pt"},
    ])

    # First pass: both run.
    RT.run_workspace_experiment(
        exp_yaml, only=None, force=False, dry_run=False,
        do_eval=False, eval_config=None, workspace_root=None,
    )
    assert len(captured_runs) == 2

    # Second pass: both skipped because best.pt exists.
    captured_runs.clear()
    RT.run_workspace_experiment(
        exp_yaml, only=None, force=False, dry_run=False,
        do_eval=False, eval_config=None, workspace_root=None,
    )
    assert captured_runs == []


def test_workspace_run_only_filter(tmp_path: Path, captured_runs):
    ws_root, ws_yaml = _make_workspace(tmp_path)
    exp_yaml = _make_experiment_yaml(tmp_path, ws_yaml, runs=[
        {"name": "yolo8n", "model": "yolov8n.pt"},
        {"name": "yolo8s", "model": "yolov8s.pt"},
    ])

    RT.run_workspace_experiment(
        exp_yaml, only="yolo8s", force=False, dry_run=False,
        do_eval=False, eval_config=None, workspace_root=None,
    )
    assert len(captured_runs) == 1
    assert captured_runs[0]["name"] == "yolo8s"


def test_workspace_run_rejects_dataset_swap(tmp_path: Path, captured_runs):
    ws_root, ws_yaml = _make_workspace(tmp_path)
    # Build a second dataset.
    other = ws_root / "collective" / "datasets" / "other_v1"
    other.mkdir(parents=True)
    (other / "data.yaml").write_text(yaml.safe_dump({
        "path": str(other), "nc": 1, "names": ["chick"],
        "train": "images/train", "val": "images/val", "test": "images/test",
    }))
    (other / "recipe.yaml").write_text(yaml.safe_dump({
        "name": "other_v1", "chamber_type": "collective",
    }))
    (other / "manifest.csv").write_text("split,chamber_id\n")

    # First launch: dataset = full_v1
    exp_yaml1 = _make_experiment_yaml(tmp_path, ws_yaml,
                                      dataset_name="full_v1")
    RT.run_workspace_experiment(
        exp_yaml1, only=None, force=False, dry_run=False,
        do_eval=False, eval_config=None, workspace_root=None,
    )

    # Second launch with same experiment_name, different dataset → SystemExit
    exp_yaml2 = _make_experiment_yaml(tmp_path, ws_yaml,
                                      dataset_name="other_v1")
    with pytest.raises(SystemExit, match="cannot now switch"):
        RT.run_workspace_experiment(
            exp_yaml2, only=None, force=True, dry_run=False,
            do_eval=False, eval_config=None, workspace_root=None,
        )


def test_workspace_run_dataset_missing_data_yaml(tmp_path: Path, captured_runs):
    ws_root, ws_yaml = _make_workspace(tmp_path)
    # Swap dataset_name to one that doesn't exist.
    exp_yaml = _make_experiment_yaml(tmp_path, ws_yaml,
                                     dataset_name="not_built_yet")
    with pytest.raises(SystemExit, match="data.yaml not found"):
        RT.run_workspace_experiment(
            exp_yaml, only=None, force=False, dry_run=False,
            do_eval=False, eval_config=None, workspace_root=None,
        )


def test_dry_run_doesnt_write_meta(tmp_path: Path, captured_runs):
    ws_root, ws_yaml = _make_workspace(tmp_path)
    exp_yaml = _make_experiment_yaml(tmp_path, ws_yaml)

    RT.run_workspace_experiment(
        exp_yaml, only=None, force=False, dry_run=True,
        do_eval=False, eval_config=None, workspace_root=None,
    )

    exp_dir = ws_root / "collective" / "models" / "W2_test"
    assert not (exp_dir / "meta.json").exists()
    assert not (exp_dir / "snapshots").exists()


def test_workspace_root_placeholder_resolution(tmp_path: Path, captured_runs):
    """When workspace_yaml has '{workspace_root}', --workspace-root resolves it."""
    ws_root, ws_yaml = _make_workspace(tmp_path)
    exp_yaml = tmp_path / "exp.yaml"
    exp_yaml.write_text(yaml.safe_dump({
        "chamber_type": "collective",
        # Use the placeholder; --workspace-root override will fill it.
        "workspace_yaml": "{workspace_root}/collective/workspace.yaml",
        "experiment_name": "W2_placeholder_test",
        "dataset_name": "full_v1",
        "phase": 1,
        "defaults": {"epochs": 2, "exist_ok": True},
        "runs": [{"name": "yolo8n", "model": "yolov8n.pt"}],
    }))

    rc = RT.run_workspace_experiment(
        exp_yaml, only=None, force=False, dry_run=False,
        do_eval=False, eval_config=None,
        workspace_root=str(ws_root),
    )
    assert rc == 0
    assert (ws_root / "collective" / "models" / "W2_placeholder_test"
            / "meta.json").exists()


def test_legacy_mode_still_works(tmp_path: Path, captured_runs):
    """A YAML without chamber_type takes the legacy path unchanged."""
    output_root = tmp_path / "legacy_out"
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"nc": 1, "names": ["chick"]}))
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(yaml.safe_dump({
        "output_root": str(output_root),
        "data": str(data_yaml),
        "defaults": {"epochs": 1, "exist_ok": True},
        "runs": [{"name": "yolo8n", "model": "yolov8n.pt"}],
    }))

    rc = RT.run_legacy_experiment(
        str(legacy), only=None, force=False, dry_run=False,
        do_eval=False, eval_config=None,
    )
    assert rc == 0
    assert len(captured_runs) == 1
    assert captured_runs[0]["data"] == str(data_yaml)
    assert captured_runs[0]["project"] == str(output_root)
