"""
tests/test_lineage.py – Step E1 lineage utilities.

Validates the side-effect-free public surface of ``avistrack.lineage``:
recipe hashing, soft git probes, ``ExperimentMeta`` round-trip,
snapshot capture, ``models/index.csv`` maintenance, and the
``trace_lineage`` reverse-lookup.

These tests do not depend on ultralytics/torch and run on a tmpdir.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from avistrack import lineage as L


# ── hash_recipe ─────────────────────────────────────────────────────────

def test_hash_recipe_stable_across_formatting(tmp_path: Path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(
        "name: r1\n"
        "chamber_type: collective\n"
        "split:\n"
        "  ratios:\n"
        "    train: 0.8\n"
        "    val: 0.1\n"
        "    test: 0.1\n"
    )
    # Same payload, reordered keys + extra blank lines.
    b.write_text(
        "split:\n"
        "  ratios: {test: 0.1, val: 0.1, train: 0.8}\n"
        "\n"
        "chamber_type: collective\n"
        "name: r1\n"
    )
    assert L.hash_recipe(a) == L.hash_recipe(b)


def test_hash_recipe_changes_with_payload(tmp_path: Path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("name: r1\nchamber_type: collective\n")
    b.write_text("name: r2\nchamber_type: collective\n")
    assert L.hash_recipe(a) != L.hash_recipe(b)


# ── git probes ──────────────────────────────────────────────────────────

def test_git_sha_in_repo(tmp_path: Path):
    # Build a tiny throwaway repo so we don't depend on the parent repo state.
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "--allow-empty", "-q", "-m", "init"],
                   cwd=tmp_path, check=True)
    sha = L.git_sha(tmp_path)
    assert sha != "unknown"
    assert len(sha) == 40
    assert L.git_dirty(tmp_path) is False


def test_git_sha_outside_repo(tmp_path: Path):
    # tmp_path itself is not a git repo and has no parent repo.
    assert L.git_sha(tmp_path) == "unknown"
    assert L.git_dirty(tmp_path) is False
    assert L.git_uncommitted_diff(tmp_path) == ""


def test_git_dirty_detects_uncommitted(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "--allow-empty", "-q", "-m", "init"],
                   cwd=tmp_path, check=True)
    (tmp_path / "x.txt").write_text("hello")
    assert L.git_dirty(tmp_path) is True
    diff = L.git_uncommitted_diff(tmp_path)
    # Untracked files don't show in `git diff` by default — that's expected.
    # Add it to the index to surface in --cached.
    subprocess.run(["git", "add", "x.txt"], cwd=tmp_path, check=True)
    diff = L.git_uncommitted_diff(tmp_path)
    assert "x.txt" in diff
    assert "staged" in diff


# ── ExperimentMeta round-trip ───────────────────────────────────────────

def _make_meta(**overrides) -> L.ExperimentMeta:
    base = dict(
        experiment_name="W2_collective_phase1",
        chamber_type="collective",
        dataset_name="full_v1",
        recipe_hash="abc123",
        git_sha="deadbeef",
        git_dirty=False,
        started_at=L.now_iso(),
        workspace_root="/tmp/wkspc",
    )
    base.update(overrides)
    return L.ExperimentMeta(**base)


def test_meta_roundtrip(tmp_path: Path):
    meta = _make_meta()
    L.write_meta(tmp_path, meta)
    got = L.read_meta(tmp_path)
    assert got == meta


def test_meta_refuses_clobber(tmp_path: Path):
    L.write_meta(tmp_path, _make_meta())
    with pytest.raises(FileExistsError):
        L.write_meta(tmp_path, _make_meta(experiment_name="other"))


def test_meta_overwrite_flag(tmp_path: Path):
    L.write_meta(tmp_path, _make_meta())
    L.write_meta(tmp_path, _make_meta(ended_at=L.now_iso()), overwrite=True)
    got = L.read_meta(tmp_path)
    assert got.ended_at is not None


def test_update_meta_partial(tmp_path: Path):
    L.write_meta(tmp_path, _make_meta())
    L.update_meta(tmp_path, ended_at="2026-05-01T00:00:00+00:00",
                  final_weights="phase3/winner/weights/best.pt")
    got = L.read_meta(tmp_path)
    assert got.ended_at == "2026-05-01T00:00:00+00:00"
    assert got.final_weights == "phase3/winner/weights/best.pt"


def test_update_meta_rejects_unknown_field(tmp_path: Path):
    L.write_meta(tmp_path, _make_meta())
    with pytest.raises(AttributeError):
        L.update_meta(tmp_path, nonexistent="x")


def test_meta_from_json_drops_unknown_keys():
    """Forward compat: unknown fields don't blow up read_meta()."""
    payload = _make_meta().to_json()
    payload["future_field"] = "xxx"
    got = L.ExperimentMeta.from_json(payload)
    assert got.experiment_name == "W2_collective_phase1"


# ── Snapshots ───────────────────────────────────────────────────────────

def test_take_snapshot_clean(tmp_path: Path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    exp_yaml    = tmp_path / "experiment.yaml"
    recipe_yaml = tmp_path / "recipe.yaml"
    data_yaml   = tmp_path / "data.yaml"
    exp_yaml.write_text("experiment_name: foo\n")
    recipe_yaml.write_text("name: r1\n")
    data_yaml.write_text("path: /x\n")

    snap = L.take_snapshot(exp_dir, exp_yaml, recipe_yaml, data_yaml,
                           repo_root=tmp_path)   # not a git repo → no diff

    assert (snap / "experiment.yaml").read_text() == "experiment_name: foo\n"
    assert (snap / "recipe.yaml").read_text() == "name: r1\n"
    assert (snap / "data.yaml").read_text() == "path: /x\n"
    assert not (snap / "uncommitted.diff").exists()


def test_take_snapshot_dirty_writes_diff(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "--allow-empty", "-q", "-m", "init"],
                   cwd=repo, check=True)
    # Make repo dirty: add+stage a file so it shows up in --cached diff.
    (repo / "tracked.txt").write_text("hello")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)

    exp_yaml    = repo / "experiment.yaml"
    recipe_yaml = repo / "recipe.yaml"
    data_yaml   = repo / "data.yaml"
    for p in (exp_yaml, recipe_yaml, data_yaml):
        p.write_text(f"# {p.name}\n")

    snap = L.take_snapshot(repo / "exp", exp_yaml, recipe_yaml, data_yaml,
                           repo_root=repo)

    diff_path = snap / "uncommitted.diff"
    assert diff_path.exists()
    assert "tracked.txt" in diff_path.read_text()


# ── index.csv ───────────────────────────────────────────────────────────

def test_append_index_creates_then_updates(tmp_path: Path):
    models = tmp_path / "models"
    meta = _make_meta()

    L.append_index(models, meta)
    rows = list(_read_csv(models / "index.csv"))
    assert len(rows) == 1
    assert rows[0]["experiment_name"] == "W2_collective_phase1"
    assert rows[0]["ended_at"] == ""

    # Same experiment_name → row updated in place, not appended.
    meta2 = _make_meta(ended_at="2026-05-01T00:00:00+00:00")
    L.append_index(models, meta2, final_weights="final/best.pt")
    rows = list(_read_csv(models / "index.csv"))
    assert len(rows) == 1
    assert rows[0]["ended_at"] == "2026-05-01T00:00:00+00:00"
    assert rows[0]["final_weights"] == "final/best.pt"

    # Different experiment_name → appended.
    L.append_index(models, _make_meta(experiment_name="other"))
    rows = list(_read_csv(models / "index.csv"))
    assert len(rows) == 2


def test_rebuild_index_from_meta_files(tmp_path: Path):
    models = tmp_path / "models"
    for name in ["expA", "expB"]:
        L.write_meta(models / name, _make_meta(experiment_name=name))
    # Drop a non-experiment dir (no meta.json) — should be skipped.
    (models / "scratch").mkdir()

    n = L.rebuild_index(models)
    assert n == 2
    rows = list(_read_csv(models / "index.csv"))
    names = sorted(r["experiment_name"] for r in rows)
    assert names == ["expA", "expB"]


def test_rebuild_index_empty_models_root(tmp_path: Path):
    models = tmp_path / "models"
    n = L.rebuild_index(models)
    assert n == 0
    assert (models / "index.csv").exists()


# ── trace_lineage ───────────────────────────────────────────────────────

def test_trace_lineage_finds_experiment_meta(tmp_path: Path):
    """A best.pt buried under phase{N}/{run}/weights/ should resolve."""
    exp_dir = tmp_path / "models" / "expA"
    L.write_meta(exp_dir, _make_meta(experiment_name="expA"))
    weights = exp_dir / "phase1" / "yolo8n" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_text("(fake)")

    out = L.trace_lineage(weights)
    assert out["kind"] == "experiment_artifact"
    assert out["experiment_meta"]["experiment_name"] == "expA"


def test_trace_lineage_dataset_dir(tmp_path: Path):
    ds = tmp_path / "datasets" / "full_v1"
    ds.mkdir(parents=True)
    (ds / "recipe.yaml").write_text(
        "name: full_v1\nchamber_type: collective\n"
    )
    with open(ds / "manifest.csv", "w") as f:
        f.write("split,chamber_id,wave_id,clip_stem\n")
        f.write("train,collective_104A,wave2,clip0001\n")
        f.write("val,collective_104B,wave2,clip0002\n")

    out = L.trace_lineage(ds)
    assert out["kind"] == "dataset"
    assert out["recipe"]["name"] == "full_v1"
    assert sorted(out["chambers"]) == ["collective_104A", "collective_104B"]
    assert out["waves"] == ["wave2"]
    assert out["recipe_hash"] == L.hash_recipe(ds / "recipe.yaml")


def test_trace_lineage_eval_config(tmp_path: Path):
    exp_dir = tmp_path / "models" / "expA"
    L.write_meta(exp_dir, _make_meta(experiment_name="expA"))
    eval_dir = exp_dir / "eval" / "full_v1"
    eval_dir.mkdir(parents=True)
    cfg = {"experiment_name": "expA", "dataset_name": "full_v1",
           "weights": "final/best.pt"}
    (eval_dir / "eval_config.yaml").write_text(yaml.safe_dump(cfg))

    out = L.trace_lineage(eval_dir / "eval_config.yaml")
    assert out["kind"] == "eval"
    assert out["experiment_name"] == "expA"
    assert out["experiment_meta"]["experiment_name"] == "expA"


def test_trace_lineage_batch_meta(tmp_path: Path):
    batch = tmp_path / "batch_runs" / "run42"
    batch.mkdir(parents=True)
    payload = {"experiment_name": "expA", "weights": "x.pt",
               "git_sha": "abc"}
    (batch / "_meta.json").write_text(json.dumps(payload))

    out = L.trace_lineage(batch / "_meta.json")
    assert out["kind"] == "batch_output"
    assert out["experiment_name"] == "expA"


def test_trace_lineage_unknown(tmp_path: Path):
    p = tmp_path / "loose.txt"
    p.write_text("nothing")
    out = L.trace_lineage(p)
    assert out["kind"] == "unknown"


# ── helpers ─────────────────────────────────────────────────────────────

def _read_csv(path: Path):
    import csv as _csv
    with open(path, newline="") as f:
        yield from _csv.DictReader(f)
