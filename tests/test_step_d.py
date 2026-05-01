"""
tests/test_step_d.py – recipe + list_clips + import_annotations + build_dataset.

End-to-end coverage of Step D from improve-plan.md:

* RecipeConfig schema validation
* list_clips filter logic
* import_annotations: pair discovery, label validation, file copy
* build_dataset: clip filter, frame enumeration, split, materialise

The end-to-end build test confirms two datasets with different recipes
coexist under the same workspace (verification §5 row D in improve-plan).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from avistrack.config import (
    RecipeConfig,
    load_recipe,
    load_workspace,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_tool(filename: str):
    path = REPO_ROOT / "tools" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def _bootstrap_workspace(tmp_path: Path, chamber_type: str = "collective") -> Path:
    """Reuse init_chamber_workspace.py to produce a real workspace tree."""
    init_tool = _load_tool("init_chamber_workspace.py")
    workspace_root = tmp_path / "wkspc"
    init_tool.init_workspace(
        workspace_root=workspace_root, chamber_type=chamber_type,
        n_subjects=9, fps=30, target_size=[640, 640],
        timezone="America/New_York", force=False,
    )
    return workspace_root


# ── RecipeConfig schema ─────────────────────────────────────────────────

def _write_recipe(path: Path, **overrides) -> Path:
    base = {
        "name": "full_v1",
        "chamber_type": "collective",
        "include": {"chambers": ["*"], "waves": ["*"], "layouts": ["*"]},
        "exclude": {"source_videos": [], "clip_paths": []},
        "require_annotations": True,
        "split":  {"ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                   "stratify": "chamber", "seed": 42},
        "classes": ["chick"],
    }
    base.update(overrides)
    path.write_text(yaml.safe_dump(base, sort_keys=False))
    return path


def test_recipe_template_loads(tmp_path: Path):
    template = REPO_ROOT / "configs" / "recipe_template.yaml"
    cfg = load_recipe(template)
    assert cfg.name == "full_v1"
    assert cfg.chamber_type == "collective"
    assert cfg.split.stratify == "chamber"
    assert cfg.classes == ["chick"]


def test_recipe_rejects_bad_stratify(tmp_path: Path):
    p = _write_recipe(tmp_path / "r.yaml",
                      split={"ratios": {"train": 1.0}, "stratify": "weather", "seed": 1})
    with pytest.raises(Exception):
        load_recipe(p)


def test_recipe_rejects_bad_ratios(tmp_path: Path):
    p = _write_recipe(tmp_path / "r.yaml",
                      split={"ratios": {"train": -0.1, "val": 1.1},
                             "stratify": "chamber", "seed": 1})
    with pytest.raises(Exception):
        load_recipe(p)


def test_recipe_rejects_bad_name(tmp_path: Path):
    p = _write_recipe(tmp_path / "r.yaml", name="has space")
    with pytest.raises(Exception):
        load_recipe(p)


# ── list_clips ─────────────────────────────────────────────────────────

def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _row(**overrides) -> dict:
    base = {
        "clip_path": "clips/collective_104A/wave2/foo.mp4",
        "chamber_id": "collective_104A",
        "wave_id": "wave2",
        "source_video": "Day1_RGB.mkv",
        "source_drive_uuid": "ABCD-1234",
        "layout": "structured",
        "start_sec": "10.00", "duration_sec": "3.00",
        "fps": "30.0", "sampled_at": "2026-05-01T00:00:00+00:00",
    }
    base.update(overrides)
    return base


def test_list_clips_filter_chamber_wave_layout(tmp_path: Path):
    list_tool = _load_tool("list_clips.py")
    annotations_root = tmp_path / "annotations"
    rows = [
        _row(chamber_id="collective_104A", wave_id="wave2", layout="structured",
             clip_path="clips/collective_104A/wave2/a.mp4"),
        _row(chamber_id="collective_104A", wave_id="wave1_legacy", layout="legacy",
             clip_path="clips/collective_104A/wave1_legacy/b.mp4"),
        _row(chamber_id="collective_104B", wave_id="wave2", layout="structured",
             clip_path="clips/collective_104B/wave2/c.mp4"),
    ]
    out = list_tool.filter_clips(
        rows, annotations_root,
        chambers=["collective_104A"], waves=None, layouts=None,
        source_substr=None, only_annotated=False, only_unannotated=False,
    )
    assert {r["clip_path"] for r in out} == {
        "clips/collective_104A/wave2/a.mp4",
        "clips/collective_104A/wave1_legacy/b.mp4",
    }

    out = list_tool.filter_clips(
        rows, annotations_root,
        chambers=None, waves=None, layouts=["legacy"],
        source_substr=None, only_annotated=False, only_unannotated=False,
    )
    assert {r["clip_path"] for r in out} == {
        "clips/collective_104A/wave1_legacy/b.mp4",
    }


def test_list_clips_filter_annotated(tmp_path: Path):
    list_tool = _load_tool("list_clips.py")
    annotations_root = tmp_path / "annotations"
    # Annotate only "a"
    ann_dir = annotations_root / "collective_104A" / "wave2" / "a"
    ann_dir.mkdir(parents=True)
    (ann_dir / "frame_001.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    rows = [
        _row(clip_path="clips/collective_104A/wave2/a.mp4"),
        _row(clip_path="clips/collective_104A/wave2/b.mp4"),
    ]
    annotated = list_tool.filter_clips(
        rows, annotations_root,
        chambers=None, waves=None, layouts=None, source_substr=None,
        only_annotated=True, only_unannotated=False,
    )
    assert [r["clip_path"] for r in annotated] == ["clips/collective_104A/wave2/a.mp4"]
    unannotated = list_tool.filter_clips(
        rows, annotations_root,
        chambers=None, waves=None, layouts=None, source_substr=None,
        only_annotated=False, only_unannotated=True,
    )
    assert [r["clip_path"] for r in unannotated] == ["clips/collective_104A/wave2/b.mp4"]


# ── import_annotations ─────────────────────────────────────────────────

def test_validate_label_text_good():
    imp = _load_tool("import_annotations.py")
    assert imp.validate_label_text("0 0.5 0.5 0.1 0.1\n", n_classes=1) == []


def test_validate_label_text_bad():
    imp = _load_tool("import_annotations.py")
    issues = imp.validate_label_text("0 0.5 0.5 1.5 0.1\n", n_classes=1)
    assert any("outside [0, 1]" in i for i in issues)
    issues = imp.validate_label_text("3 0.5 0.5 0.1 0.1\n", n_classes=1)
    assert any("class id 3" in i for i in issues)
    issues = imp.validate_label_text("0 0.5 0.5\n", n_classes=1)
    assert any("expected 5 fields" in i for i in issues)


def _make_cvat_export(source_dir: Path, n_frames: int = 4,
                      with_orphan: bool = False) -> Path:
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "obj.names").write_text("chick\n")
    for i in range(1, n_frames + 1):
        (source_dir / f"frame_{i:03d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (source_dir / f"frame_{i:03d}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    if with_orphan:
        (source_dir / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    return source_dir


def test_import_one_clip_end_to_end(tmp_path: Path):
    imp = _load_tool("import_annotations.py")
    workspace_root = _bootstrap_workspace(tmp_path)
    workspace_chamber_dir = workspace_root / "collective"

    src = _make_cvat_export(tmp_path / "cvat_export", n_frames=3)

    meta = imp.import_one_clip(
        workspace_chamber_dir=workspace_chamber_dir,
        chamber_id="collective_104A", wave_id="wave2",
        clip_stem="my_clip_s10", source_dir=src,
        move=False, force=False,
    )
    assert meta["n_frames"] == 3
    assert meta["classes"]  == ["chick"]

    frames = sorted((workspace_chamber_dir / "frames" / "collective_104A" /
                     "wave2" / "my_clip_s10").glob("frame_*.png"))
    labels = sorted((workspace_chamber_dir / "annotations" / "collective_104A" /
                     "wave2" / "my_clip_s10").glob("frame_*.txt"))
    assert len(frames) == 3
    assert len(labels) == 3
    meta_json = json.loads(
        (workspace_chamber_dir / "annotations" / "collective_104A" /
         "wave2" / "my_clip_s10" / "_meta.json").read_text()
    )
    assert meta_json["n_frames"] == 3


def test_import_refuses_double_without_force(tmp_path: Path):
    imp = _load_tool("import_annotations.py")
    workspace_root = _bootstrap_workspace(tmp_path)
    workspace_chamber_dir = workspace_root / "collective"
    src = _make_cvat_export(tmp_path / "cvat_export", n_frames=2)

    imp.import_one_clip(
        workspace_chamber_dir=workspace_chamber_dir,
        chamber_id="c1", wave_id="w1", clip_stem="clip_a",
        source_dir=src, move=False, force=False,
    )
    with pytest.raises(SystemExit, match="already populated"):
        imp.import_one_clip(
            workspace_chamber_dir=workspace_chamber_dir,
            chamber_id="c1", wave_id="w1", clip_stem="clip_a",
            source_dir=src, move=False, force=False,
        )
    # --force succeeds.
    imp.import_one_clip(
        workspace_chamber_dir=workspace_chamber_dir,
        chamber_id="c1", wave_id="w1", clip_stem="clip_a",
        source_dir=src, move=False, force=True,
    )


def test_import_rejects_orphan_label(tmp_path: Path):
    imp = _load_tool("import_annotations.py")
    workspace_root = _bootstrap_workspace(tmp_path)
    src = _make_cvat_export(tmp_path / "cvat_export", n_frames=2, with_orphan=True)
    with pytest.raises(SystemExit, match="without a matching image"):
        imp.import_one_clip(
            workspace_chamber_dir=workspace_root / "collective",
            chamber_id="c1", wave_id="w1", clip_stem="clip_a",
            source_dir=src, move=False, force=False,
        )


def test_import_rejects_bad_label(tmp_path: Path):
    imp = _load_tool("import_annotations.py")
    workspace_root = _bootstrap_workspace(tmp_path)
    src = tmp_path / "cvat_export"
    src.mkdir()
    (src / "frame_001.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (src / "frame_001.txt").write_text("0 0.5 0.5 1.5 0.1\n")  # bad: w out of range
    with pytest.raises(SystemExit, match="label validation failed"):
        imp.import_one_clip(
            workspace_chamber_dir=workspace_root / "collective",
            chamber_id="c1", wave_id="w1", clip_stem="clip_a",
            source_dir=src, move=False, force=False,
        )


# ── build_dataset (end-to-end) ─────────────────────────────────────────

def _seed_workspace_with_clips(workspace_chamber_dir: Path, plan: list[dict]) -> Path:
    """Populate manifests/all_clips.csv + frames/ + annotations/ for testing.

    plan items: {chamber, wave, layout, clip_stem, n_frames}
    """
    rows = []
    for spec in plan:
        chamber, wave = spec["chamber"], spec["wave"]
        clip_stem = spec["clip_stem"]
        layout    = spec.get("layout", "structured")
        n_frames  = spec["n_frames"]
        rel_clip  = f"clips/{chamber}/{wave}/{clip_stem}.mp4"

        frame_dir = workspace_chamber_dir / "frames"      / chamber / wave / clip_stem
        ann_dir   = workspace_chamber_dir / "annotations" / chamber / wave / clip_stem
        frame_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, n_frames + 1):
            (frame_dir / f"frame_{i:03d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (ann_dir   / f"frame_{i:03d}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        rows.append({
            "clip_path": rel_clip,
            "chamber_id": chamber, "wave_id": wave,
            "source_video": f"{clip_stem}.mkv",
            "source_drive_uuid": "ABCD-1234",
            "layout": layout,
            "start_sec": "10.00", "duration_sec": "3.00",
            "fps": "30.0", "sampled_at": "2026-05-01T00:00:00+00:00",
        })

    manifest = workspace_chamber_dir / "manifests" / "all_clips.csv"
    _write_manifest(manifest, rows)
    return manifest


def test_build_dataset_full_v1_then_chambers_1and2(tmp_path: Path):
    """Verification §5 row D: two recipes, two coexisting datasets."""
    build = _load_tool("build_dataset.py")
    workspace_root = _bootstrap_workspace(tmp_path)
    workspace_yaml = workspace_root / "collective" / "workspace.yaml"
    chamber_dir    = workspace_root / "collective"

    plan = [
        {"chamber": "collective_104A", "wave": "wave2",         "clip_stem": "a1", "n_frames": 6},
        {"chamber": "collective_104A", "wave": "wave1_legacy",  "clip_stem": "a2", "n_frames": 4, "layout": "legacy"},
        {"chamber": "collective_104B", "wave": "wave2",         "clip_stem": "b1", "n_frames": 5},
        {"chamber": "collective_104C", "wave": "wave2",         "clip_stem": "c1", "n_frames": 5},
    ]
    _seed_workspace_with_clips(chamber_dir, plan)

    # Recipe 1: everything (full_v1).
    full_recipe = _write_recipe(tmp_path / "full_v1.yaml",
        name="full_v1",
        split={"ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
               "stratify": "chamber", "seed": 42})
    summary = build.build(workspace_yaml=workspace_yaml,
                          recipe_path=full_recipe, force=False)
    assert summary["n_clips"]  == 4
    assert summary["n_frames"] == 6 + 4 + 5 + 5
    assert sum(summary["splits"].values()) == summary["n_frames"]

    full_dir = chamber_dir / "datasets" / "full_v1"
    assert (full_dir / "data.yaml").exists()
    assert (full_dir / "manifest.csv").exists()
    assert (full_dir / "recipe.yaml").exists()
    # Train/val/test image dirs non-empty.
    for split in ("train", "val", "test"):
        n_img = len(list((full_dir / "images" / split).iterdir()))
        n_lbl = len(list((full_dir / "labels" / split).iterdir()))
        assert n_img == n_lbl == summary["splits"][split]
    # data.yaml is Ultralytics-shaped.
    data_yaml = yaml.safe_load((full_dir / "data.yaml").read_text())
    assert data_yaml["nc"] == 1
    assert data_yaml["names"] == ["chick"]
    assert data_yaml["train"] == "images/train"

    # Recipe 2: only chambers 104A + 104B (chambers_1and2).
    second_recipe = _write_recipe(tmp_path / "ch12.yaml",
        name="chambers_1and2",
        include={"chambers": ["collective_104A", "collective_104B"],
                 "waves":    ["*"],
                 "layouts":  ["*"]},
        split={"ratios": {"train": 0.7, "val": 0.3}, "stratify": "wave", "seed": 7})
    summary2 = build.build(workspace_yaml=workspace_yaml,
                           recipe_path=second_recipe, force=False)
    assert summary2["n_clips"]  == 3
    assert summary2["n_frames"] == 6 + 4 + 5
    # No 'test' key in output, since ratios omitted it.
    assert set(summary2["splits"].keys()) == {"train", "val"}

    second_dir = chamber_dir / "datasets" / "chambers_1and2"
    assert second_dir.exists()
    # Both datasets coexist under the same workspace.
    assert full_dir.exists()
    # The chambers_1and2 dataset must not contain frames from 104C.
    rows = list(csv.DictReader((second_dir / "manifest.csv").open()))
    assert {r["chamber_id"] for r in rows} == {"collective_104A", "collective_104B"}


def test_build_dataset_refuses_overwrite(tmp_path: Path):
    build = _load_tool("build_dataset.py")
    workspace_root = _bootstrap_workspace(tmp_path)
    workspace_yaml = workspace_root / "collective" / "workspace.yaml"
    chamber_dir    = workspace_root / "collective"
    _seed_workspace_with_clips(chamber_dir, [
        {"chamber": "c", "wave": "w", "clip_stem": "x", "n_frames": 3},
    ])
    recipe = _write_recipe(tmp_path / "r.yaml", name="d1",
                           split={"ratios": {"train": 1.0}, "stratify": "none", "seed": 1})
    build.build(workspace_yaml=workspace_yaml, recipe_path=recipe, force=False)

    with pytest.raises(SystemExit, match="already exists"):
        build.build(workspace_yaml=workspace_yaml, recipe_path=recipe, force=False)

    # --force wipes and rebuilds.
    build.build(workspace_yaml=workspace_yaml, recipe_path=recipe, force=True)


def test_build_dataset_chamber_type_mismatch(tmp_path: Path):
    build = _load_tool("build_dataset.py")
    workspace_root = _bootstrap_workspace(tmp_path)
    workspace_yaml = workspace_root / "collective" / "workspace.yaml"
    chamber_dir    = workspace_root / "collective"
    _seed_workspace_with_clips(chamber_dir, [
        {"chamber": "c", "wave": "w", "clip_stem": "x", "n_frames": 3},
    ])
    recipe = _write_recipe(tmp_path / "r.yaml", chamber_type="vr")
    with pytest.raises(SystemExit, match="chamber_type"):
        build.build(workspace_yaml=workspace_yaml, recipe_path=recipe, force=False)


def test_build_dataset_skips_unannotated_clips(tmp_path: Path):
    build = _load_tool("build_dataset.py")
    workspace_root = _bootstrap_workspace(tmp_path)
    workspace_yaml = workspace_root / "collective" / "workspace.yaml"
    chamber_dir    = workspace_root / "collective"

    # Two clips: one annotated, one not.
    _seed_workspace_with_clips(chamber_dir, [
        {"chamber": "c", "wave": "w", "clip_stem": "annotated",   "n_frames": 4},
    ])
    # Manually add an unannotated clip row (no frames/annotations dirs)
    manifest = chamber_dir / "manifests" / "all_clips.csv"
    rows = list(csv.DictReader(manifest.open()))
    rows.append(_row(chamber_id="c", wave_id="w",
                     clip_path="clips/c/w/unannotated.mp4",
                     source_video="unannotated.mkv"))
    _write_manifest(manifest, rows)

    recipe = _write_recipe(tmp_path / "r.yaml", name="d1",
                           split={"ratios": {"train": 1.0}, "stratify": "none", "seed": 1})
    summary = build.build(workspace_yaml=workspace_yaml, recipe_path=recipe, force=False)
    assert summary["skipped_unannotated"] == 1
    assert summary["n_frames"] == 4
