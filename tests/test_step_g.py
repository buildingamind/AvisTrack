"""
tests/test_step_g.py – batch-output lineage + run_batch workspace mode.

Covers:

* avistrack/lineage.py – BatchMeta dataclass round-trip, write/finalize
  refusal-without-overwrite, trace_lineage from parquet → _meta.json →
  experiment meta.
* cli/run_batch.py     – workspace-mode arg validation, weights
  resolution, frozen tracker yaml schema, _meta.json gate.

cv2 / torch / yolov8 are not required: the per-video tracker pool is
not exercised here. End-to-end execution is reserved for the manual
smoke test described in improve-plan §G.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import textwrap
import types
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Stub cv2 only when missing — keeps the module-level imports under
# cli/run_batch.py loadable on bare environments.
try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cv2"] = types.ModuleType("cv2")

from avistrack import lineage as L


def _import_run_batch():
    path = REPO_ROOT / "cli" / "run_batch.py"
    spec = importlib.util.spec_from_file_location("cli_run_batch_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_batch = _import_run_batch()


# ── Workspace fixture (mirrors test_step_f.py) ────────────────────────────

WORKSPACE_YAML = """
chamber_type: collective
workspace:
  root:        "{workspace_root}/{chamber_type}"
  clips:       "{root}/clips"
  annotations: "{root}/annotations"
  manifests:   "{root}/manifests"
  dataset:     "{root}/datasets"
  models:      "{root}/models"
chamber:
  n_subjects: 9
  fps: 30
  target_size: [640, 640]
time:
  timezone:    "America/Chicago"
"""

SOURCES_YAML = """
chamber_type: collective
chambers:
  - chamber_id:  collective_104A
    drive_uuid:  "ABCD-1234"
    drive_label: "104-A"
    waves:
      - wave_id: wave2
        layout: structured
        wave_subpath: "Wave2"
        raw_videos_subpath: "{wave_subpath}/00_raw_videos"
        metadata_subpath:   "{wave_subpath}/02_Chamber_Metadata"
"""

TRACKER_YAML = """
model:
  backend: yolo
  weights: PLACEHOLDER_WILL_BE_OVERRIDDEN
chamber:
  n_subjects: 9
output:
  format: parquet
  dir:    "PLACEHOLDER_WILL_BE_OVERRIDDEN"
"""


def _bootstrap(tmp_path: Path):
    workspace_root = tmp_path / "wkspc"
    chamber_dir = workspace_root / "collective"
    chamber_dir.mkdir(parents=True)
    workspace_yaml = chamber_dir / "workspace.yaml"
    sources_yaml   = chamber_dir / "sources.yaml"
    workspace_yaml.write_text(textwrap.dedent(WORKSPACE_YAML).lstrip())
    sources_yaml.write_text(textwrap.dedent(SOURCES_YAML).lstrip())

    drive_root = tmp_path / "drive"
    raw = drive_root / "Wave2" / "00_raw_videos"
    raw.mkdir(parents=True)
    (drive_root / "Wave2" / "02_Chamber_Metadata").mkdir(parents=True)
    (raw / "Day1_Cam1_RGB.mkv").write_bytes(b"x")

    # Dummy experiment with final/best.pt
    exp_dir = chamber_dir / "models" / "W2_collective_phase3"
    (exp_dir / "final").mkdir(parents=True)
    (exp_dir / "final" / "best.pt").write_bytes(b"weights")
    L.write_meta(exp_dir, L.ExperimentMeta(
        experiment_name="W2_collective_phase3",
        chamber_type="collective",
        dataset_name="full_v1",
        recipe_hash="deadbeef",
        git_sha="abc1234",
        git_dirty=False,
        started_at="2026-04-30T10:00:00+00:00",
        workspace_root=str(workspace_root),
        ended_at="2026-04-30T18:00:00+00:00",
        final_weights=str(exp_dir / "final" / "best.pt"),
    ))

    # Tracker config yaml (legacy-shaped)
    tracker_yaml = tmp_path / "tracker.yaml"
    tracker_yaml.write_text(textwrap.dedent(TRACKER_YAML).lstrip())

    return workspace_root, workspace_yaml, sources_yaml, drive_root, exp_dir, tracker_yaml


def _patch_probe(monkeypatch, drive_root: Path) -> None:
    monkeypatch.setattr(
        "avistrack.config.loader.probe_drive_mount",
        lambda uuid: drive_root if uuid == "ABCD-1234" else None,
    )


# ══════════════════════════════════════════════════════════════════════════
# BatchMeta + lineage helpers
# ══════════════════════════════════════════════════════════════════════════

def _sample_batch_meta(**overrides) -> L.BatchMeta:
    base = dict(
        batch_run_name=  "run_X",
        experiment_name= "W2_collective_phase3",
        chamber_type=    "collective",
        chamber_id=      "collective_104A",
        wave_id=         "wave2",
        drive_uuid=      "ABCD-1234",
        weights=         "/path/to/best.pt",
        workspace_root=  "/media/ssd/avistrack",
        started_at=      "2026-05-01T12:00:00+00:00",
        git_sha=         "abc1234",
        git_dirty=       False,
        tracker_config=  "/tmp/tracker.yaml",
    )
    base.update(overrides)
    return L.BatchMeta(**base)


def test_batch_meta_roundtrip():
    m = _sample_batch_meta()
    blob = m.to_json()
    assert blob["chamber_id"] == "collective_104A"
    assert blob["ended_at"] is None
    again = L.BatchMeta.from_json(blob)
    assert again == m


def test_batch_meta_from_json_drops_unknown_keys():
    payload = _sample_batch_meta().to_json()
    payload["future_field"] = "shrug"
    again = L.BatchMeta.from_json(payload)
    assert again.chamber_id == "collective_104A"


def test_write_batch_meta_refuses_clobber(tmp_path: Path):
    L.write_batch_meta(tmp_path, _sample_batch_meta())
    with pytest.raises(FileExistsError):
        L.write_batch_meta(tmp_path, _sample_batch_meta())


def test_write_batch_meta_overwrite(tmp_path: Path):
    L.write_batch_meta(tmp_path, _sample_batch_meta())
    L.write_batch_meta(tmp_path,
                       _sample_batch_meta(batch_run_name="run_Y"),
                       overwrite=True)
    assert L.read_batch_meta(tmp_path).batch_run_name == "run_Y"


def test_finalize_batch_meta_updates_fields(tmp_path: Path):
    L.write_batch_meta(tmp_path, _sample_batch_meta())
    L.finalize_batch_meta(
        tmp_path,
        ended_at="2026-05-01T13:00:00+00:00",
        n_videos=5, n_succeeded=4, n_failed=1,
    )
    m = L.read_batch_meta(tmp_path)
    assert m.ended_at == "2026-05-01T13:00:00+00:00"
    assert (m.n_videos, m.n_succeeded, m.n_failed) == (5, 4, 1)


def test_finalize_rejects_unknown_field(tmp_path: Path):
    L.write_batch_meta(tmp_path, _sample_batch_meta())
    with pytest.raises(AttributeError):
        L.finalize_batch_meta(tmp_path, no_such_field="x")


# ══════════════════════════════════════════════════════════════════════════
# trace_lineage for batch outputs
# ══════════════════════════════════════════════════════════════════════════

def test_trace_lineage_batch_meta_direct(tmp_path: Path):
    L.write_batch_meta(tmp_path, _sample_batch_meta())
    out = L.trace_lineage(tmp_path / L.BATCH_META_FILENAME)
    assert out["kind"] == "batch_output"
    assert out["chamber_id"] == "collective_104A"
    assert out["experiment_name"] == "W2_collective_phase3"


def test_trace_lineage_parquet_finds_sibling_meta(tmp_path: Path):
    """Plan §G acceptance: show_lineage parquet → experiment + dataset."""
    workspace_root, *_ , exp_dir, _ = _bootstrap(tmp_path)
    run_dir = tmp_path / "batch_outputs" / "run_X"
    run_dir.mkdir(parents=True)
    L.write_batch_meta(run_dir, _sample_batch_meta(
        batch_run_name="run_X",
        workspace_root=str(workspace_root),
    ))
    parquet = run_dir / "video1.parquet"
    parquet.write_bytes(b"\x00")

    out = L.trace_lineage(parquet)
    assert out["kind"] == "batch_output"
    assert out["batch_meta_path"] == str(run_dir / "_meta.json")
    assert out["chamber_id"] == "collective_104A"
    # experiment_meta is chased through workspace_root + chamber_type +
    # experiment_name and lands on the per-experiment meta.json.
    assert "experiment_meta" in out
    assert out["experiment_meta"]["experiment_name"] == "W2_collective_phase3"
    assert out["experiment_meta"]["dataset_name"] == "full_v1"


def test_trace_lineage_parquet_without_meta_returns_unknown(tmp_path: Path):
    parquet = tmp_path / "orphan.parquet"
    parquet.write_bytes(b"\x00")
    out = L.trace_lineage(parquet)
    assert out["kind"] == "unknown"


def test_trace_lineage_meta_with_missing_workspace_skips_experiment(tmp_path: Path):
    """A batch meta whose workspace_root is offline still resolves cleanly."""
    L.write_batch_meta(tmp_path, _sample_batch_meta(
        workspace_root="/nonexistent/path",
    ))
    out = L.trace_lineage(tmp_path / L.BATCH_META_FILENAME)
    assert out["kind"] == "batch_output"
    assert "experiment_meta" not in out


# ══════════════════════════════════════════════════════════════════════════
# run_batch workspace helpers
# ══════════════════════════════════════════════════════════════════════════

def _ns(**kw) -> argparse.Namespace:
    defaults = dict(
        config=None, workspace_yaml=None, sources_yaml=None,
        chamber_id=None, wave_id=None, experiment_name=None,
        tracker_config=None, batch_output_dir=None, batch_run_name=None,
        modality="rgb", force_meta=False, workers=1, force=False, limit=None,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_is_workspace_args_detects_any_flag():
    assert run_batch._is_workspace_args(_ns(workspace_yaml="x")) is True
    assert run_batch._is_workspace_args(_ns(experiment_name="x")) is True
    assert run_batch._is_workspace_args(_ns()) is False
    assert run_batch._is_workspace_args(_ns(config="legacy.yaml")) is False


def test_validate_workspace_args_rejects_config_combo():
    args = _ns(
        config="legacy.yaml", workspace_yaml="w", chamber_id="c", wave_id="v",
        experiment_name="e", tracker_config="t",
        batch_output_dir="o", batch_run_name="r",
    )
    with pytest.raises(SystemExit, match="--config cannot be combined"):
        run_batch._validate_workspace_args(args)


def test_validate_workspace_args_rejects_partial_flags():
    args = _ns(workspace_yaml="w")  # all others None
    with pytest.raises(SystemExit, match="workspace mode requires"):
        run_batch._validate_workspace_args(args)


def test_resolve_workspace_weights_ok(tmp_path: Path):
    workspace_root, *_, exp_dir, _ = _bootstrap(tmp_path)
    weights = run_batch._resolve_workspace_weights(
        workspace_root, "collective", "W2_collective_phase3")
    assert weights == exp_dir / "final" / "best.pt"


def test_resolve_workspace_weights_missing(tmp_path: Path):
    workspace_root, *_ = _bootstrap(tmp_path)
    with pytest.raises(SystemExit, match="no final weights"):
        run_batch._resolve_workspace_weights(
            workspace_root, "collective", "no_such_experiment")


def test_freeze_tracker_config_overrides_drive_and_output(tmp_path: Path):
    workspace_root, *_, drive_root, exp_dir, tracker_yaml = _bootstrap(tmp_path)
    output_dir = tmp_path / "batch_out" / "run_X"
    frozen = run_batch._freeze_tracker_config(
        tracker_config_yaml=tracker_yaml,
        raw_videos=drive_root / "Wave2" / "00_raw_videos",
        roi_file=drive_root / "Wave2" / "02_Chamber_Metadata" / "camera_rois.json",
        valid_ranges=drive_root / "Wave2" / "02_Chamber_Metadata" / "valid_ranges.json",
        time_calibration=drive_root / "Wave2" / "02_Chamber_Metadata" / "time_calibration.json",
        weights=exp_dir / "final" / "best.pt",
        output_dir=output_dir,
        timezone="America/Chicago",
        target_size=[640, 640],
        output_yaml_path=output_dir / "tracker_config.yaml",
    )

    assert frozen.exists()
    raw = yaml.safe_load(frozen.read_text())
    # Drive section is fully resolved, no PLACEHOLDER strings remain.
    assert raw["drive"]["raw_videos"].endswith("Wave2/00_raw_videos")
    assert raw["drive"]["roi_file"].endswith("camera_rois.json")
    assert raw["drive"]["time_calibration"].endswith("time_calibration.json")
    # Weights replaced.
    assert raw["model"]["weights"] == str(exp_dir / "final" / "best.pt")
    # Output dir replaced.
    assert raw["output"]["dir"] == str(output_dir)
    assert raw["output"]["format"] == "parquet"
    # Timezone preserved when caller didn't override.
    assert raw["time"]["timezone"] == "America/Chicago"


def test_resolve_workspace_inputs_end_to_end(tmp_path: Path, monkeypatch):
    workspace_root, w, s, drive_root, exp_dir, tracker_yaml = _bootstrap(tmp_path)
    _patch_probe(monkeypatch, drive_root)

    args = _ns(
        workspace_yaml=str(w), chamber_id="collective_104A", wave_id="wave2",
        experiment_name="W2_collective_phase3",
        tracker_config=str(tracker_yaml),
        batch_output_dir=str(tmp_path / "batch_outputs"),
        batch_run_name="run_X",
    )
    out = run_batch._resolve_workspace_inputs(args)

    assert out["weights"] == exp_dir / "final" / "best.pt"
    assert out["output_dir"] == tmp_path / "batch_outputs" / "run_X"
    assert out["frozen_tracker"].exists()
    assert out["frozen_tracker"].name == "tracker_config.yaml"
    assert [v.name for v in out["videos"]] == ["Day1_Cam1_RGB.mkv"]
    # Initial batch meta is fully populated except ended_at + counts.
    bm = out["batch_meta"]
    assert bm.batch_run_name == "run_X"
    assert bm.experiment_name == "W2_collective_phase3"
    assert bm.chamber_id == "collective_104A"
    assert bm.drive_uuid == "ABCD-1234"
    assert bm.weights == str(exp_dir / "final" / "best.pt")
    assert bm.workspace_root == str(workspace_root)
    assert bm.ended_at is None
    assert bm.n_videos is None


def test_resolve_workspace_inputs_refuses_existing_meta(tmp_path: Path, monkeypatch):
    workspace_root, w, s, drive_root, exp_dir, tracker_yaml = _bootstrap(tmp_path)
    _patch_probe(monkeypatch, drive_root)

    output_dir = tmp_path / "batch_outputs" / "run_X"
    output_dir.mkdir(parents=True)
    (output_dir / L.BATCH_META_FILENAME).write_text("{}")

    args = _ns(
        workspace_yaml=str(w), chamber_id="collective_104A", wave_id="wave2",
        experiment_name="W2_collective_phase3",
        tracker_config=str(tracker_yaml),
        batch_output_dir=str(tmp_path / "batch_outputs"),
        batch_run_name="run_X",
    )
    with pytest.raises(SystemExit, match="already exists"):
        run_batch._resolve_workspace_inputs(args)


def test_resolve_workspace_inputs_force_meta_allows(tmp_path: Path, monkeypatch):
    workspace_root, w, s, drive_root, exp_dir, tracker_yaml = _bootstrap(tmp_path)
    _patch_probe(monkeypatch, drive_root)

    output_dir = tmp_path / "batch_outputs" / "run_X"
    output_dir.mkdir(parents=True)
    (output_dir / L.BATCH_META_FILENAME).write_text("{}")

    args = _ns(
        workspace_yaml=str(w), chamber_id="collective_104A", wave_id="wave2",
        experiment_name="W2_collective_phase3",
        tracker_config=str(tracker_yaml),
        batch_output_dir=str(tmp_path / "batch_outputs"),
        batch_run_name="run_X",
        force_meta=True,
    )
    # Should not raise.
    out = run_batch._resolve_workspace_inputs(args)
    assert out["batch_meta_path"] == output_dir / L.BATCH_META_FILENAME
