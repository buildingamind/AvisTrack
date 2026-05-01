"""
Lineage utilities for AvisTrack training / eval / batch outputs.

Step E of ``improve-plan.md`` introduces a workspace-aware lineage chain:

    recipe.yaml  →  datasets/{name}/  →  models/{exp}/  →  eval/{name}/
                                                       ↘  batch_outputs/...

This module is the single source of truth for that bookkeeping. It is
deliberately side-effect-free at import time and has no dependency on
ultralytics or torch — callers are ``train/run_train.py``,
``train/run_pipeline.py``, ``eval/run_eval.py``, ``cli/run_batch.py``
and the ``tools/{list_experiments,show_lineage,rebuild_index}.py`` CLIs.

Public surface
--------------
* :func:`hash_recipe`           – stable SHA256 over a recipe yaml
* :func:`git_sha` /
  :func:`git_dirty` /
  :func:`git_uncommitted_diff`  – soft git probes (return ``"unknown"`` /
                                  ``False`` outside a repo)
* :class:`ExperimentMeta`       – the per-experiment ``meta.json`` shape
* :func:`write_meta` /
  :func:`read_meta`             – round-trip helpers, refuse to clobber
* :func:`take_snapshot`         – freeze experiment.yaml + recipe.yaml +
                                  data.yaml (+ uncommitted.diff) under
                                  ``{exp_dir}/snapshots/``
* :func:`append_index` /
  :func:`rebuild_index`         – maintain ``models/index.csv``
* :func:`trace_lineage`         – reverse-lookup any artifact back to its
                                  experiment + dataset + recipe
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml


# ── Recipe hashing ──────────────────────────────────────────────────────

def hash_recipe(recipe_path: str | Path) -> str:
    """SHA256 over the canonical YAML serialization of a recipe.

    The hash is stable across formatting differences (key order, comments,
    indentation) because the file is parsed and re-emitted with
    ``yaml.safe_dump(sort_keys=True)`` before hashing.
    """
    p = Path(recipe_path)
    with open(p) as f:
        data = yaml.safe_load(f)
    canonical = yaml.safe_dump(data, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── Git probes ──────────────────────────────────────────────────────────

def _git(repo_root: Optional[Path], *args: str) -> Optional[str]:
    """Run ``git ...`` returning stripped stdout, or None on any failure.

    Soft on every error path so that callers can treat 'not a repo' the
    same as 'git not installed' or 'permission denied'.
    """
    cwd = str(repo_root) if repo_root else None
    try:
        out = subprocess.check_output(
            ["git", *args], cwd=cwd, stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return out.decode("utf-8", errors="replace").strip()


def git_sha(repo_root: str | Path | None = None) -> str:
    p = Path(repo_root) if repo_root else None
    sha = _git(p, "rev-parse", "HEAD")
    return sha if sha else "unknown"


def git_dirty(repo_root: str | Path | None = None) -> bool:
    p = Path(repo_root) if repo_root else None
    out = _git(p, "status", "--porcelain")
    if out is None:
        return False
    return bool(out.strip())


def git_uncommitted_diff(repo_root: str | Path | None = None) -> str:
    """Combined staged + unstaged diff. Empty string if clean / no repo."""
    p = Path(repo_root) if repo_root else None
    staged   = _git(p, "diff", "--cached") or ""
    unstaged = _git(p, "diff") or ""
    parts = []
    if staged:
        parts.append("# === staged ===\n" + staged)
    if unstaged:
        parts.append("# === unstaged ===\n" + unstaged)
    return "\n\n".join(parts)


# ── Experiment metadata ─────────────────────────────────────────────────

@dataclass
class ExperimentMeta:
    """Per-experiment ``meta.json`` payload.

    Written once when an experiment is first launched, then updated
    in-place when it ends. Lives at ``{exp_dir}/meta.json``.
    """
    experiment_name: str
    chamber_type:    str
    dataset_name:    str
    recipe_hash:     str
    git_sha:         str
    git_dirty:       bool
    started_at:      str            # ISO 8601, UTC, second precision
    workspace_root:  str
    ended_at:        Optional[str] = None
    final_weights:   Optional[str] = None
    notes:           dict           = field(default_factory=dict)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "ExperimentMeta":
        # Forward-compatible: silently drop unknown keys.
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


META_FILENAME = "meta.json"

# Batch outputs from cli/run_batch.py share an output directory and live
# next to the parquet/MOT files they describe. Underscored so a parquet
# folder can be safely synced without confusing the experiment meta.
BATCH_META_FILENAME = "_meta.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_meta(exp_dir: str | Path, meta: ExperimentMeta,
               *, overwrite: bool = False) -> Path:
    """Write ``{exp_dir}/meta.json``.

    Refuses to clobber an existing file unless ``overwrite=True`` — this
    is the guard that enforces the 'experiment_name not allowed to be
    re-used' rule from improve-plan §1.
    """
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    target = exp_dir / META_FILENAME
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"meta.json already exists at {target}. "
            f"Refusing to overwrite — pick a unique experiment_name or "
            f"pass overwrite=True after a deliberate ended_at update."
        )
    with open(target, "w") as f:
        json.dump(meta.to_json(), f, indent=2)
    return target


def read_meta(exp_dir: str | Path) -> ExperimentMeta:
    target = Path(exp_dir) / META_FILENAME
    with open(target) as f:
        return ExperimentMeta.from_json(json.load(f))


def update_meta(exp_dir: str | Path, **fields_to_set) -> ExperimentMeta:
    """Read meta.json, merge fields, write back. Returns the updated meta."""
    meta = read_meta(exp_dir)
    for k, v in fields_to_set.items():
        if not hasattr(meta, k):
            raise AttributeError(f"ExperimentMeta has no field {k!r}")
        setattr(meta, k, v)
    write_meta(exp_dir, meta, overwrite=True)
    return meta


# ── Batch output metadata ───────────────────────────────────────────────

@dataclass
class BatchMeta:
    """Per-batch ``_meta.json`` payload written by ``cli/run_batch.py``.

    Pinning the experiment, chamber drive and git state at run time means
    any parquet under ``{batch_run_dir}/`` can be reverse-traced even
    after the workspace or repo move.

    Mandatory fields are populated when the batch starts; ``ended_at`` and
    counts are filled in when ``finalize_batch_meta`` is called at the
    end of a run.
    """
    batch_run_name:    str
    experiment_name:   str
    chamber_type:      str
    chamber_id:        str
    wave_id:           str
    drive_uuid:        str
    weights:           str
    workspace_root:    str
    started_at:        str
    git_sha:           str
    git_dirty:         bool
    tracker_config:    Optional[str] = None
    ended_at:          Optional[str] = None
    n_videos:          Optional[int] = None
    n_succeeded:       Optional[int] = None
    n_failed:          Optional[int] = None
    notes:             dict          = field(default_factory=dict)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "BatchMeta":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def write_batch_meta(
    output_dir: str | Path, meta: BatchMeta,
    *, overwrite: bool = False,
) -> Path:
    """Write ``{output_dir}/_meta.json``.

    Refuses to clobber an existing file unless ``overwrite=True``. The
    plan §G calls for run_batch.py to **refuse to start** if a previous
    ``_meta.json`` is present, which is enforced by the default.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / BATCH_META_FILENAME
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"_meta.json already exists at {target}. Refusing to overwrite — "
            f"pick a fresh batch_run_name or pass overwrite=True after a "
            f"deliberate end-of-run update."
        )
    with open(target, "w") as f:
        json.dump(meta.to_json(), f, indent=2)
    return target


def read_batch_meta(output_dir: str | Path) -> BatchMeta:
    target = Path(output_dir) / BATCH_META_FILENAME
    with open(target) as f:
        return BatchMeta.from_json(json.load(f))


def finalize_batch_meta(output_dir: str | Path, **fields_to_set) -> BatchMeta:
    """Read ``_meta.json``, merge fields, write back. Returns updated meta."""
    meta = read_batch_meta(output_dir)
    for k, v in fields_to_set.items():
        if not hasattr(meta, k):
            raise AttributeError(f"BatchMeta has no field {k!r}")
        setattr(meta, k, v)
    write_batch_meta(output_dir, meta, overwrite=True)
    return meta


# ── Snapshots ───────────────────────────────────────────────────────────

SNAPSHOT_DIR = "snapshots"


def take_snapshot(
    exp_dir: str | Path,
    experiment_yaml: str | Path,
    recipe_yaml:     str | Path,
    data_yaml:       str | Path,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    """Freeze the inputs that drove this experiment under
    ``{exp_dir}/snapshots/``.

    If git is dirty, the combined diff is captured as
    ``snapshots/uncommitted.diff`` so the exact state can be reproduced.
    Re-running is idempotent (files overwritten, no dedicated history).
    """
    snap = Path(exp_dir) / SNAPSHOT_DIR
    snap.mkdir(parents=True, exist_ok=True)

    shutil.copy2(experiment_yaml, snap / "experiment.yaml")
    shutil.copy2(recipe_yaml,     snap / "recipe.yaml")
    shutil.copy2(data_yaml,       snap / "data.yaml")

    diff = git_uncommitted_diff(repo_root)
    diff_path = snap / "uncommitted.diff"
    if diff:
        diff_path.write_text(diff)
    else:
        # Wipe a stale diff if previous run was dirty and current is clean.
        if diff_path.exists():
            diff_path.unlink()

    return snap


# ── models/index.csv ─────────────────────────────────────────────────────

INDEX_FILENAME = "index.csv"
INDEX_FIELDS = [
    "experiment_name", "chamber_type", "dataset_name",
    "recipe_hash", "git_sha", "git_dirty",
    "started_at", "ended_at", "final_weights",
]


def _meta_to_row(meta: ExperimentMeta) -> dict:
    row = {k: getattr(meta, k) for k in INDEX_FIELDS}
    row["git_dirty"] = "1" if row["git_dirty"] else "0"
    for k in row:
        if row[k] is None:
            row[k] = ""
    return row


def _read_index(models_root: Path) -> list[dict]:
    idx = models_root / INDEX_FILENAME
    if not idx.exists():
        return []
    with open(idx, newline="") as f:
        return list(csv.DictReader(f))


def _write_index(models_root: Path, rows: list[dict]) -> None:
    models_root.mkdir(parents=True, exist_ok=True)
    idx = models_root / INDEX_FILENAME
    with open(idx, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in INDEX_FIELDS})


def append_index(models_root: str | Path, meta: ExperimentMeta,
                 final_weights: str | Path | None = None) -> None:
    """Append ``meta`` to ``{models_root}/index.csv``.

    If a row with the same ``experiment_name`` already exists it is
    updated in place — important for the common pattern of writing the
    row when phase 1 starts then refreshing it when phase 3 finishes.
    """
    models_root = Path(models_root)
    if final_weights is not None:
        meta = dataclasses.replace(meta, final_weights=str(final_weights))
    rows = _read_index(models_root)
    new_row = _meta_to_row(meta)
    for i, r in enumerate(rows):
        if r.get("experiment_name") == meta.experiment_name:
            rows[i] = new_row
            break
    else:
        rows.append(new_row)
    _write_index(models_root, rows)


def rebuild_index(models_root: str | Path) -> int:
    """Rebuild ``{models_root}/index.csv`` from each ``{exp}/meta.json``.

    Returns the number of rows written. Useful when index.csv is lost or
    has drifted out of sync with the on-disk experiments.
    """
    models_root = Path(models_root)
    rows: list[dict] = []
    if not models_root.exists():
        _write_index(models_root, rows)
        return 0
    for exp_dir in sorted(p for p in models_root.iterdir() if p.is_dir()):
        meta_path = exp_dir / META_FILENAME
        if not meta_path.exists():
            continue
        try:
            meta = read_meta(exp_dir)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        rows.append(_meta_to_row(meta))
    _write_index(models_root, rows)
    return len(rows)


# ── Lineage tracing ─────────────────────────────────────────────────────

def _find_meta_upwards(start: Path) -> Optional[Path]:
    """Walk up from ``start`` looking for a directory with ``meta.json``."""
    cur = start if start.is_dir() else start.parent
    for _ in range(10):   # depth bound: workspace/models/{exp}/phase{N}/{run}/weights/best.pt
        candidate = cur / META_FILENAME
        if candidate.exists():
            return candidate
        if cur == cur.parent:
            return None
        cur = cur.parent
    return None


def _find_batch_meta_upwards(start: Path) -> Optional[Path]:
    """Walk up from ``start`` looking for a sibling ``_meta.json``.

    Used by :func:`trace_lineage` to chase parquet outputs back to the
    batch run that produced them. Bounded depth so a misplaced parquet
    that has no batch meta nearby returns None promptly.
    """
    cur = start if start.is_dir() else start.parent
    for _ in range(5):
        candidate = cur / BATCH_META_FILENAME
        if candidate.exists():
            return candidate
        if cur == cur.parent:
            return None
        cur = cur.parent
    return None


def _experiment_meta_for_batch(batch_meta_dict: dict) -> Optional[dict]:
    """Best-effort lookup of the workspace experiment_meta for a batch.

    Reads the experiment's ``meta.json`` under
    ``{workspace_root}/{chamber_type}/models/{experiment_name}/`` if both
    fields are recorded. Returns None when fields are absent or the file
    is unreachable (offline workspace, since-renamed experiment, etc.).
    """
    workspace_root = batch_meta_dict.get("workspace_root")
    chamber_type   = batch_meta_dict.get("chamber_type")
    exp_name       = batch_meta_dict.get("experiment_name")
    if not (workspace_root and chamber_type and exp_name):
        return None
    candidate = Path(workspace_root) / chamber_type / "models" / exp_name / META_FILENAME
    if not candidate.exists():
        return None
    try:
        return json.loads(candidate.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def trace_lineage(artifact: str | Path) -> dict:
    """Reverse-lookup an artifact back to its experiment + dataset + recipe.

    Recognised inputs:
      * Anything under ``{workspace}/models/{exp}/...``
        → reads ``{exp}/meta.json``
      * A batch output's sibling ``_meta.json``
        → its ``experiment_name`` field is then used to load the
        workspace meta if ``workspace_root`` is also recorded
      * An eval output's ``eval_config.yaml`` → reads ``experiment_name``
      * A dataset's ``manifest.csv`` (or its parent directory) → reports
        the recipe and chambers/waves it covers, no experiment

    Returns a dict; the ``kind`` key tells the caller which fields are
    populated.
    """
    p = Path(artifact)
    out: dict = {"artifact": str(p)}

    # Eval config (yaml in models/{exp}/eval/{ds}/eval_config.yaml)
    if p.name == "eval_config.yaml" and p.is_file():
        with open(p) as f:
            cfg = yaml.safe_load(f) or {}
        out["kind"] = "eval"
        out.update(cfg)
        # Walk up to find the experiment meta.
        meta_path = _find_meta_upwards(p.parent)
        if meta_path:
            out["experiment_meta"] = json.loads(meta_path.read_text())
        return out

    # Batch output sibling — given _meta.json directly
    if p.name == BATCH_META_FILENAME and p.is_file():
        out["kind"] = "batch_output"
        batch = json.loads(p.read_text())
        out.update(batch)
        exp_meta = _experiment_meta_for_batch(batch)
        if exp_meta is not None:
            out["experiment_meta"] = exp_meta
        return out

    # Batch output sibling — given a parquet/MOT under a batch run dir
    if p.is_file() and p.suffix.lower() in {".parquet", ".txt", ".csv"}:
        batch_meta_path = _find_batch_meta_upwards(p)
        if batch_meta_path is not None:
            out["kind"] = "batch_output"
            out["batch_meta_path"] = str(batch_meta_path)
            batch = json.loads(batch_meta_path.read_text())
            out.update(batch)
            exp_meta = _experiment_meta_for_batch(batch)
            if exp_meta is not None:
                out["experiment_meta"] = exp_meta
            return out

    # Dataset manifest or directory
    if p.name == "manifest.csv" or (p.is_dir() and (p / "manifest.csv").exists()):
        ds_dir = p.parent if p.name == "manifest.csv" else p
        out["kind"] = "dataset"
        out["dataset_dir"] = str(ds_dir)
        recipe = ds_dir / "recipe.yaml"
        if recipe.exists():
            with open(recipe) as f:
                out["recipe"] = yaml.safe_load(f)
            out["recipe_hash"] = hash_recipe(recipe)
        manifest = ds_dir / "manifest.csv"
        if manifest.exists():
            chambers, waves = set(), set()
            with open(manifest, newline="") as f:
                for row in csv.DictReader(f):
                    chambers.add(row.get("chamber_id", ""))
                    waves.add(row.get("wave_id", ""))
            out["chambers"] = sorted(c for c in chambers if c)
            out["waves"]    = sorted(w for w in waves   if w)
        return out

    # Default: walk up looking for an experiment meta.json
    meta_path = _find_meta_upwards(p)
    if meta_path:
        out["kind"] = "experiment_artifact"
        out["meta_path"] = str(meta_path)
        out["experiment_meta"] = json.loads(meta_path.read_text())
        return out

    out["kind"] = "unknown"
    return out
