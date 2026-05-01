"""
Config loaders.

Three entry points coexist during the multi-chamber storage migration:

* :func:`load_config` – legacy ``drive.root`` configs (Wave2/Wave3 style).
  Resolves ``{root}`` placeholders and returns :class:`AvisTrackConfig`.
* :func:`load_workspace` – ``workspace.yaml`` describing one chamber-type
  workspace. Resolves ``{workspace_root}`` and ``{chamber_type}``.
* :func:`load_sources` – ``sources.yaml`` registering chamber drives.
  Resolves ``{workspace_root}`` and ``{chamber_type}``, then probes each
  chamber's ``drive_uuid`` to fill in ``chamber_root`` for any drive
  currently mounted. Wave-level ``{chamber_root}`` / ``{wave_subpath}``
  resolution is also performed when the drive is online.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml

from avistrack.config.drive_probe import probe_drive_mount
from avistrack.config.schema import (
    AvisTrackConfig,
    ExperimentConfig,
    RecipeConfig,
    SourcesConfig,
    WorkspaceConfig,
)


# ── Legacy single-file config ────────────────────────────────────────────

def load_config(path: str | Path) -> AvisTrackConfig:
    """
    Load, resolve, and validate a legacy YAML config file.

    Any value containing ``{root}`` is replaced by ``drive.root``::

        drive:
          root: /media/woodlab/104-A/Wave3
          raw_videos: "{root}/00_raw_videos"
    """
    raw = _read_yaml(path)

    root = raw.get("drive", {}).get("root")
    if root:
        raw = _resolve_placeholders(raw, {"root": root})

    return AvisTrackConfig.model_validate(raw)


# ── Workspace config ─────────────────────────────────────────────────────

def load_workspace(
    path: str | Path,
    workspace_root: Optional[str | Path] = None,
) -> WorkspaceConfig:
    """
    Load a chamber-type workspace.yaml.

    The expected on-disk layout is
    ``{workspace_root}/{chamber_type}/workspace.yaml``. If
    ``workspace_root`` is not supplied, it is derived as the grandparent
    of the yaml file.

    Resolves ``{workspace_root}`` and ``{chamber_type}`` placeholders,
    then validates that the resolved root does not live inside a git repo.
    """
    path = Path(path)
    raw = _read_yaml(path)

    if workspace_root is None:
        workspace_root = path.parent.parent

    chamber_type = raw.get("chamber_type")
    if not chamber_type:
        raise ValueError(f"workspace yaml missing required field 'chamber_type': {path}")

    raw = _resolve_placeholders(raw, {
        "workspace_root": str(workspace_root),
        "chamber_type":   chamber_type,
    })

    # Second pass: {root} inside the workspace.* block refers to
    # workspace.root (e.g. ``clips: "{root}/clips"``).
    workspace_block = raw.get("workspace") or {}
    resolved_root = workspace_block.get("root")
    if resolved_root:
        raw["workspace"] = _resolve_placeholders(workspace_block, {"root": resolved_root})

    cfg = WorkspaceConfig.model_validate(raw)
    _check_workspace_not_in_repo(Path(cfg.workspace.root), source=path)
    return cfg


# ── Sources config ───────────────────────────────────────────────────────

def load_sources(
    path: str | Path,
    workspace_root: Optional[str | Path] = None,
    probe: bool = True,
) -> SourcesConfig:
    """
    Load a chamber-type sources.yaml.

    Layout: ``{workspace_root}/{chamber_type}/sources.yaml``.

    For each registered chamber, if ``probe`` is True the drive UUID is
    looked up against currently-mounted volumes; the matching mount point
    is written into the chamber's ``chamber_root`` field, and any
    ``{chamber_root}`` / ``{wave_subpath}`` placeholders in the chamber's
    waves are resolved. Chambers whose drives are offline are returned
    with ``chamber_root`` left as ``None`` – callers that need a specific
    chamber should check this and surface a clear error.
    """
    path = Path(path)
    raw = _read_yaml(path)

    if workspace_root is None:
        workspace_root = path.parent.parent

    chamber_type = raw.get("chamber_type")
    if not chamber_type:
        raise ValueError(f"sources yaml missing required field 'chamber_type': {path}")

    raw = _resolve_placeholders(raw, {
        "workspace_root": str(workspace_root),
        "chamber_type":   chamber_type,
    })

    for chamber in raw.get("chambers", []) or []:
        mount = probe_drive_mount(chamber.get("drive_uuid", "")) if probe else None
        if mount is not None:
            chamber["chamber_root"] = str(mount)

        for wave in chamber.get("waves", []) or []:
            # {wave_subpath} self-reference is resolvable without the drive,
            # so we always do it. {chamber_root} only resolves when mounted.
            replacements = {"wave_subpath": wave.get("wave_subpath", "")}
            if mount is not None:
                replacements["chamber_root"] = str(mount)
            _resolve_inplace(wave, replacements)

    return SourcesConfig.model_validate(raw)


# ── Recipe config ────────────────────────────────────────────────────────

def load_experiment(
    path: str | Path,
    workspace_root: Optional[str | Path] = None,
) -> ExperimentConfig:
    """
    Load a workspace-aware training experiment yaml.

    Resolves ``{workspace_root}`` in the ``workspace_yaml`` field so that
    ``train/run_train.py`` can hand the caller a fully-qualified path
    without the user having to bake the workspace location into the
    experiment yaml.

    ``workspace_root`` may be passed in (e.g. by tests) or supplied via
    the ``AVISTRACK_WORKSPACE_ROOT`` environment variable; if neither is
    provided and the field still contains ``{workspace_root}`` after
    parsing, validation fails so the user sees a clear error rather than
    a confusing path-not-found later.
    """
    raw = _read_yaml(path)

    if workspace_root is None:
        import os
        env = os.environ.get("AVISTRACK_WORKSPACE_ROOT")
        if env:
            workspace_root = env

    if workspace_root is not None:
        raw = _resolve_placeholders(raw, {"workspace_root": str(workspace_root)})

    return ExperimentConfig.model_validate(raw)


def load_recipe(path: str | Path) -> RecipeConfig:
    """
    Load and validate a dataset recipe.yaml.

    Recipes do not contain placeholders (their job is to describe a
    selection over the workspace inventory, not file paths), so this
    loader is a thin schema-validating wrapper.
    """
    raw = _read_yaml(path)
    return RecipeConfig.model_validate(raw)


# ── Internal helpers ─────────────────────────────────────────────────────

def _read_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _resolve_placeholders(obj, replacements: dict) -> object:
    """Recursively walk a parsed-YAML tree and substitute ``{key}`` patterns."""
    if isinstance(obj, str):
        for k, v in replacements.items():
            obj = obj.replace(f"{{{k}}}", str(v))
        return obj
    if isinstance(obj, dict):
        return {k: _resolve_placeholders(v, replacements) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_placeholders(item, replacements) for item in obj]
    return obj


def _resolve_inplace(obj: dict, replacements: dict) -> None:
    """Resolve placeholders in a dict in place (used for one-chamber updates)."""
    for k, v in list(obj.items()):
        obj[k] = _resolve_placeholders(v, replacements)


def _check_workspace_not_in_repo(workspace_root: Path, source: Path) -> None:
    """
    Refuse to load a workspace whose root resolves inside a git repository.

    Workspaces hold large clip / dataset / model artefacts and must never
    sit inside the pipeline repo. We walk up from the resolved root and
    fail if any ancestor contains a ``.git`` entry.
    """
    try:
        resolved = workspace_root.expanduser().resolve()
    except OSError:
        return

    for parent in [resolved, *resolved.parents]:
        if (parent / ".git").exists():
            raise ValueError(
                f"workspace_root {resolved!s} resolves inside a git repo at "
                f"{parent!s}. Workspaces must live outside the code repo "
                f"(loaded from {source})."
            )
