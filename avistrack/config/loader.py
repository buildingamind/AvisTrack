"""
Config loader: reads a YAML file, resolves {root} placeholders, and
validates the result against the Pydantic schema.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from avistrack.config.schema import AvisTrackConfig


def load_config(path: str | Path) -> AvisTrackConfig:
    """
    Load, resolve, and validate a YAML config file.

    Placeholder resolution
    ----------------------
    Any value containing ``{root}`` is replaced by the value of
    ``drive.root`` (if set).  This keeps configs DRY:

        drive:
          root: /media/woodlab/104-A/Wave3
          raw_videos: "{root}/00_raw_videos"   # → resolved at load time
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Resolve {root} placeholders
    root = raw.get("drive", {}).get("root")
    if root:
        raw = _resolve_placeholders(raw, {"root": root})

    return AvisTrackConfig.model_validate(raw)


# ── Internal ──────────────────────────────────────────────────────────────

def _resolve_placeholders(obj, replacements: dict) -> object:
    """Recursively walk the parsed YAML and substitute {key} patterns."""
    if isinstance(obj, str):
        for k, v in replacements.items():
            obj = obj.replace(f"{{{k}}}", str(v))
        return obj
    if isinstance(obj, dict):
        return {k: _resolve_placeholders(v, replacements) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_placeholders(item, replacements) for item in obj]
    return obj
