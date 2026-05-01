#!/usr/bin/env python3
"""
tools/init_chamber_workspace.py
───────────────────────────────
Bootstrap the central workspace for one chamber type.

Layout produced (per ``improve-plan.md`` §1):

    {workspace_root}/{chamber_type}/
        workspace.yaml              ← from configs/workspace_template.yaml
        sources.yaml                ← empty registry, chambers added later
        clips/                      ← immutable per-clip inventory
        annotations/                ← YOLO txt mirroring clips/
        manifests/                  ← all_clips.csv lives here
        datasets/                   ← recipe-driven dataset views
        models/                     ← experiments + lineage index

Usage
-----
    python tools/init_chamber_workspace.py \
        --workspace-root /media/ssd/avistrack \
        --chamber-type collective

    python tools/init_chamber_workspace.py \
        --workspace-root /media/ssd/avistrack \
        --chamber-type vr \
        --n-subjects 1 --fps 30

The created ``workspace.yaml`` is validated by
``avistrack.config.load_workspace`` before we exit, so any template
breakage surfaces immediately.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT     = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "configs"
WORKSPACE_TEMPLATE = TEMPLATES_DIR / "workspace_template.yaml"
SOURCES_TEMPLATE   = TEMPLATES_DIR / "chamber_type_sources_template.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from avistrack.config import load_sources, load_workspace  # noqa: E402


WORKSPACE_SUBDIRS = ("clips", "frames", "annotations", "manifests", "datasets", "models")


def _refuse_inside_repo(workspace_root: Path) -> None:
    """Mirror loader's git-repo guard so users get the error before we write."""
    try:
        resolved = workspace_root.expanduser().resolve()
    except OSError:
        return
    for parent in [resolved, *resolved.parents]:
        if (parent / ".git").exists():
            raise SystemExit(
                f"refusing to bootstrap inside git repo {parent} "
                f"(workspace_root={resolved}). Pick a path outside the code repo."
            )


def _render_workspace_yaml(chamber_type: str, n_subjects: int, fps: int,
                           target_size: list[int], timezone: str) -> str:
    """Customise workspace_template.yaml for this chamber type."""
    raw = yaml.safe_load(WORKSPACE_TEMPLATE.read_text())
    raw["chamber_type"] = chamber_type
    raw.setdefault("chamber", {})
    raw["chamber"]["n_subjects"] = n_subjects
    raw["chamber"]["fps"] = fps
    raw["chamber"]["target_size"] = list(target_size)
    raw.setdefault("time", {})
    raw["time"]["timezone"] = timezone
    return yaml.safe_dump(raw, sort_keys=False)


def _render_sources_yaml(chamber_type: str) -> str:
    """Empty sources.yaml: chamber_type set, no chambers registered yet."""
    return yaml.safe_dump(
        {"chamber_type": chamber_type, "chambers": []},
        sort_keys=False,
    )


def init_workspace(
    workspace_root: Path,
    chamber_type: str,
    n_subjects: int,
    fps: int,
    target_size: list[int],
    timezone: str,
    force: bool,
) -> Path:
    """Create the directory tree + yaml files. Returns the chamber-type dir."""
    _refuse_inside_repo(workspace_root)
    if not WORKSPACE_TEMPLATE.exists() or not SOURCES_TEMPLATE.exists():
        raise SystemExit(
            "missing yaml templates under configs/ – expected "
            f"{WORKSPACE_TEMPLATE.name} and {SOURCES_TEMPLATE.name}"
        )

    chamber_dir = workspace_root / chamber_type
    workspace_yaml = chamber_dir / "workspace.yaml"
    sources_yaml   = chamber_dir / "sources.yaml"

    if (workspace_yaml.exists() or sources_yaml.exists()) and not force:
        raise SystemExit(
            f"{chamber_dir} already initialised (workspace.yaml or sources.yaml "
            f"exists). Re-run with --force to overwrite."
        )

    chamber_dir.mkdir(parents=True, exist_ok=True)
    for sub in WORKSPACE_SUBDIRS:
        (chamber_dir / sub).mkdir(exist_ok=True)

    workspace_yaml.write_text(_render_workspace_yaml(
        chamber_type=chamber_type,
        n_subjects=n_subjects,
        fps=fps,
        target_size=target_size,
        timezone=timezone,
    ))
    sources_yaml.write_text(_render_sources_yaml(chamber_type))

    # Validate by round-tripping through the loaders.
    load_workspace(workspace_yaml, workspace_root=workspace_root)
    load_sources(sources_yaml, workspace_root=workspace_root, probe=False)

    return chamber_dir


def _print_tree(chamber_dir: Path) -> None:
    print(f"  {chamber_dir}/")
    print(f"    workspace.yaml")
    print(f"    sources.yaml")
    for sub in WORKSPACE_SUBDIRS:
        print(f"    {sub}/")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace-root", required=True, type=Path,
                   help="Central workspace root (always-online external SSD).")
    p.add_argument("--chamber-type", required=True,
                   help="e.g. collective, vr")
    p.add_argument("--n-subjects", type=int, default=9,
                   help="Animals per chamber (default 9)")
    p.add_argument("--fps", type=int, default=30, help="Recording fps (default 30)")
    p.add_argument("--target-size", type=int, nargs=2, default=[640, 640],
                   metavar=("W", "H"),
                   help="Inference resolution after perspective warp (default 640 640)")
    p.add_argument("--timezone", default="America/New_York",
                   help="IANA timezone (default America/New_York)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing workspace.yaml/sources.yaml.")
    args = p.parse_args()

    chamber_dir = init_workspace(
        workspace_root=args.workspace_root,
        chamber_type=args.chamber_type,
        n_subjects=args.n_subjects,
        fps=args.fps,
        target_size=args.target_size,
        timezone=args.timezone,
        force=args.force,
    )

    print(f"Initialised chamber workspace for '{args.chamber_type}':")
    _print_tree(chamber_dir)
    print()
    print("Next steps:")
    print(f"  1. Plug in a chamber drive, then register it:")
    print(f"     python tools/register_chamber_source.py \\")
    print(f"         --workspace-root {args.workspace_root} \\")
    print(f"         --chamber-type {args.chamber_type} \\")
    print(f"         --chamber-id {args.chamber_type}_<id>")
    print(f"  2. Sample clips once a wave has been registered.")


if __name__ == "__main__":
    main()
