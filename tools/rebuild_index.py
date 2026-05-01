#!/usr/bin/env python3
"""
tools/rebuild_index.py
──────────────────────
Rebuild ``{workspace}/{chamber_type}/models/index.csv`` from the
``meta.json`` files of each experiment subdirectory. Use this when the
index has been lost, deleted, or has drifted out of sync.

Usage
-----
    python tools/rebuild_index.py \\
        --workspace-yaml /media/wkspc/collective/workspace.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from avistrack import lineage as L
from avistrack.config.loader import load_workspace


def _resolve_workspace_yaml(workspace_yaml: str, workspace_root) -> Path:
    if "{workspace_root}" in workspace_yaml:
        if workspace_root is None:
            raise SystemExit(
                "workspace_yaml has '{workspace_root}'; pass --workspace-root."
            )
        workspace_yaml = workspace_yaml.replace("{workspace_root}", str(workspace_root))
    return Path(workspace_yaml).expanduser().resolve()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace-yaml", required=True)
    p.add_argument("--workspace-root", default=None)
    args = p.parse_args()

    ws_path = _resolve_workspace_yaml(args.workspace_yaml, args.workspace_root)
    workspace = load_workspace(ws_path)
    models_root = Path(workspace.workspace.models)

    n = L.rebuild_index(models_root)
    print(f"✅ Rebuilt {models_root / L.INDEX_FILENAME} with {n} row(s).")


if __name__ == "__main__":
    main()
