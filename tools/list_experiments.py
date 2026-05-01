#!/usr/bin/env python3
"""
tools/list_experiments.py
─────────────────────────
List training experiments registered in a workspace's ``models/index.csv``.

Usage
-----
    python tools/list_experiments.py \\
        --workspace-yaml /media/wkspc/collective/workspace.yaml \\
        [--dataset-name full_v1] \\
        [--since 2026-04-01] \\
        [--sort started_at|ended_at]

Default sort is ``started_at`` descending (most recent first).
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from avistrack.config.loader import load_workspace
from avistrack import lineage as L


def _resolve_workspace_yaml(workspace_yaml: str, workspace_root) -> Path:
    if "{workspace_root}" in workspace_yaml:
        if workspace_root is None:
            raise SystemExit(
                "workspace_yaml has '{workspace_root}'; pass --workspace-root."
            )
        workspace_yaml = workspace_yaml.replace("{workspace_root}", str(workspace_root))
    return Path(workspace_yaml).expanduser().resolve()


def _read_index(index_csv: Path) -> list[dict]:
    if not index_csv.exists():
        return []
    with open(index_csv, newline="") as f:
        return list(csv.DictReader(f))


def _filter_rows(rows: list[dict], *, chamber_type, dataset_name, since) -> list[dict]:
    out = rows
    if chamber_type:
        out = [r for r in out if r.get("chamber_type") == chamber_type]
    if dataset_name:
        out = [r for r in out if r.get("dataset_name") == dataset_name]
    if since:
        try:
            cutoff = datetime.fromisoformat(since)
        except ValueError:
            raise SystemExit(f"--since must be an ISO 8601 date, got {since!r}")
        kept = []
        for r in out:
            started = r.get("started_at", "")
            if not started:
                continue
            try:
                started_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
            except ValueError:
                continue
            if started_dt.replace(tzinfo=None) >= cutoff.replace(tzinfo=None):
                kept.append(r)
        out = kept
    return out


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("(no experiments)")
        return
    cols = ["experiment_name", "chamber_type", "dataset_name",
            "started_at", "ended_at", "git_sha", "final_weights"]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep    = "  ".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace-yaml", required=True)
    p.add_argument("--workspace-root", default=None)
    p.add_argument("--chamber-type", default=None)
    p.add_argument("--dataset-name", default=None)
    p.add_argument("--since", default=None,
                   help="Only include experiments started on/after this ISO date.")
    p.add_argument("--sort", choices=["started_at", "ended_at", "experiment_name"],
                   default="started_at")
    p.add_argument("--asc", action="store_true",
                   help="Sort ascending instead of descending.")
    args = p.parse_args()

    ws_path = _resolve_workspace_yaml(args.workspace_yaml, args.workspace_root)
    workspace = load_workspace(ws_path)
    models_root = Path(workspace.workspace.models)
    index_csv = models_root / L.INDEX_FILENAME

    rows = _read_index(index_csv)
    rows = _filter_rows(
        rows,
        chamber_type=args.chamber_type,
        dataset_name=args.dataset_name,
        since=args.since,
    )
    rows.sort(key=lambda r: r.get(args.sort, ""), reverse=not args.asc)

    print(f"📂 {index_csv}  ({len(rows)} experiments after filter)\n")
    _print_table(rows)


if __name__ == "__main__":
    main()
