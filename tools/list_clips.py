#!/usr/bin/env python3
"""
tools/list_clips.py
───────────────────
Browse ``manifests/all_clips.csv`` of a chamber-type workspace.

The output is meant to help you author a ``recipe.yaml``: filter clips
by chamber / wave / layout / annotation status, see counts, copy-paste
selectors into the recipe.

Usage
-----
    # All clips for one chamber type
    python tools/list_clips.py \\
        --workspace-yaml /media/ssd/avistrack/collective/workspace.yaml

    # Only annotated clips on chamber 104A wave2
    python tools/list_clips.py \\
        --workspace-yaml /media/ssd/avistrack/collective/workspace.yaml \\
        --chamber collective_104A --wave wave2 --annotated

    # Pipe-friendly raw CSV
    python tools/list_clips.py --workspace-yaml ... --csv > clips.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from avistrack.config import load_workspace  # noqa: E402


def is_annotated(annotations_root: Path, chamber_id: str, wave_id: str,
                 clip_stem: str) -> bool:
    """A clip counts as annotated if its annotation dir has any .txt file."""
    d = annotations_root / chamber_id / wave_id / clip_stem
    if not d.is_dir():
        return False
    return any(p.suffix == ".txt" for p in d.iterdir())


def load_clips(all_clips_csv: Path) -> list[dict]:
    if not all_clips_csv.exists():
        return []
    with open(all_clips_csv, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def filter_clips(
    rows: list[dict],
    annotations_root: Path,
    chambers: Optional[list[str]],
    waves: Optional[list[str]],
    layouts: Optional[list[str]],
    source_substr: Optional[str],
    only_annotated: bool,
    only_unannotated: bool,
) -> list[dict]:
    out = []
    for row in rows:
        if chambers and row.get("chamber_id") not in chambers:
            continue
        if waves and row.get("wave_id") not in waves:
            continue
        if layouts and row.get("layout") not in layouts:
            continue
        if source_substr and source_substr.lower() not in row.get("source_video", "").lower():
            continue
        if only_annotated or only_unannotated:
            stem = Path(row.get("clip_path", "")).stem
            ann = is_annotated(annotations_root, row.get("chamber_id", ""),
                               row.get("wave_id", ""), stem)
            if only_annotated and not ann:
                continue
            if only_unannotated and ann:
                continue
        out.append(row)
    return out


def print_table(rows: list[dict], annotations_root: Path) -> None:
    if not rows:
        print("(no clips matched)")
        return
    cols = ["chamber_id", "wave_id", "layout", "source_video",
            "start_sec", "duration_sec", "annotated", "clip_path"]
    widths = {c: len(c) for c in cols}
    enriched = []
    for row in rows:
        stem = Path(row.get("clip_path", "")).stem
        ann = "Y" if is_annotated(annotations_root, row.get("chamber_id", ""),
                                  row.get("wave_id", ""), stem) else "."
        rec = {**row, "annotated": ann}
        for c in cols:
            widths[c] = max(widths[c], len(str(rec.get(c, ""))))
        enriched.append(rec)

    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("  ".join("-" * widths[c] for c in cols))
    for r in enriched:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    print()
    n_ann = sum(1 for r in enriched if r["annotated"] == "Y")
    print(f"Total: {len(enriched)} clip(s) ; annotated: {n_ann} ; "
          f"unannotated: {len(enriched) - n_ann}")


def print_csv(rows: list[dict]) -> None:
    if not rows:
        return
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace-yaml", required=True, type=Path)
    p.add_argument("--chamber", action="append",
                   help="Filter by chamber_id (repeatable)")
    p.add_argument("--wave", action="append",
                   help="Filter by wave_id (repeatable)")
    p.add_argument("--layout", action="append", choices=["structured", "legacy"],
                   help="Filter by layout (repeatable)")
    p.add_argument("--source", help="Substring match on source_video filename")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--annotated",   action="store_true",
                   help="Only clips that have at least one .txt label")
    g.add_argument("--unannotated", action="store_true",
                   help="Only clips with no annotations yet")
    p.add_argument("--csv", action="store_true",
                   help="Emit raw CSV on stdout instead of a table")
    args = p.parse_args()

    workspace = load_workspace(args.workspace_yaml)
    manifests_root = Path(workspace.workspace.manifests)
    annotations_root = Path(workspace.workspace.annotations)
    all_clips_csv = manifests_root / "all_clips.csv"

    rows = load_clips(all_clips_csv)
    if not rows:
        print(f"No clips in manifest {all_clips_csv}", file=sys.stderr)
        sys.exit(0)

    filtered = filter_clips(
        rows, annotations_root,
        chambers=args.chamber, waves=args.wave, layouts=args.layout,
        source_substr=args.source,
        only_annotated=args.annotated,
        only_unannotated=args.unannotated,
    )

    if args.csv:
        print_csv(filtered)
    else:
        print_table(filtered, annotations_root)


if __name__ == "__main__":
    main()
