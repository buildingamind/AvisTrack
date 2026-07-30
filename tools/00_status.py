#!/usr/bin/env python3
"""
tools/00_status.py
──────────────────
Read-only "where am I" dashboard for one chamber-type workspace. Reads the
existing manifests / annotation batches / datasets / models and prints a
one-screen summary of what has been sampled, annotated, built, and trained
per chamber x wave. Writes NOTHING.

Usage
-----
    python tools/00_status.py \\
        --workspace-yaml /media/ssd/avistrack/plus/workspace.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# A Windows console defaults to cp1252, which cannot encode the box-drawing
# and arrow characters these tools print. Without this the tool dies on the
# *summary*, after the real work has already been committed to disk.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from avistrack.config import load_workspace  # noqa: E402


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return {}


def main():
    ap = argparse.ArgumentParser(
        description="Read-only status dashboard for a chamber-type workspace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--workspace-yaml", required=True, type=Path)
    args = ap.parse_args()

    ws = load_workspace(args.workspace_yaml)
    root        = Path(ws.workspace.root)
    manifests   = Path(ws.workspace.manifests)
    annotations = Path(ws.workspace.annotations)
    datasets    = Path(ws.workspace.dataset)
    models      = Path(ws.workspace.models)

    print(f"AvisTrack workspace status — {ws.chamber_type}")
    print(f"workspace: {root}")

    # ── clips per (chamber, wave) ──────────────────────────────────────
    clips = _read_csv(manifests / "all_clips.csv")
    clips_by_cw: dict[tuple[str, str], int] = defaultdict(int)
    for r in clips:
        clips_by_cw[(r.get("chamber_id", ""), r.get("wave_id", ""))] += 1

    # ── annotated frames per (chamber, wave), from every batch's _meta ──
    ann_by_cw: dict[tuple[str, str], int] = defaultdict(int)
    batches_by_cw: dict[tuple[str, str], list[str]] = defaultdict(list)
    batch_metas: list[dict] = []
    if annotations.is_dir():
        for bdir in sorted(annotations.iterdir()):
            meta = _read_json(bdir / "_meta.json")
            if not meta:
                continue
            batch_metas.append(meta)
            for cw, v in meta.get("by_chamber_wave", {}).items():
                if "/" in cw:
                    ch, wv = cw.split("/", 1)
                else:
                    ch, wv = cw, ""
                ann_by_cw[(ch, wv)] += int(v.get("n", 0))
                batches_by_cw[(ch, wv)].append(meta.get("batch_id", bdir.name))

    all_cw = sorted(set(clips_by_cw) | set(ann_by_cw))
    print("\n── Clips & annotations (per chamber / wave) " + "─" * 18)
    print(f"  {'chamber':<12} {'wave':<8} {'clips':>6} {'annotated':>10}  batches")
    if not all_cw:
        print("  (nothing sampled yet)")
    for ch, wv in all_cw:
        b = ",".join(sorted(set(batches_by_cw.get((ch, wv), [])))) or "-"
        print(f"  {ch:<12} {wv:<8} {clips_by_cw.get((ch, wv), 0):>6} "
              f"{ann_by_cw.get((ch, wv), 0):>10}  {b}")

    # ── annotation batches ─────────────────────────────────────────────
    print("\n── Annotation batches " + "─" * 40)
    if not batch_metas:
        print("  (no annotation batches)")
    for m in sorted(batch_metas, key=lambda x: x.get("batch_id", "")):
        print(f"  {m.get('batch_id',''):<28} "
              f"{m.get('n_frames',0)} frames "
              f"({m.get('n_positive',0)} pos / {m.get('n_negative',0)} neg)  "
              f"proj{m.get('cvat_project_id','?')} {m.get('cvat_project_name','')}")

    # ── datasets ───────────────────────────────────────────────────────
    print("\n── Datasets " + "─" * 50)
    ds_dirs = [d for d in sorted(datasets.iterdir())
               if d.is_dir()] if datasets.is_dir() else []
    if not ds_dirs:
        print("  (no datasets built)")
    for d in ds_dirs:
        rows = _read_csv(d / "manifest.csv")
        if not rows:
            print(f"  {d.name:<28} (no manifest)")
            continue
        by_split: dict[str, int] = defaultdict(int)
        waves, chambers = set(), set()
        for r in rows:
            by_split[r.get("split", "?")] += 1
            waves.add(r.get("wave_id", ""))
            chambers.add(r.get("chamber_id", ""))
        splits = " / ".join(f"{s} {by_split[s]}" for s in ("train", "val", "test") if s in by_split)
        print(f"  {d.name:<28} {len(rows)} frames  {splits}")
        print(f"  {'':<28} waves: {','.join(sorted(w for w in waves if w))}  "
              f"chambers: {','.join(sorted(c for c in chambers if c))}")

    # ── models / experiments ───────────────────────────────────────────
    print("\n── Models / experiments " + "─" * 38)
    m_dirs = [d for d in sorted(models.iterdir())
              if d.is_dir()] if models.is_dir() else []
    if not m_dirs:
        print("  (no experiments)")
    for d in m_dirs:
        meta = _read_json(d / "meta.json")
        champ = _read_json(d / "final" / "champion.meta.json")
        line = f"  {d.name:<28} dataset={meta.get('dataset_name','?')}"
        if champ:
            metric = champ.get("selection_metric", "")
            val = champ.get("metrics", {}).get(metric)
            val_s = f"{val:.4f}" if isinstance(val, (int, float)) else "?"
            line += f"  champion={champ.get('champion_run','?')} ({metric}={val_s})"
        print(line)

    print()


if __name__ == "__main__":
    main()
