#!/usr/bin/env python3
"""
tools/import_annotations.py
───────────────────────────
Import a CVAT YOLO 1.1 export (or any image+label folder) for ONE clip
into the workspace.

Layout produced::

    {workspace}/{chamber_type}/
        frames/{chamber_id}/{wave_id}/{clip_stem}/<image>.png
        annotations/{chamber_id}/{wave_id}/{clip_stem}/<image>.txt
        annotations/{chamber_id}/{wave_id}/{clip_stem}/_meta.json

Each ``<image>.png`` has a sibling ``<image>.txt`` (one ``cls cx cy w h``
per detection, normalised). Labels without a matching image are an
error. ``_meta.json`` records the import (source path, n_frames,
imported_at, classes from ``obj.names`` if present).

Usage
-----
    python tools/import_annotations.py \\
        --workspace-yaml /media/ssd/avistrack/collective/workspace.yaml \\
        --chamber-id collective_104A \\
        --wave-id    wave2 \\
        --clip-stem  collective_104A_wave2_Day1_Cam1_RGB_s37_transformed \\
        --source-dir /tmp/cvat_export/obj_train_data
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from avistrack.config import load_workspace  # noqa: E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
META_FILENAME = "_meta.json"
OBJ_NAMES_FILENAME = "obj.names"


# ── Validation ───────────────────────────────────────────────────────────

def validate_label_text(text: str, n_classes: Optional[int]) -> list[str]:
    """Return a list of issue strings (empty = valid)."""
    issues = []
    for ln, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append(f"line {ln}: expected 5 fields, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            issues.append(f"line {ln}: non-numeric field")
            continue
        if cls < 0 or (n_classes is not None and cls >= n_classes):
            issues.append(f"line {ln}: class id {cls} outside [0, {n_classes})")
        if any(c < 0 or c > 1 for c in coords):
            issues.append(f"line {ln}: coords outside [0, 1]: {coords}")
    return issues


# ── Source discovery ────────────────────────────────────────────────────

def discover_pairs(source_dir: Path) -> tuple[list[Path], list[Path], list[str]]:
    """
    Walk ``source_dir`` and return (images, labels, orphans).

    A pair is matched on basename stem. ``orphans`` collects label files
    whose image is missing.
    """
    images_by_stem: dict[str, Path] = {}
    labels_by_stem: dict[str, Path] = {}
    for p in source_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            images_by_stem[p.stem] = p
        elif p.suffix.lower() == ".txt" and p.name not in {"train.txt", "val.txt", "test.txt"}:
            labels_by_stem[p.stem] = p

    images, labels, orphans = [], [], []
    for stem, label_path in sorted(labels_by_stem.items()):
        img = images_by_stem.get(stem)
        if img is None:
            orphans.append(stem)
            continue
        images.append(img)
        labels.append(label_path)
    return images, labels, orphans


def read_obj_names(source_dir: Path) -> list[str]:
    """Read CVAT's obj.names if present."""
    f = source_dir / OBJ_NAMES_FILENAME
    if not f.exists():
        # CVAT sometimes nests it inside obj_train_data/.
        for p in source_dir.rglob(OBJ_NAMES_FILENAME):
            f = p
            break
    if not f.exists():
        return []
    return [line.strip() for line in f.read_text().splitlines() if line.strip()]


# ── Source extraction (zip support) ──────────────────────────────────────

def extract_zip_to(source_zip: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_zip) as zf:
        zf.extractall(dest)
    return dest


# ── Import driver ───────────────────────────────────────────────────────

def import_one_clip(
    workspace_chamber_dir: Path,
    chamber_id: str,
    wave_id: str,
    clip_stem: str,
    source_dir: Path,
    move: bool,
    force: bool,
) -> dict:
    """
    Copy / move ``source_dir`` contents into the workspace. Returns a
    dict suitable for serialising as ``_meta.json``.

    Raises SystemExit on validation failure or refusal-to-overwrite.
    """
    if not source_dir.is_dir():
        raise SystemExit(f"source-dir not found or not a directory: {source_dir}")

    classes = read_obj_names(source_dir)
    n_classes = len(classes) if classes else None

    images, labels, orphans = discover_pairs(source_dir)
    if not labels:
        raise SystemExit(f"no .txt label files found under {source_dir}")
    if orphans:
        raise SystemExit(
            f"{len(orphans)} label file(s) without a matching image: "
            f"{orphans[:5]}{'…' if len(orphans) > 5 else ''}"
        )

    # Pre-validate every label so we never half-import.
    bad: list[tuple[str, list[str]]] = []
    for label_path in labels:
        issues = validate_label_text(label_path.read_text(), n_classes)
        if issues:
            bad.append((label_path.name, issues))
    if bad:
        msg = ["label validation failed:"]
        for name, issues in bad[:5]:
            msg.append(f"  {name}:")
            for it in issues[:3]:
                msg.append(f"    - {it}")
        if len(bad) > 5:
            msg.append(f"  … and {len(bad) - 5} more file(s) with issues")
        raise SystemExit("\n".join(msg))

    frames_dir      = workspace_chamber_dir / "frames"      / chamber_id / wave_id / clip_stem
    annotations_dir = workspace_chamber_dir / "annotations" / chamber_id / wave_id / clip_stem

    if (frames_dir.exists() and any(frames_dir.iterdir())) or \
       (annotations_dir.exists() and any(annotations_dir.iterdir())):
        if not force:
            raise SystemExit(
                f"{annotations_dir} or {frames_dir} already populated. "
                f"Re-run with --force to overwrite."
            )
        shutil.rmtree(frames_dir,      ignore_errors=True)
        shutil.rmtree(annotations_dir, ignore_errors=True)

    frames_dir.mkdir(parents=True,      exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    op = shutil.move if move else shutil.copy2
    for img, lbl in zip(images, labels):
        op(str(img), str(frames_dir      / img.name))
        op(str(lbl), str(annotations_dir / lbl.name))

    meta = {
        "chamber_id":  chamber_id,
        "wave_id":     wave_id,
        "clip_stem":   clip_stem,
        "n_frames":    len(labels),
        "classes":     classes,
        "source_dir":  str(source_dir),
        "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "operation":   "move" if move else "copy",
    }
    (annotations_dir / META_FILENAME).write_text(
        json.dumps(meta, indent=2, sort_keys=True)
    )
    return meta


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace-yaml", required=True, type=Path)
    p.add_argument("--chamber-id", required=True)
    p.add_argument("--wave-id",    required=True)
    p.add_argument("--clip-stem",  required=True,
                   help="Workspace clip stem (matches filename in clips/{chamber}/{wave}/)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--source-dir", type=Path,
                     help="Already-extracted folder of image+txt pairs")
    src.add_argument("--zip", type=Path,
                     help="Path to a CVAT export .zip; will be extracted to a temp dir")
    p.add_argument("--move", action="store_true",
                   help="Move files instead of copying")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing frames/annotations for this clip")
    args = p.parse_args()

    workspace = load_workspace(args.workspace_yaml)
    workspace_chamber_dir = Path(workspace.workspace.root)

    if args.zip:
        if not args.zip.exists():
            raise SystemExit(f"--zip not found: {args.zip}")
        tmp_extract = workspace_chamber_dir / ".import_tmp" / args.clip_stem
        if tmp_extract.exists():
            shutil.rmtree(tmp_extract)
        source_dir = extract_zip_to(args.zip, tmp_extract)
    else:
        source_dir = args.source_dir

    try:
        meta = import_one_clip(
            workspace_chamber_dir=workspace_chamber_dir,
            chamber_id=args.chamber_id,
            wave_id=args.wave_id,
            clip_stem=args.clip_stem,
            source_dir=source_dir,
            move=args.move,
            force=args.force,
        )
    finally:
        if args.zip and tmp_extract.exists():
            shutil.rmtree(tmp_extract, ignore_errors=True)

    print(f"✅ Imported {meta['n_frames']} frame(s) for "
          f"{args.chamber_id}/{args.wave_id}/{args.clip_stem}")
    print(f"   frames      : {workspace_chamber_dir / 'frames' / args.chamber_id / args.wave_id / args.clip_stem}")
    print(f"   annotations : {workspace_chamber_dir / 'annotations' / args.chamber_id / args.wave_id / args.clip_stem}")
    if meta["classes"]:
        print(f"   classes     : {meta['classes']}")


if __name__ == "__main__":
    main()
