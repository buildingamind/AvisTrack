#!/usr/bin/env python3
"""
tools/04_import_annotations.py
───────────────────────────
Import a CVAT YOLO 1.1 export into the workspace.

Two modes
---------
**CVAT-project mode** (recommended) — ingests an entire CVAT project's
zip export (one project = one annotation batch) and lays the .txt files
out flat under ``annotations/{batch_id}/``::

    {workspace}/{chamber_type}/annotations/{batch_id}/
        _meta.json       cvat_project_id/name, n_frames, n_positive,
                         n_negative, exported_at, source_zip, imported_at
        obj.names        copied from the CVAT export
        <frame_stem>.txt one per frame, flat. Frames the annotator left
                         empty (negative samples) get an empty .txt.

    {workspace}/{chamber_type}/manifests/annotation_batches.csv
                         appended with one row per import.

Frame stems are parsed against ``sources.yaml`` to determine each
frame's chamber_id + wave_id (longest-prefix match). Frames whose
matching .png is not already present under
``frames/{chamber_id}/{wave_id}/`` cause an error: this tool never
moves frames, it only writes annotations next to existing frames.

Usage::

    python tools/04_import_annotations.py \\
        --workspace-yaml /media/ssd/avistrack/vr/workspace.yaml \\
        --cvat-project-zip /tmp/cvat_export_proj6.zip \\
        --cvat-project-id 6 \\
        --cvat-project-name W3VRChamber_Final_Annotation

**Per-clip mode** (legacy, kept for backwards compatibility) imports a
single CVAT task whose images all belong to one clip::

    python tools/04_import_annotations.py \\
        --workspace-yaml /media/ssd/avistrack/collective/workspace.yaml \\
        --chamber-id collective_104A \\
        --wave-id    wave2 \\
        --clip-stem  collective_104A_wave2_Day1_Cam1_RGB_s37_transformed \\
        --source-dir /tmp/cvat_export/obj_train_data
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from avistrack.config import load_workspace  # noqa: E402
from avistrack.config.loader import load_sources  # noqa: E402
from avistrack.config.schema import SourcesConfig  # noqa: E402
from avistrack.annotations import parse_frame_name  # noqa: E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
META_FILENAME = "_meta.json"
OBJ_NAMES_FILENAME = "obj.names"

ANNOTATION_BATCHES_CSV = "annotation_batches.csv"
ANNOTATION_BATCHES_FIELDS = [
    "batch_id",
    "chamber_type",
    "cvat_project_id",
    "cvat_project_name",
    "n_frames",
    "n_positive",
    "n_negative",
    "exported_at",
    "imported_at",
    "source_zip",
]


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


# ── Import driver (per-clip, legacy) ─────────────────────────────────────

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


# ── CVAT-project mode ────────────────────────────────────────────────────

def derive_annotation_batch_id(manifests_root: Path, chamber_type: str) -> str:
    """``{chamber_type}_{YYYY-MM-DD}_batch{NN}``, NN auto-incremented from
    ``manifests/annotation_batches.csv``."""
    today = date.today().isoformat()
    prefix = f"{chamber_type}_{today}_batch"
    csv_path = manifests_root / ANNOTATION_BATCHES_CSV
    n = 1
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                bid = row.get("batch_id", "")
                if bid.startswith(prefix):
                    try:
                        nn = int(bid[len(prefix):])
                        if nn >= n:
                            n = nn + 1
                    except ValueError:
                        pass
    return f"{prefix}{n:02d}"


def append_batch_row(manifests_root: Path, row: dict) -> None:
    csv_path = manifests_root / ANNOTATION_BATCHES_CSV
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATION_BATCHES_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in ANNOTATION_BATCHES_FIELDS})


def import_cvat_project_zip(
    workspace_chamber_dir: Path,
    chamber_type:          str,
    sources:               SourcesConfig,
    cvat_project_id:       int,
    cvat_project_name:     str,
    source_zip:            Path,
    manifests_root:        Path,
    frames_root:           Path,
    annotations_root:      Path,
    *,
    dry_run:               bool = False,
    force:                 bool = False,
) -> dict:
    """
    Ingest a single CVAT YOLO 1.1 project zip as one annotation batch.

    Produces ``annotations/{batch_id}/`` flat with one .txt per frame
    (empty .txt for unannotated frames = negative samples). Frames whose
    matching .png is not already present under
    ``frames/{chamber_id}/{wave_id}/`` raise — frames are not moved.
    """
    if not source_zip.exists():
        raise SystemExit(f"--cvat-project-zip not found: {source_zip}")

    # Extract to a tmp dir under the workspace.
    tmp_extract = workspace_chamber_dir / ".import_tmp" / f"cvat_proj_{cvat_project_id}"
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract)
    extract_zip_to(source_zip, tmp_extract)

    try:
        # Collect every image (frame) and every label (annotation) in the
        # extracted tree. CVAT YOLO 1.1 puts them under obj_train_data/.
        images_by_stem: dict[str, Path] = {}
        labels_by_stem: dict[str, Path] = {}
        for p in tmp_extract.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                images_by_stem.setdefault(p.stem, p)
            elif p.suffix.lower() == ".txt" and p.name not in {
                "train.txt", "val.txt", "test.txt"
            }:
                labels_by_stem.setdefault(p.stem, p)

        if not images_by_stem:
            raise SystemExit(f"no image files found inside {source_zip}")

        # Read obj.names (used for label validation + provenance).
        classes = read_obj_names(tmp_extract)
        n_classes = len(classes) if classes else None

        # Parse + route every frame.
        per_frame: list[dict] = []
        parse_errors: list[str] = []
        missing_frames: list[str] = []
        validation_errors: list[tuple[str, list[str]]] = []
        for stem, img_path in sorted(images_by_stem.items()):
            try:
                chamber_id, wave_id, _clip = parse_frame_name(stem, sources)
            except ValueError as e:
                parse_errors.append(str(e))
                continue
            expected_frame = (frames_root / chamber_id / wave_id /
                              f"{stem}{img_path.suffix.lower()}")
            if not expected_frame.exists():
                # Try other supported extensions.
                hit = None
                for ext in IMAGE_EXTENSIONS:
                    cand = frames_root / chamber_id / wave_id / f"{stem}{ext}"
                    if cand.exists():
                        hit = cand
                        break
                if hit is None:
                    missing_frames.append(
                        f"{chamber_id}/{wave_id}/{stem} (expected under frames/)"
                    )
                    continue
            label_path = labels_by_stem.get(stem)
            label_text = ""
            if label_path is not None:
                label_text = label_path.read_text()
                issues = validate_label_text(label_text, n_classes)
                if issues:
                    validation_errors.append((stem, issues))
                    continue
            per_frame.append({
                "stem":        stem,
                "chamber_id":  chamber_id,
                "wave_id":     wave_id,
                "label_text":  label_text,    # empty string for negatives
                "has_box":     bool(label_text.strip()),
            })

        if parse_errors:
            msg = ["frame name parsing failed:"]
            msg += [f"  {e}" for e in parse_errors[:10]]
            if len(parse_errors) > 10:
                msg.append(f"  … and {len(parse_errors) - 10} more")
            raise SystemExit("\n".join(msg))
        if missing_frames:
            msg = [
                f"{len(missing_frames)} frame(s) in the export have no matching "
                f".png under {frames_root} — frames must be sampled+extracted first."
            ]
            msg += [f"  {f}" for f in missing_frames[:10]]
            if len(missing_frames) > 10:
                msg.append(f"  … and {len(missing_frames) - 10} more")
            raise SystemExit("\n".join(msg))
        if validation_errors:
            msg = ["label validation failed:"]
            for stem, issues in validation_errors[:5]:
                msg.append(f"  {stem}:")
                for it in issues[:3]:
                    msg.append(f"    - {it}")
            if len(validation_errors) > 5:
                msg.append(f"  … and {len(validation_errors) - 5} more")
            raise SystemExit("\n".join(msg))

        # Decide batch_id (auto-increment per-day).
        batch_id = derive_annotation_batch_id(manifests_root, chamber_type)
        batch_dir = annotations_root / batch_id

        # Per-chamber/wave counts for the dry-run summary.
        by_cw: dict[tuple[str, str], dict[str, int]] = {}
        for f in per_frame:
            key = (f["chamber_id"], f["wave_id"])
            entry = by_cw.setdefault(key, {"n": 0, "pos": 0, "neg": 0})
            entry["n"] += 1
            if f["has_box"]:
                entry["pos"] += 1
            else:
                entry["neg"] += 1

        n_total = len(per_frame)
        n_pos   = sum(1 for f in per_frame if f["has_box"])
        n_neg   = n_total - n_pos

        summary = {
            "batch_id":          batch_id,
            "chamber_type":      chamber_type,
            "cvat_project_id":   cvat_project_id,
            "cvat_project_name": cvat_project_name,
            "n_frames":          n_total,
            "n_positive":        n_pos,
            "n_negative":        n_neg,
            "by_chamber_wave":   by_cw,
            "batch_dir":         batch_dir,
            "dry_run":           dry_run,
        }

        if dry_run:
            return summary

        if batch_dir.exists():
            if not force:
                raise SystemExit(
                    f"{batch_dir} already exists. Re-run with --force to wipe + reimport."
                )
            shutil.rmtree(batch_dir)
        batch_dir.mkdir(parents=True)

        # Write the .txt files (empty = negative).
        for f in per_frame:
            (batch_dir / f"{f['stem']}.txt").write_text(f["label_text"])

        # Copy obj.names into the batch dir for self-containment.
        for p in tmp_extract.rglob(OBJ_NAMES_FILENAME):
            shutil.copy2(p, batch_dir / OBJ_NAMES_FILENAME)
            break

        imported_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        meta = {
            "batch_id":          batch_id,
            "chamber_type":      chamber_type,
            "cvat_project_id":   cvat_project_id,
            "cvat_project_name": cvat_project_name,
            "classes":           classes,
            "n_frames":          n_total,
            "n_positive":        n_pos,
            "n_negative":        n_neg,
            "exported_at":       "",   # filled by caller if known
            "imported_at":       imported_at,
            "source_zip":        str(source_zip.resolve()),
            "by_chamber_wave":   {f"{c}/{w}": v for (c, w), v in by_cw.items()},
        }
        (batch_dir / META_FILENAME).write_text(
            json.dumps(meta, indent=2, sort_keys=True)
        )

        append_batch_row(manifests_root, {
            "batch_id":          batch_id,
            "chamber_type":      chamber_type,
            "cvat_project_id":   str(cvat_project_id),
            "cvat_project_name": cvat_project_name,
            "n_frames":          str(n_total),
            "n_positive":        str(n_pos),
            "n_negative":        str(n_neg),
            "exported_at":       "",
            "imported_at":       imported_at,
            "source_zip":        str(source_zip.resolve()),
        })

        summary["dry_run"] = False
        return summary
    finally:
        if tmp_extract.exists():
            shutil.rmtree(tmp_extract, ignore_errors=True)


def _print_cvat_summary(summary: dict) -> None:
    tag = "DRY-RUN" if summary.get("dry_run") else "OK"
    print(f"\n[{tag}] CVAT project import")
    print(f"  batch_id          : {summary['batch_id']}")
    print(f"  chamber_type      : {summary['chamber_type']}")
    print(f"  cvat_project      : {summary['cvat_project_id']} "
          f"({summary['cvat_project_name']})")
    print(f"  total frames      : {summary['n_frames']}")
    print(f"    positive (boxed): {summary['n_positive']}")
    print(f"    negative (empty): {summary['n_negative']}")
    print(f"  per chamber/wave  :")
    for (c, w), v in sorted(summary["by_chamber_wave"].items()):
        print(f"    {c}/{w}: total={v['n']} pos={v['pos']} neg={v['neg']}")
    print(f"  → {summary['batch_dir']}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace-yaml", required=True, type=Path)

    # CVAT-project-mode flags (preferred). Mutually exclusive with the
    # legacy per-clip flags below.
    p.add_argument("--cvat-project-zip", type=Path, default=None,
                   help="CVAT YOLO 1.1 project export zip — entire project "
                        "becomes one annotation batch")
    p.add_argument("--cvat-project-id", type=int, default=None,
                   help="CVAT project id (recorded in _meta.json)")
    p.add_argument("--cvat-project-name", default=None,
                   help="CVAT project name (recorded in _meta.json)")
    p.add_argument("--sources-yaml", type=Path, default=None,
                   help="sources.yaml (default: sibling of workspace.yaml)")
    p.add_argument("--exported-at", default=None,
                   help="ISO 8601 timestamp of when the zip was exported "
                        "(optional, recorded in _meta.json)")
    p.add_argument("--dry-run", action="store_true",
                   help="Parse + validate the zip, print what would be written, "
                        "but do not touch the workspace")

    # Legacy per-clip flags.
    p.add_argument("--chamber-id", default=None)
    p.add_argument("--wave-id",    default=None)
    p.add_argument("--clip-stem",  default=None,
                   help="Workspace clip stem (matches filename in clips/{chamber}/{wave}/)")
    p.add_argument("--source-dir", type=Path, default=None,
                   help="Already-extracted folder of image+txt pairs")
    p.add_argument("--zip", type=Path, default=None,
                   help="Path to a CVAT task export .zip; will be extracted to a temp dir")

    p.add_argument("--move", action="store_true",
                   help="Move files instead of copying (per-clip mode only)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing batch dir / clip dir")

    args = p.parse_args()

    cvat_mode = args.cvat_project_zip is not None
    legacy_mode = any([args.chamber_id, args.wave_id, args.clip_stem,
                       args.source_dir, args.zip])
    if cvat_mode and legacy_mode:
        raise SystemExit(
            "--cvat-project-zip cannot be combined with per-clip flags "
            "(--chamber-id/--wave-id/--clip-stem/--source-dir/--zip)"
        )
    if not cvat_mode and not legacy_mode:
        raise SystemExit(
            "either --cvat-project-zip (CVAT-project mode) or per-clip flags "
            "are required (see --help)"
        )

    workspace = load_workspace(args.workspace_yaml)
    workspace_chamber_dir = Path(workspace.workspace.root)

    if cvat_mode:
        if args.cvat_project_id is None or args.cvat_project_name is None:
            raise SystemExit(
                "--cvat-project-id and --cvat-project-name are required with "
                "--cvat-project-zip"
            )
        sources_yaml = args.sources_yaml or args.workspace_yaml.with_name("sources.yaml")
        if not sources_yaml.exists():
            raise SystemExit(f"sources.yaml not found at {sources_yaml}")
        sources = load_sources(sources_yaml,
                               workspace_root=workspace_chamber_dir.parent,
                               probe=False)
        if sources.chamber_type != workspace.chamber_type:
            raise SystemExit(
                f"chamber_type mismatch: workspace={workspace.chamber_type!r} "
                f"sources={sources.chamber_type!r}"
            )

        manifests_root   = Path(workspace.workspace.manifests)
        frames_root      = Path(workspace.workspace.frames or
                                (workspace_chamber_dir / "frames"))
        annotations_root = Path(workspace.workspace.annotations)

        summary = import_cvat_project_zip(
            workspace_chamber_dir=workspace_chamber_dir,
            chamber_type=workspace.chamber_type,
            sources=sources,
            cvat_project_id=args.cvat_project_id,
            cvat_project_name=args.cvat_project_name,
            source_zip=args.cvat_project_zip,
            manifests_root=manifests_root,
            frames_root=frames_root,
            annotations_root=annotations_root,
            dry_run=args.dry_run,
            force=args.force,
        )

        # If user supplied --exported-at, patch the meta files post hoc.
        if args.exported_at and not args.dry_run:
            meta_path = summary["batch_dir"] / META_FILENAME
            meta = json.loads(meta_path.read_text())
            meta["exported_at"] = args.exported_at
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
            # Also patch the manifest row (rewrite the csv).
            csv_path = manifests_root / ANNOTATION_BATCHES_CSV
            rows = list(csv.DictReader(open(csv_path, newline="")))
            for r in rows:
                if r.get("batch_id") == summary["batch_id"]:
                    r["exported_at"] = args.exported_at
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=ANNOTATION_BATCHES_FIELDS)
                w.writeheader()
                w.writerows({k: r.get(k, "") for k in ANNOTATION_BATCHES_FIELDS} for r in rows)

        _print_cvat_summary(summary)
        return

    # ── Legacy per-clip mode ─────────────────────────────────────────────
    missing = [name for name, val in (
        ("--chamber-id", args.chamber_id),
        ("--wave-id",    args.wave_id),
        ("--clip-stem",  args.clip_stem),
    ) if not val]
    if missing:
        raise SystemExit(f"per-clip mode requires {', '.join(missing)}")
    if not (args.source_dir or args.zip):
        raise SystemExit("per-clip mode requires --source-dir or --zip")
    if args.source_dir and args.zip:
        raise SystemExit("--source-dir and --zip are mutually exclusive")

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

    print(f"[OK] Imported {meta['n_frames']} frame(s) for "
          f"{args.chamber_id}/{args.wave_id}/{args.clip_stem}")
    print(f"   frames      : {workspace_chamber_dir / 'frames' / args.chamber_id / args.wave_id / args.clip_stem}")
    print(f"   annotations : {workspace_chamber_dir / 'annotations' / args.chamber_id / args.wave_id / args.clip_stem}")
    if meta["classes"]:
        print(f"   classes     : {meta['classes']}")


if __name__ == "__main__":
    main()
