#!/usr/bin/env python3
"""
tools/ingest_cvat_export.py
───────────────────────────
Validate that a CVAT "YOLO 1.1" export, manually unzipped into
``01_Dataset_MOT_Format/<split>/annotation/<batch_id>/``, pairs cleanly with the
approved images at ``01_Dataset_MOT_Format/<split>/images/<batch_id>/``, and
update the frame manifest with an ``Annotated`` column.

Does NOT do format conversion — CVAT's YOLO 1.1 export already lays down YOLO
``.txt`` label files (one per image, ``<cls> <cx> <cy> <w> <h>`` normalized),
which Ultralytics consumes directly.

Checks performed:
  * Each approved image has a matching ``.txt`` in annotation/<batch_id>/
  * No orphan ``.txt`` files (label without a corresponding image)
  * Sample-check of label formatting (5 random files): field count, normalized range

Usage
-----
    python tools/ingest_cvat_export.py \\
        --config configs/VR/wave3_vr.yaml \\
        --split  train \\
        --batch  VR_Wave3_VR_2026-04-24_batch01
"""

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from avistrack.config.loader import load_config


SPLIT_TO_DIR = {"train": "train", "val": "val_tuning", "test": "test_golden"}
SAMPLE_CHECK_N = 5


def load_manifest_rows(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = [c.strip() for c in (reader.fieldnames or [])]
        rows = [{k.strip(): (v or "").strip() for k, v in r.items()} for r in reader]
    return fields, rows


def write_manifest(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in fields} for r in rows)


def validate_label_file(p: Path) -> list[str]:
    issues = []
    try:
        with open(p) as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    issues.append(f"{p.name}:{ln} expected 5 fields, got {len(parts)}")
                    continue
                try:
                    cls = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    issues.append(f"{p.name}:{ln} non-numeric field")
                    continue
                if cls < 0:
                    issues.append(f"{p.name}:{ln} class id negative")
                if any(c < 0 or c > 1 for c in coords):
                    issues.append(f"{p.name}:{ln} coords outside [0,1]: {coords}")
    except OSError as e:
        issues.append(f"{p.name}: read error {e}")
    return issues


def main():
    ap = argparse.ArgumentParser(description="Validate CVAT YOLO 1.1 labels and update manifest.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--batch", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset_root = Path(cfg.drive.dataset)
    metadata_root = Path(cfg.drive.metadata)
    images_dir = dataset_root / SPLIT_TO_DIR[args.split] / "images" / args.batch
    labels_dir = dataset_root / SPLIT_TO_DIR[args.split] / "annotation" / args.batch
    manifest_path = metadata_root / "frame_manifests" / args.split / f"{args.batch}.csv"

    for p, label in [(images_dir, "images"), (labels_dir, "annotations"), (manifest_path, "manifest")]:
        if not p.exists():
            print(f"❌ {label.title()} path not found: {p}")
            sys.exit(1)

    image_stems = {p.stem for p in images_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    label_stems = {p.stem for p in labels_dir.iterdir()
                   if p.is_file() and p.suffix.lower() == ".txt"}

    fields, rows = load_manifest_rows(manifest_path)
    if "Annotated" not in fields:
        fields.append("Annotated")

    approved_stems = {Path(r["Frame_Filename"]).stem
                      for r in rows if r.get("Triage_Status") == "approved"}

    missing_labels = sorted(approved_stems - label_stems)
    orphan_labels = sorted(label_stems - image_stems)

    print(f"📁 Images: {len(image_stems)} files in {images_dir}")
    print(f"🏷️  Labels: {len(label_stems)} files in {labels_dir}")
    print(f"✅ Approved per manifest: {len(approved_stems)}")

    if missing_labels:
        print(f"\n⚠️  {len(missing_labels)} approved image(s) without label:")
        for s in missing_labels[:10]:
            print(f"   {s}")
        if len(missing_labels) > 10:
            print(f"   … and {len(missing_labels) - 10} more")
    if orphan_labels:
        print(f"\n⚠️  {len(orphan_labels)} orphan label(s) (no matching image):")
        for s in orphan_labels[:10]:
            print(f"   {s}")

    # Sample format check
    sample = random.sample(sorted(label_stems), min(SAMPLE_CHECK_N, len(label_stems)))
    issues = []
    for stem in sample:
        issues += validate_label_file(labels_dir / f"{stem}.txt")
    if issues:
        print(f"\n❗ Format issues in sampled labels:")
        for line in issues[:20]:
            print(f"   {line}")
    else:
        print(f"\n✅ Sample-checked {len(sample)} label files — format OK.")

    # Update manifest
    for r in rows:
        stem = Path(r["Frame_Filename"]).stem
        r["Annotated"] = "yes" if stem in label_stems else "no"
    write_manifest(manifest_path, fields, rows)
    n_yes = sum(1 for r in rows if r.get("Annotated") == "yes")
    print(f"\n📝 Manifest updated — {n_yes}/{len(rows)} rows marked Annotated=yes.")


if __name__ == "__main__":
    main()
