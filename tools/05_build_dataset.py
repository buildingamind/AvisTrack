#!/usr/bin/env python3
"""
tools/05_build_dataset.py
──────────────────────
Materialise one ``datasets/{recipe.name}/`` view from a chamber-type
workspace's ``clips/`` + ``frames/`` (flat) + ``annotations/`` (flat-batch)
inventory.

Annotation layout (flat-batch journal)::

    {workspace}/{chamber_type}/annotations/
        {chamber_type}_{YYYY-MM-DD}_batchNN/
            _meta.json
            obj.names
            <frame_full_filename>.txt    (flat; empty .txt = negative sample)

A frame may appear in multiple batches (re-annotation). The recipe's
``annotations.resolution`` field decides which wins (``latest`` | ``first``
| ``error``).

Frame layout (flat)::

    {workspace}/{chamber_type}/frames/{chamber_id}/{wave_id}/<frame_stem>.png

Output layout (Ultralytics-compatible)::

    {workspace}/{chamber_type}/datasets/{name}/
        recipe.yaml      ← copy of the input recipe (frozen at build time)
        manifest.csv     ← per-frame: split, chamber, wave, clip, frame, batch, image, label
        data.yaml        ← Ultralytics dataset config
        images/{train,val,test}/<symlink>.png
        labels/{train,val,test}/<symlink>.txt

Usage
-----
    python tools/05_build_dataset.py \\
        --workspace-yaml /media/.../vr/workspace.yaml \\
        --recipe         configs/VR/recipe_pre-w4_vr_2026-05-02.yaml
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from avistrack.config import RecipeConfig, load_recipe, load_workspace  # noqa: E402
from avistrack.config.loader import load_sources  # noqa: E402
from avistrack.annotations import parse_frame_name  # noqa: E402

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
SPLITS = ("train", "val", "test")


# ── Filtering ────────────────────────────────────────────────────────────

def _accept(value: str, allowlist: list[str]) -> bool:
    """Return True if value is in the list, or the list is the wildcard ['*']."""
    if not allowlist or allowlist == ["*"]:
        return True
    return value in allowlist


def filter_clips(rows: list[dict], recipe: RecipeConfig) -> list[dict]:
    out = []
    excluded_videos = set(recipe.exclude.source_videos)
    excluded_paths  = set(recipe.exclude.clip_paths)
    for row in rows:
        if not _accept(row.get("chamber_id", ""), recipe.include.chambers):
            continue
        if not _accept(row.get("wave_id", ""), recipe.include.waves):
            continue
        if not _accept(row.get("layout", ""), recipe.include.layouts):
            continue
        if row.get("source_video", "") in excluded_videos:
            continue
        if row.get("clip_path", "") in excluded_paths:
            continue
        out.append(row)
    return out


# ── Annotation batch resolution ──────────────────────────────────────────

def list_eligible_batches(annotations_root: Path, ann_cfg) -> list[str]:
    """List batch directories that pass the recipe's annotations filter.

    Returned list is sorted ascending by batch_id (= sorted by date+NN).
    """
    if not annotations_root.is_dir():
        return []
    available = sorted(
        p.name for p in annotations_root.iterdir()
        if p.is_dir() and (p / "_meta.json").exists()
    )
    if ann_cfg.batches == ["*"]:
        kept = list(available)
    else:
        unknown = [b for b in ann_cfg.batches if b not in available]
        if unknown:
            raise SystemExit(
                f"recipe.annotations.batches references unknown batch_id(s): "
                f"{unknown!r} (available: {available!r})"
            )
        kept = [b for b in available if b in ann_cfg.batches]
    excluded = set(ann_cfg.exclude_batches)
    return [b for b in kept if b not in excluded]


def resolve_labels(
    annotations_root: Path,
    batches:          list[str],
    resolution:       str,
) -> dict[str, Path]:
    """Build {frame_stem: label_path}, applying multi-batch resolution policy.

    ``batches`` arrives sorted ascending; ``latest`` picks the highest-sorting
    batch_id, ``first`` picks the lowest, ``error`` raises on duplicates.
    """
    frame_to_hits: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for batch_id in batches:
        for txt in (annotations_root / batch_id).glob("*.txt"):
            if txt.name.startswith("_"):
                continue
            frame_to_hits[txt.stem].append((batch_id, txt))

    batch_order = {b: i for i, b in enumerate(batches)}
    out: dict[str, Path] = {}
    for stem, hits in frame_to_hits.items():
        if len(hits) == 1:
            out[stem] = hits[0][1]
            continue
        if resolution == "error":
            ids = [h[0] for h in hits]
            raise SystemExit(
                f"frame {stem!r} found in {len(hits)} batches: {ids!r}; "
                f"set annotations.resolution=latest|first to disambiguate"
            )
        hits.sort(key=lambda h: batch_order[h[0]])
        out[stem] = hits[-1][1] if resolution == "latest" else hits[0][1]
    return out


# ── Frame enumeration (flat layout) ──────────────────────────────────────

def enumerate_flat_frames(
    frames_root:           Path,
    sources,
    eligible_clip_stems:   set[str],
) -> list[dict]:
    """Walk ``frames/{ch}/{wv}/*.png`` (non-recursive) and decode names.

    Frames whose decoded ``clip_stem`` is not in ``eligible_clip_stems``
    are skipped. Subdirectories like ``_rejected/`` are ignored because
    we only ``glob('*.png')`` directly inside the wave dir.
    """
    out = []
    if not frames_root.is_dir():
        return out
    for ch_dir in sorted(frames_root.iterdir()):
        if not ch_dir.is_dir():
            continue
        for wv_dir in sorted(ch_dir.iterdir()):
            if not wv_dir.is_dir():
                continue
            for img in sorted(wv_dir.glob("*.png")):
                stem = img.stem
                try:
                    ch_id, wv_id, clip_stem = parse_frame_name(stem, sources)
                except ValueError:
                    continue
                if clip_stem not in eligible_clip_stems:
                    continue
                out.append({
                    "chamber_id": ch_id,
                    "wave_id":    wv_id,
                    "clip_stem":  clip_stem,
                    "frame_stem": stem,
                    "image_path": img,
                })
    return out


# ── Splitting ────────────────────────────────────────────────────────────

def stratify_key(frame: dict, mode: str) -> str:
    if mode == "chamber":
        return frame["chamber_id"]
    if mode == "wave":
        return f"{frame['chamber_id']}/{frame['wave_id']}"
    if mode == "clip":
        return f"{frame['chamber_id']}/{frame['wave_id']}/{frame['clip_stem']}"
    return "_all_"


def split_frames(frames: list[dict], recipe: RecipeConfig) -> dict[str, list[dict]]:
    """Group by stratify key, shuffle each group with the recipe seed,
    then slice each group according to ``ratios``. Splits with ratio 0
    (or absent) are omitted from the output."""
    rnd = random.Random(recipe.split.seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for f in frames:
        groups[stratify_key(f, recipe.split.stratify)].append(f)

    split_names = [s for s in SPLITS if recipe.split.ratios.get(s, 0) > 0]
    ratios = [recipe.split.ratios.get(s, 0) for s in split_names]
    total  = sum(ratios)
    norm   = [r / total for r in ratios]

    out: dict[str, list[dict]] = {s: [] for s in split_names}
    for key in sorted(groups):
        group = groups[key][:]
        rnd.shuffle(group)
        n = len(group)
        cuts = [int(round(sum(norm[:i + 1]) * n)) for i in range(len(split_names))]
        prev = 0
        for s, cut in zip(split_names, cuts):
            out[s].extend(group[prev:cut])
            prev = cut
    return out


# ── Dataset materialisation ─────────────────────────────────────────────

def _link_or_copy(src: Path, dst: Path) -> str:
    """Symlink src→dst; fall back to hardlink, then to copy. Returns mode used."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
        return "symlink"
    except (OSError, NotImplementedError):
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"


def unique_link_name(frame: dict, ext: str) -> str:
    """Globally-unique name across all clips: chamber_wave_clip_frame.ext."""
    return (f"{frame['chamber_id']}__{frame['wave_id']}__"
            f"{frame['clip_stem']}__{frame['frame_stem']}{ext}")


def materialise(
    dataset_dir: Path,
    splits:      dict[str, list[dict]],
    recipe:      RecipeConfig,
    recipe_path: Path,
) -> Path:
    """Write images/, labels/, data.yaml, manifest.csv, recipe.yaml."""
    dataset_dir.mkdir(parents=True)

    manifest_rows = []
    for split, frames in splits.items():
        for f in frames:
            img_name = unique_link_name(f, f["image_path"].suffix.lower())
            img_dst  = dataset_dir / "images" / split / img_name
            mode_img = _link_or_copy(f["image_path"], img_dst)

            label_path = f.get("label_path")
            if label_path is not None:
                lbl_name = unique_link_name(f, ".txt")
                lbl_dst  = dataset_dir / "labels" / split / lbl_name
                mode_lbl = _link_or_copy(label_path, lbl_dst)
                label_link = str(lbl_dst.relative_to(dataset_dir))
                label_src  = str(label_path)
            else:
                mode_lbl   = "none"
                label_link = ""
                label_src  = ""

            manifest_rows.append({
                "split":      split,
                "chamber_id": f["chamber_id"],
                "wave_id":    f["wave_id"],
                "clip_stem":  f["clip_stem"],
                "frame_stem": f["frame_stem"],
                "batch_id":   f.get("batch_id", ""),
                "image_link": str(img_dst.relative_to(dataset_dir)),
                "label_link": label_link,
                "image_src":  str(f["image_path"]),
                "label_src":  label_src,
                "link_mode":  f"img={mode_img}/lbl={mode_lbl}",
            })

    # data.yaml
    data_yaml = {
        "path":  str(dataset_dir.resolve()),
        "nc":    len(recipe.classes),
        "names": list(recipe.classes),
    }
    for split in splits:
        data_yaml[split] = f"images/{split}"
    (dataset_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    # manifest.csv
    if manifest_rows:
        with open(dataset_dir / "manifest.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    # frozen recipe copy
    shutil.copy2(recipe_path, dataset_dir / "recipe.yaml")
    return dataset_dir


# ── Driver ──────────────────────────────────────────────────────────────

def build(
    workspace_yaml: Path,
    recipe_path:    Path,
    force:          bool,
) -> dict:
    """Returns a summary dict suitable for a compact CLI report."""
    workspace = load_workspace(workspace_yaml)
    recipe    = load_recipe(recipe_path)
    if recipe.chamber_type != workspace.chamber_type:
        raise SystemExit(
            f"recipe.chamber_type={recipe.chamber_type!r} disagrees with "
            f"workspace.chamber_type={workspace.chamber_type!r}"
        )

    workspace_chamber_dir = Path(workspace.workspace.root)
    manifests_root        = Path(workspace.workspace.manifests)
    annotations_root      = Path(workspace.workspace.annotations)
    frames_root           = Path(workspace.workspace.frames or
                                 (workspace_chamber_dir / "frames"))
    datasets_root         = Path(workspace.workspace.dataset)

    all_clips_csv = manifests_root / "all_clips.csv"
    if not all_clips_csv.exists():
        raise SystemExit(f"manifest not found: {all_clips_csv}")

    with open(all_clips_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    eligible_clips = filter_clips(rows, recipe)
    eligible_clip_stems = {Path(r["clip_path"]).stem for r in eligible_clips}

    sources_yaml = workspace_chamber_dir / "sources.yaml"
    if not sources_yaml.exists():
        raise SystemExit(f"sources.yaml not found: {sources_yaml}")
    sources = load_sources(sources_yaml, probe=False)

    all_frames = enumerate_flat_frames(frames_root, sources, eligible_clip_stems)

    batches    = list_eligible_batches(annotations_root, recipe.annotations)
    labels_map = resolve_labels(annotations_root, batches, recipe.annotations.resolution)

    frames: list[dict] = []
    skipped_no_label = 0
    for f in all_frames:
        label = labels_map.get(f["frame_stem"])
        if label is None:
            if recipe.require_annotations:
                skipped_no_label += 1
                continue
            f["label_path"] = None
            f["batch_id"]   = ""
        else:
            f["label_path"] = label
            f["batch_id"]   = label.parent.name
        frames.append(f)

    if not frames:
        raise SystemExit(
            f"no frames matched recipe '{recipe.name}'. "
            f"({len(eligible_clips)} clip(s) eligible, "
            f"{len(all_frames)} frame(s) discovered, "
            f"{skipped_no_label} had no annotation in selected batches.)"
        )

    splits = split_frames(frames, recipe)

    dataset_dir = datasets_root / recipe.name
    if dataset_dir.exists():
        if not force:
            raise SystemExit(
                f"{dataset_dir} already exists. Re-run with --force to rebuild "
                f"(this will wipe the directory)."
            )
        shutil.rmtree(dataset_dir)

    materialise(dataset_dir, splits, recipe, recipe_path)

    return {
        "dataset_dir":      dataset_dir,
        "n_clips":          len(eligible_clips),
        "n_frames_total":   len(all_frames),
        "n_frames":         len(frames),
        "skipped_no_label": skipped_no_label,
        "n_batches":        len(batches),
        "batches":          batches,
        "splits":           {s: len(splits[s]) for s in splits},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace-yaml", required=True, type=Path)
    p.add_argument("--recipe",         required=True, type=Path)
    p.add_argument("--force", action="store_true",
                   help="Wipe an existing datasets/{name}/ before rebuilding.")
    args = p.parse_args()

    summary = build(
        workspace_yaml=args.workspace_yaml,
        recipe_path=args.recipe,
        force=args.force,
    )

    print(f"\n[OK] Built dataset at {summary['dataset_dir']}")
    print(f"   eligible clips      : {summary['n_clips']}")
    print(f"   frames discovered   : {summary['n_frames_total']}")
    print(f"   skipped no-label    : {summary['skipped_no_label']}")
    print(f"   batches used        : {summary['n_batches']} {summary['batches']!r}")
    print(f"   frames in dataset   : {summary['n_frames']}")
    for s, n in summary["splits"].items():
        print(f"     {s:5s} : {n}")


if __name__ == "__main__":
    main()
