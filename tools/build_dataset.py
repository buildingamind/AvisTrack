#!/usr/bin/env python3
"""
tools/build_dataset.py
──────────────────────
Materialise one ``datasets/{recipe.name}/`` view from a chamber-type
workspace's ``clips/`` + ``frames/`` + ``annotations/`` inventory.

Output layout (Ultralytics-compatible)::

    {workspace}/{chamber_type}/datasets/{name}/
        recipe.yaml      ← copy of the input recipe (frozen at build time)
        manifest.csv     ← per-frame: split, chamber, wave, clip_stem, image, label
        data.yaml        ← Ultralytics dataset config
        images/{train,val,test}/<symlink>.png
        labels/{train,val,test}/<symlink>.txt

The build is deterministic given the recipe + workspace state: same
``recipe.split.seed`` and same source frames → identical splits. Datasets
are immutable: re-running with the same name refuses to overwrite unless
``--force`` is passed (in which case the directory is wiped first).

Usage
-----
    python tools/build_dataset.py \\
        --workspace-yaml /media/ssd/avistrack/collective/workspace.yaml \\
        --recipe         datasets_recipes/full_v1.yaml
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


# ── Frame enumeration ────────────────────────────────────────────────────

def find_image_for_label(frames_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        cand = frames_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def collect_frames(
    annotations_root: Path,
    frames_root:      Path,
    clip_row:         dict,
) -> list[dict]:
    """
    For one clip row, return a list of {chamber_id, wave_id, clip_stem,
    image_path, label_path, frame_stem} for every (image, label) pair
    that exists on disk.
    """
    chamber_id = clip_row["chamber_id"]
    wave_id    = clip_row["wave_id"]
    clip_stem  = Path(clip_row["clip_path"]).stem

    ann_dir   = annotations_root / chamber_id / wave_id / clip_stem
    frame_dir = frames_root      / chamber_id / wave_id / clip_stem

    if not ann_dir.is_dir():
        return []

    out = []
    for label in sorted(ann_dir.glob("*.txt")):
        if label.name.startswith("_"):  # skip _meta and friends
            continue
        image = find_image_for_label(frame_dir, label.stem)
        if image is None:
            continue
        out.append({
            "chamber_id": chamber_id,
            "wave_id":    wave_id,
            "clip_stem":  clip_stem,
            "frame_stem": label.stem,
            "image_path": image,
            "label_path": label,
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
    """
    Group by stratify key, shuffle each group with the recipe seed,
    then slice each group according to ``ratios``. Splits with ratio 0
    (or absent) are omitted from the output.
    """
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
        # cumulative cut points so all frames are assigned exactly once
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
    splits: dict[str, list[dict]],
    recipe: RecipeConfig,
    recipe_path: Path,
) -> Path:
    """Write images/, labels/, data.yaml, manifest.csv, recipe.yaml."""
    dataset_dir.mkdir(parents=True)

    manifest_rows = []
    for split, frames in splits.items():
        for f in frames:
            img_name = unique_link_name(f, f["image_path"].suffix.lower())
            lbl_name = unique_link_name(f, ".txt")
            img_dst  = dataset_dir / "images" / split / img_name
            lbl_dst  = dataset_dir / "labels" / split / lbl_name
            mode_img = _link_or_copy(f["image_path"], img_dst)
            mode_lbl = _link_or_copy(f["label_path"], lbl_dst)
            manifest_rows.append({
                "split":      split,
                "chamber_id": f["chamber_id"],
                "wave_id":    f["wave_id"],
                "clip_stem":  f["clip_stem"],
                "frame_stem": f["frame_stem"],
                "image_link": str(img_dst.relative_to(dataset_dir)),
                "label_link": str(lbl_dst.relative_to(dataset_dir)),
                "image_src":  str(f["image_path"]),
                "label_src":  str(f["label_path"]),
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

    frames: list[dict] = []
    skipped_unannotated = 0
    for row in eligible_clips:
        per_clip = collect_frames(annotations_root, frames_root, row)
        if not per_clip:
            if recipe.require_annotations:
                skipped_unannotated += 1
                continue
        frames.extend(per_clip)

    if not frames:
        raise SystemExit(
            f"no labelled frames matched recipe '{recipe.name}'. "
            f"({len(eligible_clips)} clip(s) survived filtering, "
            f"{skipped_unannotated} were unannotated.)"
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
        "dataset_dir":  dataset_dir,
        "n_clips":      len(eligible_clips),
        "n_frames":     len(frames),
        "skipped_unannotated": skipped_unannotated,
        "splits":       {s: len(splits[s]) for s in splits},
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

    print(f"\n✅ Built dataset at {summary['dataset_dir']}")
    print(f"   eligible clips      : {summary['n_clips']}")
    print(f"   skipped unannotated : {summary['skipped_unannotated']}")
    print(f"   frames              : {summary['n_frames']}")
    for s, n in summary["splits"].items():
        print(f"     {s:5s} : {n}")


if __name__ == "__main__":
    main()
