#!/usr/bin/env python3
"""
tools/aggregate_for_training.py
───────────────────────────────
Incrementally copy annotated batches from one HDD at a time into a single local
training cache on dev-PC SSD, then finalize a Ultralytics-compatible dataset.yaml.

Why incremental
---------------
The user can only mount one external HDD at a time.  Each invocation with
``--hdd <mount>`` scans that HDD for batches that match the config's expected
sources, diffs against ``state.json``, and copies only what's new.  Repeat for
each HDD; then run with ``--finalize`` to emit ``dataset.yaml``.

Files are copied (not symlinked) so the cache survives unplugging the HDDs.
Each file gets a ``<wave>_<batch_id>_`` prefix to avoid cross-HDD name
collisions in the merged cache.

Config (``configs/VR/training_sets/combined_prod.yaml``)
--------------------------------------------------------
    name: VR_combined_prod
    local_cache_root: /home/user/avistrack_training_cache
    class_names: [chick]
    sources:
      - wave: Wave3_prod
        splits: [train, val]
        batches: all              # or [VR_Wave3_..._batch01, ...]
      - wave: Wave4_prod
        splits: [train, val]
        batches: all

Usage
-----
    python tools/aggregate_for_training.py \\
        --config configs/VR/training_sets/combined_prod.yaml \\
        --hdd /mnt/hdd_A

    # ... unplug, plug next HDD, run again ...

    python tools/aggregate_for_training.py \\
        --config configs/VR/training_sets/combined_prod.yaml \\
        --finalize
"""

import argparse
import json
import shutil
import sys
import yaml
from datetime import datetime
from pathlib import Path

SPLIT_TO_DIR = {"train": "train", "val": "val_tuning", "test": "test_golden"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def load_aggregate_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for required in ("name", "local_cache_root", "sources"):
        if required not in cfg:
            print(f"❌ Aggregate config missing required field: {required}")
            sys.exit(1)
    return cfg


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"config_name": None, "completed_batches": []}
    with open(state_path) as f:
        return json.load(f)


def save_state(state_path: Path, state: dict) -> None:
    state["last_updated"] = datetime.now().isoformat(timespec="seconds")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def is_batch_done(state: dict, hdd_label: str, wave: str, split: str, batch_id: str) -> bool:
    return any(
        b.get("hdd_label") == hdd_label
        and b.get("wave") == wave
        and b.get("split") == split
        and b.get("batch_id") == batch_id
        for b in state.get("completed_batches", [])
    )


def find_batches_on_hdd(hdd_root: Path, source: dict) -> list[tuple[str, str, Path, Path]]:
    """
    Returns [(split, batch_id, images_dir, annotation_dir), ...] for batches whose
    images/<batch_id>/ has matching annotation/<batch_id>/.
    """
    out = []
    wave_dir = hdd_root / source["wave"]
    if not wave_dir.exists():
        return out
    dataset_dir = wave_dir / "01_Dataset_MOT_Format"
    if not dataset_dir.exists():
        return out

    requested_batches = source.get("batches", "all")
    for split in source.get("splits", ["train", "val"]):
        split_subdir = SPLIT_TO_DIR.get(split, split)
        images_root = dataset_dir / split_subdir / "images"
        anno_root = dataset_dir / split_subdir / "annotation"
        if not images_root.exists() or not anno_root.exists():
            continue
        for img_dir in sorted(images_root.iterdir()):
            if not img_dir.is_dir() or img_dir.name.startswith("_"):
                continue
            batch_id = img_dir.name
            if requested_batches != "all" and batch_id not in requested_batches:
                continue
            anno_dir = anno_root / batch_id
            if not anno_dir.exists():
                continue
            out.append((split, batch_id, img_dir, anno_dir))
    return out


def copy_batch(images_dir: Path, anno_dir: Path, cache_root: Path, split: str,
               wave: str, batch_id: str) -> tuple[int, int]:
    """Copy approved images + matching labels into cache. Returns (n_images, n_labels)."""
    cache_split = "val" if split == "val" else split  # keep config split name
    images_out = cache_root / cache_split / "images"
    labels_out = cache_root / cache_split / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    n_img = n_lbl = 0
    prefix = f"{wave}_{batch_id}_"
    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = anno_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue  # skip unlabelled
        shutil.copy2(img_path, images_out / f"{prefix}{img_path.name}")
        shutil.copy2(label_path, labels_out / f"{prefix}{img_path.stem}.txt")
        n_img += 1
        n_lbl += 1
    return n_img, n_lbl


def run_one_hdd(cfg: dict, hdd_root: Path, cache_root: Path, state_path: Path):
    state = load_state(state_path)
    if state.get("config_name") and state["config_name"] != cfg["name"]:
        print(f"❌ State config_name mismatch (state={state['config_name']!r}, "
              f"cfg={cfg['name']!r}). Use a fresh cache or correct config.")
        sys.exit(1)
    state["config_name"] = cfg["name"]
    state.setdefault("completed_batches", [])

    hdd_label = hdd_root.name
    print(f"🔌 HDD: {hdd_root} (label = {hdd_label!r})")

    new_total_img = new_total_lbl = 0
    new_batches = 0
    for src in cfg["sources"]:
        candidates = find_batches_on_hdd(hdd_root, src)
        if not candidates:
            continue
        for split, batch_id, img_dir, anno_dir in candidates:
            if is_batch_done(state, hdd_label, src["wave"], split, batch_id):
                print(f"   skip (already copied): {src['wave']} / {split} / {batch_id}")
                continue
            n_img, n_lbl = copy_batch(img_dir, anno_dir, cache_root, split, src["wave"], batch_id)
            print(f"   copied: {src['wave']} / {split} / {batch_id} → {n_img} img / {n_lbl} lbl")
            state["completed_batches"].append({
                "hdd_label": hdd_label,
                "hdd_mount": str(hdd_root),
                "wave": src["wave"],
                "split": split,
                "batch_id": batch_id,
                "n_files": n_img,
                "copied_at": datetime.now().isoformat(timespec="seconds"),
            })
            new_total_img += n_img
            new_total_lbl += n_lbl
            new_batches += 1

    save_state(state_path, state)
    print(f"\n📦 Run summary: +{new_batches} new batch(es), "
          f"{new_total_img} images / {new_total_lbl} labels copied.")
    print(f"   State: {state_path}")
    _print_progress(cfg, state)


def _print_progress(cfg: dict, state: dict) -> None:
    print("\n📊 Progress vs. expected sources:")
    for src in cfg["sources"]:
        in_state = [b for b in state.get("completed_batches", [])
                    if b.get("wave") == src["wave"]]
        for split in src.get("splits", ["train", "val"]):
            n = sum(1 for b in in_state if b.get("split") == split)
            print(f"   {src['wave']} / {split}: {n} batch(es) collected")


def finalize(cfg: dict, cache_root: Path, state_path: Path) -> None:
    state = load_state(state_path)
    if not state.get("completed_batches"):
        print("❌ No batches in cache yet. Run with --hdd first.")
        sys.exit(1)

    # Sanity: every (wave, split) in cfg has at least one batch
    missing = []
    for src in cfg["sources"]:
        in_state = [b for b in state["completed_batches"] if b.get("wave") == src["wave"]]
        for split in src.get("splits", ["train", "val"]):
            if not any(b.get("split") == split for b in in_state):
                missing.append(f"{src['wave']}/{split}")
    if missing:
        print(f"⚠️  Some expected sources have NO batches in cache: {missing}")
        print("   Continuing with what's available; rerun --hdd for missing HDDs to expand.")

    class_names = cfg.get("class_names", ["chick"])
    dataset_yaml = {
        "path": str(cache_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": class_names,
    }
    out_path = cache_root / "dataset.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)

    n_train = len(list((cache_root / "train" / "images").glob("*"))) \
        if (cache_root / "train" / "images").exists() else 0
    n_val = len(list((cache_root / "val" / "images").glob("*"))) \
        if (cache_root / "val" / "images").exists() else 0

    # source.meta.json: provenance for every cached file
    source_meta = {
        "config_name": cfg["name"],
        "finalized_at": datetime.now().isoformat(timespec="seconds"),
        "n_train_images": n_train,
        "n_val_images": n_val,
        "completed_batches": state["completed_batches"],
    }
    with open(cache_root / "source.meta.json", "w") as f:
        json.dump(source_meta, f, indent=2)

    print(f"\n✅ Finalized: {out_path}")
    print(f"   train images: {n_train}")
    print(f"   val images:   {n_val}")
    print(f"   provenance:   {cache_root / 'source.meta.json'}")


def main():
    ap = argparse.ArgumentParser(description="Incremental multi-HDD aggregation into local YOLO cache.")
    ap.add_argument("--config", required=True, type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--hdd", type=Path, help="Mount point of the currently-plugged HDD")
    g.add_argument("--finalize", action="store_true",
                   help="Generate dataset.yaml from already-cached batches")
    args = ap.parse_args()

    cfg = load_aggregate_config(args.config)
    cache_root = Path(cfg["local_cache_root"]) / cfg["name"]
    cache_root.mkdir(parents=True, exist_ok=True)
    state_path = cache_root / "state.json"

    if args.finalize:
        finalize(cfg, cache_root, state_path)
    else:
        if not args.hdd.exists():
            print(f"❌ HDD mount does not exist: {args.hdd}")
            sys.exit(1)
        run_one_hdd(cfg, args.hdd.resolve(), cache_root, state_path)


if __name__ == "__main__":
    main()
