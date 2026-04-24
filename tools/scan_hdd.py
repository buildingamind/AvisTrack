#!/usr/bin/env python3
"""
tools/scan_hdd.py
─────────────────
Walk one or more external HDDs and catalog every raw video found under each
``Wave{N}_{TAG}/00_raw_videos/`` subtree.  For each wave, an updated
``02_Global_Metadata/video_catalog.csv`` is written on the same drive.

Output CSV fields (CSV style mirrors sample_clips.py manifests):

    Video_Filename, Original_Video_Path, Duration_Sec, Frame_Count, Fps,
    Modality, Width, Height, Last_Probed

Idempotent: re-running merges with any existing catalog by ``Original_Video_Path``,
overwriting probe fields and preserving rows whose video has since disappeared
(marked ``Missing: yes``).

Usage
-----
    python tools/scan_hdd.py --hdd /mnt/hdd_A
    python tools/scan_hdd.py --hdds /mnt/hdd_A /mnt/hdd_B
"""

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2

VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".mov"}
WAVE_DIR_RE = re.compile(r"^Wave\d+(_[A-Za-z0-9]+)?$")
RAW_SUBDIR = "00_raw_videos"
META_SUBDIR = "02_Global_Metadata"
CATALOG_NAME = "video_catalog.csv"

CATALOG_FIELDS = [
    "Video_Filename",
    "Original_Video_Path",
    "Duration_Sec",
    "Frame_Count",
    "Fps",
    "Modality",
    "Width",
    "Height",
    "Last_Probed",
    "Missing",
]


def detect_modality(name: str) -> str:
    lower = name.lower()
    if "ir" in lower and "rgb" not in lower:
        return "ir"
    if "rgb" in lower:
        return "rgb"
    return "rgb"  # default


def probe_video(path: Path) -> dict | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if n_frames < 1:
        return None
    return {
        "fps": fps,
        "n_frames": n_frames,
        "width": width,
        "height": height,
        "duration_sec": n_frames / fps if fps > 0 else 0.0,
    }


def find_wave_dirs(hdd_root: Path) -> list[Path]:
    if not hdd_root.exists():
        return []
    return sorted(
        p for p in hdd_root.iterdir()
        if p.is_dir() and WAVE_DIR_RE.match(p.name)
    )


def find_videos(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    return sorted(
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def load_catalog(catalog_path: Path) -> dict[str, dict]:
    if not catalog_path.exists():
        return {}
    rows = {}
    with open(catalog_path, newline="") as f:
        for row in csv.DictReader(f):
            key = row.get("Original_Video_Path", "").strip()
            if key:
                rows[key] = {k.strip(): (v or "").strip() for k, v in row.items()}
    return rows


def write_catalog(catalog_path: Path, rows: dict[str, dict]) -> None:
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with open(catalog_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CATALOG_FIELDS)
        writer.writeheader()
        for key in sorted(rows):
            row = rows[key]
            writer.writerow({k: row.get(k, "") for k in CATALOG_FIELDS})


def scan_wave(wave_dir: Path) -> tuple[int, int, int]:
    """Returns (n_videos_found, n_new, n_missing)."""
    raw_dir = wave_dir / RAW_SUBDIR
    catalog_path = wave_dir / META_SUBDIR / CATALOG_NAME

    rows = load_catalog(catalog_path)
    videos = find_videos(raw_dir)
    seen_paths = set()
    n_new = 0

    for v in videos:
        full = str(v.resolve())
        seen_paths.add(full)
        existing = rows.get(full)

        info = probe_video(v)
        if info is None:
            # Unreadable — mark as missing, keep any prior data
            row = existing or {"Video_Filename": v.name, "Original_Video_Path": full}
            row["Missing"] = "yes"
            rows[full] = row
            continue

        is_new = existing is None
        rows[full] = {
            "Video_Filename": v.name,
            "Original_Video_Path": full,
            "Duration_Sec": f"{info['duration_sec']:.2f}",
            "Frame_Count": str(info["n_frames"]),
            "Fps": f"{info['fps']:.3f}",
            "Modality": detect_modality(v.name),
            "Width": str(info["width"]),
            "Height": str(info["height"]),
            "Last_Probed": datetime.now().isoformat(timespec="seconds"),
            "Missing": "no",
        }
        if is_new:
            n_new += 1

    n_missing = 0
    for path, row in rows.items():
        if path not in seen_paths and row.get("Missing", "no") != "yes":
            row["Missing"] = "yes"
            n_missing += 1

    write_catalog(catalog_path, rows)
    return len(videos), n_new, n_missing


def scan_hdd(hdd_root: Path) -> None:
    wave_dirs = find_wave_dirs(hdd_root)
    if not wave_dirs:
        print(f"⚠️  No Wave{{N}}_{{TAG}} dirs under {hdd_root}")
        return
    print(f"🔍 {hdd_root} — {len(wave_dirs)} wave dir(s)")
    for wd in wave_dirs:
        n_found, n_new, n_missing = scan_wave(wd)
        catalog = wd / META_SUBDIR / CATALOG_NAME
        print(
            f"   {wd.name}: {n_found} videos "
            f"(+{n_new} new, {n_missing} newly missing) → {catalog}"
        )


def main():
    ap = argparse.ArgumentParser(description="Catalog raw videos on one or more HDDs.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--hdd", type=Path, help="Single HDD mount point")
    g.add_argument("--hdds", type=Path, nargs="+", help="Multiple HDD mount points")
    args = ap.parse_args()

    hdds = [args.hdd] if args.hdd else args.hdds
    for hdd in hdds:
        if not hdd.exists():
            print(f"❌ Mount point does not exist: {hdd}")
            sys.exit(1)
        scan_hdd(hdd)


if __name__ == "__main__":
    main()
