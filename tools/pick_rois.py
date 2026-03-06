#!/usr/bin/env python3
"""
tools/pick_rois.py
──────────────────
Interactive tool to click the 4 chamber corners for every video in a folder
that doesn't already have an ROI entry.  Results are appended directly to
the camera_rois.json on your drive.

Usage
-----
    python tools/pick_rois.py \\
        --video-dir /media/woodlab/104-A/Wave3/00_raw_videos \\
        --roi-file  /media/woodlab/104-A/Wave3/02_Global_Metadata/camera_rois.json

Controls (OpenCV window)
------------------------
    Left-click      : place a corner (order: TL → TR → BR → BL)
    R               : reset corners for the current video
    S / Enter       : save and continue to the next video
    Q / Escape      : quit without saving current video
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov"}
CORNER_LABELS    = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
CORNER_COLORS    = [
    (0, 255, 0),    # TL – green
    (0, 165, 255),  # TR – orange
    (0, 0, 255),    # BR – red
    (255, 0, 255),  # BL – magenta
]


def find_videos(root: Path) -> list[Path]:
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(root.rglob(f"*{ext}"))
    return sorted(videos)


def load_rois(roi_file: Path) -> dict:
    if roi_file.exists():
        with open(roi_file) as f:
            return json.load(f)
    return {}


def save_rois(roi_file: Path, rois: dict) -> None:
    roi_file.parent.mkdir(parents=True, exist_ok=True)
    with open(roi_file, "w") as f:
        json.dump(rois, f, indent=4)
    print(f"💾 Saved ROIs → {roi_file}")


def get_first_frame(video_path: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def pick_corners(frame: np.ndarray, video_name: str) -> list[list[int]] | None:
    """
    Display an interactive OpenCV window and let the user click 4 corners.
    Returns a list of [x, y] pairs, or None if the user skipped / quit.
    """
    corners: list[tuple[int, int]] = []
    display  = frame.copy()
    win_name = f"Pick ROI – {video_name}  (R=reset  S/Enter=save  Q/Esc=skip)"

    def _draw():
        nonlocal display
        display = frame.copy()
        for i, (cx, cy) in enumerate(corners):
            cv2.circle(display, (cx, cy), 6, CORNER_COLORS[i], -1)
            cv2.putText(display, CORNER_LABELS[i], (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, CORNER_COLORS[i], 1)
        if len(corners) == 4:
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(display, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
        remaining = 4 - len(corners)
        if remaining > 0:
            label = CORNER_LABELS[len(corners)]
            msg   = f"Click {label}  ({remaining} remaining)"
        else:
            msg = "4/4 corners set. Press S to save, R to reset."
        cv2.putText(display, msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(win_name, display)

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append((x, y))
            _draw()

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)
    cv2.setMouseCallback(win_name, _on_mouse)
    _draw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("r"), ord("R")):
            corners.clear()
            _draw()
        elif key in (ord("s"), ord("S"), 13):   # 13 = Enter
            if len(corners) == 4:
                cv2.destroyWindow(win_name)
                return [[int(x), int(y)] for x, y in corners]
            else:
                print(f"⚠️  Need exactly 4 corners, have {len(corners)}.")
        elif key in (ord("q"), ord("Q"), 27):   # 27 = Esc
            cv2.destroyWindow(win_name)
            return None
        # Also handle window close button
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            return None


def main():
    parser = argparse.ArgumentParser(description="Interactive ROI picker for chamber videos.")
    parser.add_argument("--video-dir", required=True, help="Directory to scan for videos.")
    parser.add_argument("--roi-file",  required=True, help="Path to camera_rois.json (will be created/updated).")
    parser.add_argument("--force",     action="store_true",
                        help="Re-pick ROIs for videos that already have an entry.")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    roi_file  = Path(args.roi_file)

    if not video_dir.exists():
        print(f"❌ Video directory not found: {video_dir}")
        sys.exit(1)

    rois   = load_rois(roi_file)
    videos = find_videos(video_dir)

    if not videos:
        print(f"❌ No videos found under {video_dir}")
        sys.exit(1)

    pending = [v for v in videos if (args.force or v.name not in rois)]

    print(f"\n📂 Found {len(videos)} videos total.")
    print(f"✏️  {len(pending)} need ROI picking "
          f"({'including already-set' if args.force else 'skipping existing'}).\n")

    for i, video_path in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] {video_path.name}")
        frame = get_first_frame(video_path)
        if frame is None:
            print(f"  ⚠️  Could not read first frame – skipping.")
            continue

        result = pick_corners(frame, video_path.name)
        if result is None:
            print(f"  ⏭  Skipped.")
            continue

        rois[video_path.name] = result
        save_rois(roi_file, rois)   # save after each video (crash-safe)
        print(f"  ✅ Saved corners: {result}")

    print("\nDone.")


if __name__ == "__main__":
    main()
