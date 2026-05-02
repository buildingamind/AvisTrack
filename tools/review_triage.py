#!/usr/bin/env python3
"""
tools/review_triage.py
──────────────────────
cv2 single-frame triage UI for approving / rejecting extracted frames before
sending them to CVAT.

Two modes
---------
**Workspace mode** (recommended, multi-chamber architecture):
    python tools/review_triage.py \\
        --workspace-yaml /path/to/workspace.yaml \\
        --chamber-id vr_105A --wave-id wave3 \\
        --batch vr_105A_wave3_2026-05-01_batch01

  Reads frames from   ``{workspace}/frames/{chamber}/{wave}/*.png`` (flat)
  Reads manifest from ``{workspace}/manifests/triage/{batch_id}.csv``
  Rejects (on finalize) → ``{workspace}/frames/{chamber}/{wave}/_rejected/``

**Legacy --config mode**:
    python tools/review_triage.py \\
        --config configs/VR/wave3_vr.yaml \\
        --split  train --batch <batch_id>

  Reads ``<dataset>/<split_dir>/images/<batch_id>/*.png``
  Rejects (on finalize) → ``<images>/<batch_id>/_rejected/``

Keys
----
  →  / d           approve current frame, advance
  ←  / a           reject  current frame, advance
  b                go back one frame (no status change)
  n                go forward one frame (no status change)
  space            jump to next pending
  z                undo current (back to pending)
  q  / Esc         quit (manifest auto-saved on every keystroke;
                          rejected PNGs auto-moved to _rejected/ on exit)
"""

import argparse
import csv
import shutil
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from avistrack.config.loader import load_config


SPLIT_TO_DIR = {"train": "train", "val": "val_tuning", "test": "test_golden"}
CSV_FIELDS = [
    "Frame_Filename",
    "Source_Clip",
    "Original_Video_Path",
    "Frame_Idx",
    "Timestamp",
    "Triage_Status",
]
WIN = "AvisTrack Triage"
TARGET_PX = 720            # image scaled so its larger dim hits this
MIN_CANVAS_W = 900         # widen canvas (with letterboxing) so help text fits

# Status → BGR colour
COLOUR = {
    "approved": (128, 222,  74),   # green
    "rejected": ( 96,  96, 248),   # red
    "pending":  (180, 180, 180),   # grey
}

# Windows VK_* arrow codes via cv2.waitKeyEx (other platforms differ but the
# letter fallbacks d/a/b/n still work everywhere).
KEY_LEFT_CODES  = {2424832, 65361, 81}
KEY_RIGHT_CODES = {2555904, 65363, 83}
KEY_UP_CODES    = {2490368, 65362, 82}
KEY_DOWN_CODES  = {2621440, 65364, 84}


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        print(f"❌ Manifest not found: {path}")
        sys.exit(1)
    with open(path, newline="") as f:
        return [
            {k.strip(): (v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def write_manifest(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in CSV_FIELDS} for r in rows)


def render_canvas(img: np.ndarray, idx: int, total: int, status: str,
                  fname: str, n_app: int, n_rej: int, n_pen: int) -> np.ndarray:
    """Compose the image (scaled to TARGET_PX) plus a top status bar and a
    bottom filename bar. Aspect ratio is preserved; narrow images are
    centred via black letterboxing so help text always fits."""
    h, w = img.shape[:2]
    s = TARGET_PX / max(h, w)
    if s != 1.0:
        interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
        img = cv2.resize(img, (max(int(w * s), 1), max(int(h * s), 1)),
                         interpolation=interp)
    h, w = img.shape[:2]

    canvas_w = max(w, MIN_CANVAS_W)
    colour = COLOUR.get(status, COLOUR["pending"])

    # Letterbox the image to canvas_w (centred), then draw the colour border
    # directly on the image content rect — no further width changes after this.
    if w < canvas_w:
        left = (canvas_w - w) // 2
        right = canvas_w - w - left
        img = cv2.copyMakeBorder(img, 0, 0, left, right,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.rectangle(img, (left, 0), (left + w - 1, h - 1), colour, 4)
    else:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), colour, 4)

    top_h = 56
    top = np.full((top_h, canvas_w, 3), 24, dtype=np.uint8)
    cv2.putText(top, f"{idx}/{total}", (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(top, f"approved {n_app}   rejected {n_rej}   pending {n_pen}",
                (180, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(top, status.upper(), (180, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
    cv2.putText(top, "a/<- reject   d/-> approve   b back   n next   "
                     "space next-pending   z undo   f finalize   q quit",
                (16, top_h - 6), cv2.FONT_HERSHEY_PLAIN, 0.95,
                (140, 140, 140), 1, cv2.LINE_AA)

    bot_h = 28
    bot = np.full((bot_h, canvas_w, 3), 24, dtype=np.uint8)
    cv2.putText(bot, fname, (16, 20), cv2.FONT_HERSHEY_PLAIN, 1.1,
                (180, 180, 180), 1, cv2.LINE_AA)

    return np.vstack([top, img, bot])


def finalize(rows: list[dict], images_dir: Path, manifest_path: Path) -> int:
    rejected_dir = images_dir / "_rejected"
    rejected_dir.mkdir(exist_ok=True)
    n_moved = 0
    for r in rows:
        if r.get("Triage_Status") != "rejected":
            continue
        src = images_dir / r["Frame_Filename"]
        if not src.exists():
            continue
        shutil.move(str(src), str(rejected_dir / src.name))
        n_moved += 1
    write_manifest(manifest_path, rows)
    return n_moved


def run_triage(images_dir: Path, manifest_path: Path, batch_id: str) -> None:
    rows = load_manifest(manifest_path)
    if not rows:
        print("Empty manifest, nothing to triage.")
        return

    n = len(rows)

    # Resume from first pending if any, else start at 0
    cur = 0
    for i, r in enumerate(rows):
        if (r.get("Triage_Status") or "pending") == "pending":
            cur = i
            break

    print(f"📋 Triage: {n} frame(s) · batch {batch_id}")
    print(f"   Images:   {images_dir}")
    print(f"   Manifest: {manifest_path}")
    print(f"   Keys: a/← reject  d/→ approve  b back  n next  "
          f"space next-pending  z undo  q quit (auto-finalizes)")

    # AUTOSIZE = window hugs the canvas, no stretching / aspect distortion.
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    while True:
        row = rows[cur]
        img_path = images_dir / row["Frame_Filename"]
        img = cv2.imread(str(img_path)) if img_path.exists() else None
        if img is None:
            img = np.zeros((480, 720, 3), dtype=np.uint8)
            cv2.putText(img, f"MISSING: {row['Frame_Filename']}",
                        (16, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (80, 80, 248), 2, cv2.LINE_AA)

        n_app = sum(1 for r in rows if r.get("Triage_Status") == "approved")
        n_rej = sum(1 for r in rows if r.get("Triage_Status") == "rejected")
        n_pen = n - n_app - n_rej
        status = row.get("Triage_Status") or "pending"
        canvas = render_canvas(img, cur + 1, n, status,
                               row["Frame_Filename"], n_app, n_rej, n_pen)
        cv2.imshow(WIN, canvas)
        k = cv2.waitKeyEx(0)

        # Window closed via the OS X button → behave like quit
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

        # Approve + advance
        if k in KEY_RIGHT_CODES or k == ord('d'):
            row["Triage_Status"] = "approved"
            write_manifest(manifest_path, rows)
            if cur + 1 < n:
                cur += 1
        # Reject + advance
        elif k in KEY_LEFT_CODES or k == ord('a'):
            row["Triage_Status"] = "rejected"
            write_manifest(manifest_path, rows)
            if cur + 1 < n:
                cur += 1
        # Navigation only (no status change)
        elif k == ord('b') or k in KEY_UP_CODES:
            cur = max(cur - 1, 0)
        elif k == ord('n') or k in KEY_DOWN_CODES:
            cur = min(cur + 1, n - 1)
        # Jump to next pending
        elif k == ord(' '):
            for off in range(1, n + 1):
                j = (cur + off) % n
                if (rows[j].get("Triage_Status") or "pending") == "pending":
                    cur = j
                    break
        # Undo current
        elif k == ord('z'):
            row["Triage_Status"] = "pending"
            write_manifest(manifest_path, rows)
        # Quit
        elif k == 27 or k == ord('q'):
            break

    write_manifest(manifest_path, rows)
    cv2.destroyAllWindows()

    n_app = sum(1 for r in rows if r.get("Triage_Status") == "approved")
    n_rej = sum(1 for r in rows if r.get("Triage_Status") == "rejected")
    n_pen = n - n_app - n_rej

    # Auto-finalize on exit: any rejected row whose PNG is still in the main
    # dir gets moved into _rejected/. Idempotent — safe to run repeatedly.
    n_moved = finalize(rows, images_dir, manifest_path) if n_rej > 0 else 0

    print(f"💾 Saved: {n_app} approved · {n_rej} rejected · {n_pen} pending → {manifest_path}")
    if n_moved:
        print(f"🗑️  Auto-moved {n_moved} reject(s) → {images_dir / '_rejected'}/")
    if n_pen:
        print(f"⚠️  {n_pen} frame(s) still pending — re-run to finish triage.")


def _resolve_paths_workspace(args):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from avistrack.workspace import load_context

    workspace_yaml = Path(args.workspace_yaml)
    sources_yaml = Path(args.sources_yaml) if args.sources_yaml else workspace_yaml.with_name("sources.yaml")
    if not sources_yaml.exists():
        print(f"❌ sources.yaml not found: {sources_yaml}")
        sys.exit(1)

    ctx = load_context(
        workspace_yaml=workspace_yaml,
        sources_yaml=sources_yaml,
        chamber_id=args.chamber_id,
        wave_id=args.wave_id,
        require_drive=False,
    )
    images_dir    = ctx.frame_dir                                    # workspace/frames/{ch}/{wv}/
    manifest_path = ctx.manifests_root / "triage" / f"{args.batch}.csv"
    return images_dir, manifest_path


def _resolve_paths_config(args):
    cfg = load_config(args.config)
    dataset_root = cfg.drive.dataset
    metadata_root = cfg.drive.metadata
    images_dir    = Path(dataset_root) / SPLIT_TO_DIR[args.split] / "images" / args.batch
    manifest_path = Path(metadata_root) / "frame_manifests" / args.split / f"{args.batch}.csv"
    return images_dir, manifest_path


def main():
    ap = argparse.ArgumentParser(
        description="cv2 single-frame triage UI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--workspace-yaml", default=None,
                    help="Workspace mode: path to workspace.yaml")
    ap.add_argument("--sources-yaml", default=None,
                    help="Workspace mode: sources.yaml (default: sibling of workspace-yaml)")
    ap.add_argument("--chamber-id", default=None, help="Workspace mode: chamber id")
    ap.add_argument("--wave-id", default=None, help="Workspace mode: wave id")
    ap.add_argument("--config", default=None, help="Legacy mode: AvisTrack single-drive YAML")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"],
                    help="Legacy mode only")
    ap.add_argument("--batch", required=True, help="batch_id from extract_frames.py")
    args = ap.parse_args()

    workspace_mode = bool(args.workspace_yaml or args.chamber_id or args.wave_id)
    if workspace_mode:
        if args.config:
            print("❌ --config cannot be combined with workspace-mode flags")
            sys.exit(2)
        missing = [name for name, val in (
            ("--workspace-yaml", args.workspace_yaml),
            ("--chamber-id",     args.chamber_id),
            ("--wave-id",        args.wave_id),
        ) if not val]
        if missing:
            print(f"❌ workspace mode requires {', '.join(missing)}")
            sys.exit(2)
        images_dir, manifest_path = _resolve_paths_workspace(args)
    else:
        if not args.config:
            print("❌ Either --workspace-yaml (workspace mode) or --config (legacy) is required")
            sys.exit(2)
        images_dir, manifest_path = _resolve_paths_config(args)

    if not images_dir.exists():
        print(f"❌ Images dir not found: {images_dir}")
        sys.exit(1)
    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        sys.exit(1)

    run_triage(images_dir, manifest_path, args.batch)


if __name__ == "__main__":
    main()
