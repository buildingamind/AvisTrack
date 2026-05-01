#!/usr/bin/env python3
"""
tools/pick_rois.py
──────────────────
Two-in-one ROI tool: **pick** corners interactively and **validate** an
existing ``camera_rois.json`` before sampling.

Default command is **pick** (just ``python tools/pick_rois.py --config ...``).
Use ``validate`` sub-command to check an existing ROI file.

Usage — pick (default)
~~~~~~~~~~~~~~~~~~~~~~
    python tools/pick_rois.py --config configs/wave3_collective.yaml
    python tools/pick_rois.py --config ... --modality all
    python tools/pick_rois.py pick --config ...          # explicit sub-command

Usage — validate
~~~~~~~~~~~~~~~~
    python tools/pick_rois.py validate --config configs/wave3_collective.yaml

Controls (shown in the left panel of the picker window)
    Left-click         place next corner  (① → ② → ③ → ④)
    Z / Ctrl+Z         undo last corner
    R                  reset all 4 corners
    N                  jump to a random frame
    ← / A              go to previous video
    → / D              save & go to next video  (same as S / Enter)
    S / Enter          save corners & go to next video
    Q / Escape         quit picker
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov"}

# Human-friendly corner names (ASCII-safe for cv2.putText fallback)
CORNER_NAMES = ["Upper-Left", "Upper-Right", "Lower-Right", "Lower-Left"]

# Corner colours — defined as RGB, converted to BGR for OpenCV
_CORNER_RGB = [
    (76, 201, 79),     # 1: green
    (255, 176, 46),    # 2: amber
    (239, 83, 80),     # 3: coral
    (178, 120, 255),   # 4: violet
]
CORNER_COLORS_BGR = [(b, g, r) for r, g, b in _CORNER_RGB]
CORNER_COLORS_RGB = list(_CORNER_RGB)

LINE_COLOR_BGR = (0, 220, 255)  # cyan-yellow polygon edges


# ══════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════════════════════════════════

def find_videos(root: Path, modality: Optional[str] = None) -> list[Path]:
    """
    Recursively find video files.  If *modality* is given ("rgb" / "ir"),
    only return files whose stem contains that keyword (case-insensitive).
    """
    keyword = modality.upper() if modality else None
    videos = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if keyword and keyword not in p.stem.upper():
            continue
        videos.append(p)
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


# ══════════════════════════════════════════════════════════════════════════
#  Validate
# ══════════════════════════════════════════════════════════════════════════

def validate_roi_file(
    roi_path: str | Path,
    video_names: list[str],
) -> tuple[bool, list[str]]:
    """
    Validate a camera_rois.json against a list of expected video basenames.

    Returns (ok, messages) where *ok* is True only when every check passes.

    Checks performed
    ----------------
    1. File exists
    2. Valid JSON
    3. Top-level value is a dict
    4. Every entry has exactly 4 corner points, each being [x, y] (numbers)
    5. Every video in *video_names* has an entry (exact-name or stem match)
    """
    roi_path = Path(roi_path)
    msgs: list[str] = []

    # ── 1. Exists? ────────────────────────────────────────────────────
    if not roi_path.exists():
        msgs.append(f"❌ ROI file not found: {roi_path}")
        return False, msgs

    # ── 2. Valid JSON? ────────────────────────────────────────────────
    try:
        with open(roi_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        msgs.append(f"❌ Invalid JSON: {exc}")
        return False, msgs

    # ── 3. Top-level type ─────────────────────────────────────────────
    if not isinstance(data, dict):
        msgs.append(f"❌ Top-level must be a JSON object (dict), got {type(data).__name__}")
        return False, msgs

    msgs.append(f"✅ ROI file has {len(data)} entries")

    # ── 4. Per-entry format ───────────────────────────────────────────
    bad_format: list[str] = []
    for key, corners in data.items():
        if not isinstance(corners, list) or len(corners) != 4:
            bad_format.append(f"   {key}: expected list of 4 corners, got {type(corners).__name__} len={len(corners) if isinstance(corners, list) else '?'}")
            continue
        for i, pt in enumerate(corners):
            if (not isinstance(pt, list) or len(pt) != 2
                    or not all(isinstance(c, (int, float)) for c in pt)):
                bad_format.append(f"   {key}: corner[{i}] must be [x, y] with numbers, got {pt}")
    if bad_format:
        msgs.append(f"❌ {len(bad_format)} entries have invalid format:")
        msgs.extend(bad_format)
    else:
        msgs.append(f"✅ All {len(data)} entries have valid 4-corner format")

    # ── 5. Coverage check ────────────────────────────────────────────
    if video_names:
        # Build lookup set with both full name and stem
        roi_keys_full = set(data.keys())
        roi_keys_stem = {Path(k).stem for k in data.keys()}

        missing = []
        for name in video_names:
            if name not in roi_keys_full and Path(name).stem not in roi_keys_stem:
                missing.append(name)

        if missing:
            msgs.append(f"❌ {len(missing)}/{len(video_names)} videos have NO ROI entry:")
            for m in missing[:10]:
                msgs.append(f"   {m}")
            if len(missing) > 10:
                msgs.append(f"   … and {len(missing) - 10} more")
        else:
            msgs.append(f"✅ All {len(video_names)} videos have ROI entries")
    else:
        msgs.append("ℹ️  No video list provided — skipped coverage check")

    ok = not any(m.startswith("❌") for m in msgs)
    return ok, msgs


# ══════════════════════════════════════════════════════════════════════════
#  Pick  (interactive OpenCV — single window with PIL-rendered info panel)
# ══════════════════════════════════════════════════════════════════════════

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


class VideoPlayer:
    """Thin wrapper around cv2.VideoCapture for random-access frame reads."""

    def __init__(self, video_path: Path):
        import cv2
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {video_path}")
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.idx = -1

    def seek(self, frame_idx: int):
        import cv2
        frame_idx = max(0, min(frame_idx, self.total - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.idx = frame_idx
        return frame if ret else None

    def random_frame(self):
        return self.seek(random.randint(0, max(0, self.total - 1)))

    def release(self):
        self.cap.release()


# ── Panel configuration ──────────────────────────────────────────────────

PANEL_W = 300       # width of left info panel in pixels

# Professional colour palette  (RGB for PIL)
_P = {
    "bg":       (20, 22, 30),       # main background
    "header":   (28, 32, 46),       # title bar background
    "accent":   (100, 160, 240),    # section headings
    "text":     (180, 185, 205),    # regular text
    "dim":      (80, 87, 110),      # secondary / placeholder text
    "bright":   (220, 225, 240),    # emphasis text
    "green":    (110, 200, 90),     # success / "new" status
    "amber":    (225, 185, 75),     # warning / "review" status
    "save":     (110, 210, 130),    # save key hint
    "nav":      (130, 195, 240),    # navigation key hint
    "quit":     (210, 120, 130),    # quit key hint
    "sep":      (40, 45, 60),       # separator line
    "key":      (165, 172, 205),    # key-binding key name
}


def _load_font(size: int):
    """Load a TrueType font at *size*; fall back to PIL default."""
    from PIL import ImageFont
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _render_panel_pil(panel_h: int, corners: list, video_name: str,
                      video_idx: int, video_total: int,
                      has_existing: bool, frame_idx: int, frame_total: int):
    """Build the left info panel using PIL (crisp fonts, full Unicode)."""
    import numpy as np
    from PIL import Image, ImageDraw

    C = _P
    img  = Image.new("RGB", (PANEL_W, panel_h), C["bg"])
    draw = ImageDraw.Draw(img)

    f_title = _load_font(16)
    f_head  = _load_font(14)
    f_body  = _load_font(13)
    f_small = _load_font(11)
    f_key   = _load_font(12)

    y = 0

    # ── Title bar ─────────────────────────────────────────────────
    draw.rectangle([(0, 0), (PANEL_W, 38)], fill=C["header"])
    draw.text((14, 10), "AvisTrack  ROI Picker", fill=C["accent"], font=f_title)
    y = 46

    # ── Video info ────────────────────────────────────────────────
    draw.text((14, y), f"VIDEO  {video_idx} / {video_total}",
              fill=C["bright"], font=f_head)
    y += 24

    short = video_name if len(video_name) < 38 else "\u2026" + video_name[-35:]
    draw.text((14, y), short, fill=C["dim"], font=f_small)
    y += 18

    sc = C["amber"] if has_existing else C["green"]
    sl = "Review  (has ROI)" if has_existing else "New"
    draw.text((14, y), f"Status:  {sl}", fill=sc, font=f_body)
    y += 20

    draw.text((14, y), f"Frame  {frame_idx} / {frame_total}",
              fill=C["dim"], font=f_small)
    y += 26

    # ── Separator ─────────────────────────────────────────────────
    draw.line([(10, y), (PANEL_W - 10, y)], fill=C["sep"], width=1)
    y += 12

    # ── Corners ───────────────────────────────────────────────────
    draw.text((14, y), "CORNERS", fill=C["accent"], font=f_head)
    y += 26

    for i in range(4):
        if i < len(corners):
            cx, cy = corners[i]
            bullet = "\u25CF"          # ● filled circle
            txt = f" {bullet}  {i+1}. {CORNER_NAMES[i]}   ({cx}, {cy})"
            clr = CORNER_COLORS_RGB[i]
        elif i == len(corners):
            bullet = "\u25CB"          # ○ hollow circle
            txt = f" {bullet}  {i+1}. {CORNER_NAMES[i]}   (click)"
            clr = C["bright"]
        else:
            txt = f"      {i+1}. {CORNER_NAMES[i]}   \u2014"
            clr = C["dim"]
        draw.text((14, y), txt, fill=clr, font=f_body)
        y += 22

    if len(corners) == 4:
        y += 4
        draw.text((14, y), "\u2713  All corners set!", fill=C["green"], font=f_body)
        y += 24
    else:
        y += 10

    # ── Separator ─────────────────────────────────────────────────
    draw.line([(10, y), (PANEL_W - 10, y)], fill=C["sep"], width=1)
    y += 12

    # ── Controls ──────────────────────────────────────────────────
    draw.text((14, y), "CONTROLS", fill=C["accent"], font=f_head)
    y += 26

    controls = [
        ("Click",      "Place corner",   C["text"]),
        ("Z",          "Undo last",      C["text"]),
        ("R",          "Reset all",      C["text"]),
        ("N",          "Random frame",   C["text"]),
        None,                                           # spacer
        ("S / Enter",  "Save & next",    C["save"]),
        ("\u2190 / A", "Prev video",     C["nav"]),
        ("\u2192 / D", "Next video",     C["nav"]),
        ("Q / Esc",    "Quit",           C["quit"]),
    ]
    for item in controls:
        if item is None:
            y += 8
            continue
        key, desc, clr = item
        draw.text((18, y),  key,  fill=C["key"], font=f_key)
        draw.text((110, y), desc, fill=clr,       font=f_key)
        y += 20

    # Convert PIL (RGB) → OpenCV (BGR)
    arr = np.array(img)
    return arr[:, :, ::-1].copy()


def _render_panel_cv(panel_h: int, corners: list, video_name: str,
                     video_idx: int, video_total: int,
                     has_existing: bool, frame_idx: int, frame_total: int):
    """Fallback panel using cv2.putText (no Unicode, less polished)."""
    import cv2
    import numpy as np

    panel = np.zeros((panel_h, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (30, 22, 20)     # dark bg (BGR)

    y = 0
    def _t(txt, color=(205, 185, 180), scale=0.42, thick=1, gap=20):
        nonlocal y;  y += gap
        cv2.putText(panel, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    _t("AvisTrack ROI Picker", (240, 160, 100), 0.48, 1, 28)
    _t(f"Video {video_idx}/{video_total}", (240, 225, 220), 0.46, 1, 24)
    short = video_name if len(video_name) < 34 else "..." + video_name[-31:]
    _t(short, (110, 87, 80), 0.34, 1, 18)
    sc = (75, 185, 225) if has_existing else (90, 200, 110)
    sl = "Review (has ROI)" if has_existing else "New"
    _t(f"Status: {sl}", sc, 0.40, 1, 18)
    _t(f"Frame: {frame_idx}/{frame_total}", (110, 87, 80), 0.36, 1, 18)

    y += 12
    _t("CORNERS", (240, 160, 100), 0.46, 1, 20)
    for i in range(4):
        if i < len(corners):
            cx, cy = corners[i]
            txt = f"  * {i+1}. {CORNER_NAMES[i]}  ({cx},{cy})"
            clr = CORNER_COLORS_BGR[i]
        elif i == len(corners):
            txt = f"  > {i+1}. {CORNER_NAMES[i]}  (click)"
            clr = (240, 225, 220)
        else:
            txt = f"    {i+1}. {CORNER_NAMES[i]}  -"
            clr = (110, 87, 80)
        _t(txt, clr, 0.37, 1, 18)
    if len(corners) == 4:
        _t("All corners set!", (90, 200, 110), 0.40, 1, 22)

    y += 12
    _t("CONTROLS", (240, 160, 100), 0.46, 1, 20)
    for k, d in [("Click","place corner"),("Z","undo"),("R","reset"),
                 ("N","random frame"),("",""),("S/Enter","save & next"),
                 ("<-/A","prev video"),("->/D","next video"),("Q/Esc","quit")]:
        if not k:
            y += 6;  continue
        _t(f"  {k:11s} {d}", (205, 185, 180), 0.35, 1, 17)

    return panel


def _render_panel(panel_h, corners, video_name, video_idx, video_total,
                  has_existing, frame_idx, frame_total):
    """Route to PIL renderer or cv2 fallback."""
    fn = _render_panel_pil if _HAS_PIL else _render_panel_cv
    return fn(panel_h, corners, video_name, video_idx, video_total,
              has_existing, frame_idx, frame_total)


# ── Video frame overlay ──────────────────────────────────────────────────

def _draw_video_frame(frame, corners):
    """Draw corner markers + progressive polygon lines on the video frame."""
    import cv2
    display = frame.copy()

    # Progressive lines between consecutive placed corners
    for i in range(1, len(corners)):
        pt1 = (int(corners[i - 1][0]), int(corners[i - 1][1]))
        pt2 = (int(corners[i][0]),     int(corners[i][1]))
        cv2.line(display, pt1, pt2, LINE_COLOR_BGR, 2, cv2.LINE_AA)
    # Close polygon when all 4 are placed
    if len(corners) == 4:
        cv2.line(display,
                 (int(corners[3][0]), int(corners[3][1])),
                 (int(corners[0][0]), int(corners[0][1])),
                 LINE_COLOR_BGR, 2, cv2.LINE_AA)

    # Corner markers: white ring → coloured fill → number label
    for i, (cx, cy) in enumerate(corners):
        pt = (int(cx), int(cy))
        cv2.circle(display, pt, 9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(display, pt, 6, CORNER_COLORS_BGR[i], -1, cv2.LINE_AA)
        lbl = str(i + 1)
        lx, ly = int(cx) + 14, int(cy) - 8
        # dark outline for contrast on any background
        cv2.putText(display, lbl, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, lbl, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, CORNER_COLORS_BGR[i], 1, cv2.LINE_AA)

    return display


def _compose(panel, video_display):
    """Horizontally concatenate left panel + video frame."""
    import cv2
    import numpy as np
    ph = panel.shape[0]
    vh, vw = video_display.shape[:2]
    if vh != ph:
        scale = ph / vh
        video_display = cv2.resize(video_display, (int(vw * scale), ph),
                                   interpolation=cv2.INTER_AREA)
    return np.hstack([panel, video_display])


# ── Main interactive loop (called per-video from cmd_pick) ────────────

_GO_PREV = "__GO_PREV__"   # sentinel: user pressed ← / A


def _pick_one_video(player: VideoPlayer, video_name: str,
                    initial_corners: list | None,
                    has_existing_roi: bool,
                    video_idx: int, video_total: int,
                    win: str = "AvisTrack ROI Picker",
                    win_ready: bool = False):
    """
    Interactive picker for one video.

    Parameters
    ----------
    win        : shared window name (created by cmd_pick, reused across videos)
    win_ready  : True if the window already exists (skip namedWindow/resize)

    Returns:
        list[[x,y]×4]  — saved corners
        _GO_PREV        — user pressed ← (go back)
        None            — user pressed Q (quit)
    """
    import cv2

    corners: list[tuple[int, int]] = []
    if initial_corners:
        corners = [(int(c[0]), int(c[1])) for c in initial_corners]

    frame = player.random_frame()
    if frame is None:
        return None

    def _refresh():
        vdisplay = _draw_video_frame(frame, corners)
        panel = _render_panel(
            vdisplay.shape[0], corners, video_name,
            video_idx, video_total, has_existing_roi,
            player.idx, player.total)
        composite = _compose(panel, vdisplay)
        cv2.imshow(win, composite)

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            if x > PANEL_W:
                corners.append((x - PANEL_W, y))
                _refresh()

    if not win_ready:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1200, 680)
    cv2.setMouseCallback(win, _on_mouse)
    _refresh()

    while True:
        key = cv2.waitKey(20) & 0xFF

        # ── Undo ──────────────────────────────────────────────────
        if key in (ord("z"), ord("Z"), 26):
            if corners:
                corners.pop()
                _refresh()

        # ── Reset ─────────────────────────────────────────────────
        elif key in (ord("r"), ord("R")):
            corners.clear()
            _refresh()

        # ── Random frame ──────────────────────────────────────────
        elif key in (ord("n"), ord("N")):
            nf = player.random_frame()
            if nf is not None:
                frame = nf
                _refresh()

        # ── Previous video  ← / A ────────────────────────────────
        elif key in (81, ord("a"), ord("A")):
            return _GO_PREV

        # ── Save & next  → / D / S / Enter ───────────────────────
        elif key in (83, ord("d"), ord("D"), ord("s"), ord("S"), 13):
            if len(corners) == 4:
                return [[int(x), int(y)] for x, y in corners]
            else:
                print(f"  Need 4 corners, have {len(corners)}.")

        # ── Quit  Q / Esc ────────────────────────────────────────
        elif key in (ord("q"), ord("Q"), 27):
            return None

        # Handle window close button
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                return None
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════════════
#  Sub-command handlers
# ══════════════════════════════════════════════════════════════════════════

def _resolve_paths(args):
    """
    Resolve ``(video_dir, roi_file)`` from one of three input modes:

    * **workspace mode** – ``--workspace-yaml --chamber-id --wave-id``
      (preferred for the multi-chamber layout). Resolves through
      :func:`avistrack.workspace.load_context` so legacy and structured
      waves both land in the right place: legacy waves write
      ``camera_rois.json`` into ``_avistrack_added/{wave_id}/``,
      structured waves into ``02_Chamber_Metadata/``.
    * **legacy --config** – pulls ``drive.raw_videos`` /
      ``drive.roi_file`` out of the old single-file YAML schema.
    * **explicit --video-dir + --roi-file** – overrides either source.

    Workspace mode is selected when any of its three flags is present;
    all three are then required.
    """
    video_dir      = getattr(args, "video_dir",      None)
    roi_file       = getattr(args, "roi_file",       None)
    config         = getattr(args, "config",         None)
    workspace_yaml = getattr(args, "workspace_yaml", None)
    sources_yaml   = getattr(args, "sources_yaml",   None)
    chamber_id     = getattr(args, "chamber_id",     None)
    wave_id        = getattr(args, "wave_id",        None)

    workspace_mode = bool(workspace_yaml or chamber_id or wave_id)
    if workspace_mode:
        if config:
            print("❌ --config cannot be combined with workspace-mode flags "
                  "(--workspace-yaml / --chamber-id / --wave-id)")
            sys.exit(1)
        missing = [
            name for name, val in (
                ("--workspace-yaml", workspace_yaml),
                ("--chamber-id",     chamber_id),
                ("--wave-id",        wave_id),
            ) if not val
        ]
        if missing:
            print(f"❌ workspace mode requires {', '.join(missing)}")
            sys.exit(1)

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from avistrack.workspace import load_context
        workspace_yaml = Path(workspace_yaml)
        sources_yaml = Path(sources_yaml) if sources_yaml else \
            workspace_yaml.with_name("sources.yaml")
        if not sources_yaml.exists():
            print(f"❌ sources.yaml not found at {sources_yaml}")
            sys.exit(1)
        ctx = load_context(
            workspace_yaml=workspace_yaml,
            sources_yaml=sources_yaml,
            chamber_id=chamber_id, wave_id=wave_id,
            require_drive=True,
        )
        # Explicit overrides still win (handy when a user wants to
        # validate a different roi file under the same wave).
        return (Path(video_dir) if video_dir else ctx.wave_root,
                Path(roi_file)  if roi_file  else ctx.roi_file)

    if config and (not video_dir or not roi_file):
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from avistrack.config.loader import load_config
        cfg = load_config(config)
        if not roi_file:
            roi_file = cfg.drive.roi_file
        if not video_dir:
            video_dir = cfg.drive.raw_videos

    if not video_dir:
        print("❌ --video-dir, --config, or workspace-mode flags are required"); sys.exit(1)
    if not roi_file:
        print("❌ --roi-file, --config, or workspace-mode flags are required"); sys.exit(1)

    return Path(video_dir), Path(roi_file)


def _collect_videos(video_dir: Path, modality: str) -> list[Path]:
    """
    Collect videos respecting modality ordering.
    - "rgb"  → only RGB videos
    - "ir"   → only IR videos
    - "all"  → RGB first, then IR
    """
    if modality == "all":
        rgb = find_videos(video_dir, modality="rgb")
        ir  = find_videos(video_dir, modality="ir")
        return rgb + ir
    return find_videos(video_dir, modality=modality)


def cmd_pick(args):
    """Interactive ROI picker — shows ALL videos, supports prev/next navigation."""
    import cv2

    video_dir, roi_file = _resolve_paths(args)
    modality = getattr(args, "modality", "rgb") or "rgb"

    if not video_dir.exists():
        print(f"❌ Video directory not found: {video_dir}")
        sys.exit(1)

    rois   = load_rois(roi_file)
    videos = _collect_videos(video_dir, modality)

    if not videos:
        tag = f" {modality.upper()}" if modality != "all" else ""
        print(f"❌ No{tag} videos found under {video_dir}")
        sys.exit(1)

    n_existing = sum(1 for v in videos if v.name in rois)
    print(f"\n📂 Found {len(videos)} video(s)  "
          f"({n_existing} have ROI, "
          f"{len(videos) - n_existing} new).")
    if modality == "all":
        n_rgb = sum(1 for v in videos if "RGB" in v.stem.upper())
        n_ir  = len(videos) - n_rgb
        print(f"   Order: {n_rgb} RGB first → {n_ir} IR")
    print()

    # Create window ONCE — position is kept across all videos
    win = "AvisTrack ROI Picker"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1200, 680)
    win_ready = True

    last_corners: list | None = None
    i = 0                               # current video index

    while 0 <= i < len(videos):
        video_path = videos[i]
        has_existing = video_path.name in rois
        tag = "🟡 review" if has_existing else "🟢 new"
        print(f"[{i + 1}/{len(videos)}] {video_path.name}  ({tag})")

        try:
            player = VideoPlayer(video_path)
        except RuntimeError as e:
            print(f"  ⚠️  {e} – skipping.")
            i += 1
            continue

        # Initial corners: saved ROI > last video's corners
        if has_existing:
            init_corners = rois[video_path.name]
        else:
            init_corners = last_corners

        result = _pick_one_video(
            player, video_path.name,
            initial_corners=init_corners,
            has_existing_roi=has_existing,
            video_idx=i + 1, video_total=len(videos),
            win=win, win_ready=win_ready,
        )
        player.release()

        if result is None:
            print(f"  🛑 Quit by user.")
            break
        elif result is _GO_PREV:
            i = max(0, i - 1)
            print(f"  ← Going back to video {i + 1}")
            continue
        else:
            rois[video_path.name] = result
            last_corners = result
            save_rois(roi_file, rois)
            print(f"  ✅ Saved: {result}")
            i += 1

    # Clean up window
    try:
        cv2.destroyWindow(win)
    except Exception:
        pass

    print("\nDone.")


def cmd_validate(args):
    """Validate ROI file format and coverage."""
    video_dir_p, roi_path = _resolve_paths(args)
    modality = getattr(args, "modality", "rgb")
    if not video_dir_p.exists():
        print(f"❌ Video directory not found: {video_dir_p}")
        sys.exit(1)

    videos = find_videos(video_dir_p, modality=modality)
    if not videos:
        print(f"❌ No {modality.upper()} videos found in {video_dir_p}")
        sys.exit(1)

    print(f"📂 Checking ROI for {len(videos)} {modality.upper()} video(s) "
          f"in {video_dir_p}")
    print(f"📄 ROI file: {roi_path}\n")

    ok, msgs = validate_roi_file(roi_path, [v.name for v in videos])
    for m in msgs:
        print(f"  {m}")

    if ok:
        print(f"\n✅ ROI validation passed.")
    else:
        print(f"\n❌ ROI validation FAILED.")
        print(f"   Fix the issues above, then re‑run this check.")
        print(f"\n   To pick missing ROIs interactively:")
        print(f"     python tools/pick_rois.py --config <yaml>")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════
#  Entry point — default command = pick
# ══════════════════════════════════════════════════════════════════════════

def _add_pick_args(p):
    """Add pick-specific arguments to a parser."""
    p.add_argument("--config",    default=None,
                   help="Legacy AvisTrack YAML config (auto-fills video-dir and roi-file).")
    p.add_argument("--workspace-yaml", default=None,
                   help="Path to {workspace_root}/{chamber_type}/workspace.yaml "
                        "(workspace mode — pair with --chamber-id and --wave-id).")
    p.add_argument("--sources-yaml",   default=None,
                   help="Path to sources.yaml (default: sibling of workspace.yaml).")
    p.add_argument("--chamber-id",     default=None,
                   help="Chamber id from sources.yaml (workspace mode).")
    p.add_argument("--wave-id",        default=None,
                   help="Wave id from sources.yaml (workspace mode).")
    p.add_argument("--video-dir", default=None,
                   help="Directory to scan for videos (overrides resolved path).")
    p.add_argument("--roi-file",  default=None,
                   help="Path to camera_rois.json (overrides resolved path).")
    p.add_argument("--modality",  default="rgb",
                   choices=["rgb", "ir", "all"],
                   help="Which videos to show: rgb (default), ir, or all "
                        "(RGB first then IR).")


def main():
    parser = argparse.ArgumentParser(
        description="ROI tool — interactive corner picker (default) or validator.\n\n"
                    "Run without a subcommand to start the interactive picker:\n"
                    "  python tools/pick_rois.py --config configs/wave2_collective.yaml\n\n"
                    "Or use a subcommand explicitly:\n"
                    "  python tools/pick_rois.py pick --config ...\n"
                    "  python tools/pick_rois.py validate --config ...",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Top-level also gets pick args so `python pick_rois.py --config x` works
    _add_pick_args(parser)

    sub = parser.add_subparsers(dest="command")

    # ── pick (explicit) ───────────────────────────────────────────────
    p_pick = sub.add_parser("pick", help="Interactive 4-corner ROI picker",
                            description="Click 4 chamber corners for each video.")
    _add_pick_args(p_pick)

    # ── validate ──────────────────────────────────────────────────────
    p_val = sub.add_parser("validate",
                           help="Validate ROI file format and video coverage",
                           description="Check JSON format + every video has an entry.")
    p_val.add_argument("--roi-file",  default=None,
                       help="Path to camera_rois.json")
    p_val.add_argument("--video-dir", default=None,
                       help="Directory to scan for videos")
    p_val.add_argument("--config",    default=None,
                       help="Legacy AvisTrack YAML config (auto-fills roi-file and video-dir)")
    p_val.add_argument("--workspace-yaml", default=None,
                       help="Workspace yaml (paired with --chamber-id / --wave-id).")
    p_val.add_argument("--sources-yaml",   default=None,
                       help="Path to sources.yaml (default: sibling of workspace.yaml).")
    p_val.add_argument("--chamber-id",     default=None,
                       help="Chamber id from sources.yaml (workspace mode).")
    p_val.add_argument("--wave-id",        default=None,
                       help="Wave id from sources.yaml (workspace mode).")
    p_val.add_argument("--modality",  default="rgb", choices=["rgb", "ir"],
                       help="Filter videos by modality keyword (default: rgb)")

    args = parser.parse_args()

    if args.command == "pick" or args.command is None:
        cmd_pick(args)
    elif args.command == "validate":
        cmd_validate(args)


if __name__ == "__main__":
    main()
