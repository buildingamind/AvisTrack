#!/usr/bin/env python3
"""
tools/calibrate_time.py
───────────────────────
Build a frame ↔ burn-in time mapping for each video using sparse
Google Cloud Vision OCR.

Subcommands
~~~~~~~~~~~
  roi        Pick the OCR text region for each video (per-video, like pick_rois.py)
  calibrate  Run sparse OCR → time_calibration.json (cost estimate + preview)
  verify     Spot-check calibration accuracy against interpolated predictions

Usage
~~~~~
  # 1. Pick where the timestamp lives on screen:
  python tools/calibrate_time.py roi --config configs/wave3_collective.yaml

  # 2. Run calibration (estimates cost, shows preview, asks to confirm):
  python tools/calibrate_time.py calibrate --config configs/wave3_collective.yaml

  # 3. Verify results:
  python tools/calibrate_time.py verify --config configs/wave3_collective.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time as _time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# Allow importing avistrack from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from avistrack.config.loader import load_config  # noqa: E402

# ══════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov"}

# Google Cloud Vision pricing (USD)
VISION_FREE_TIER = 1000  # free TEXT_DETECTION calls per month
VISION_PRICE_PER_1000 = 1.50

# ══════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════════════════════════════════


def find_videos(root: Path, modality: str = "rgb") -> list[Path]:
    """Recursively find video files whose stem contains *modality*."""
    keyword = modality.upper()
    return sorted(
        p
        for p in root.rglob("*")
        if p.suffix.lower() in VIDEO_EXTENSIONS and keyword in p.stem.upper()
    )


def load_ocr_roi(roi_path: Path) -> dict | None:
    """Load ``ocr_roi.json`` → ``{video_name: [x,y,w,h]}``."""
    if not roi_path.exists():
        return None
    with open(roi_path) as f:
        return json.load(f)


def get_roi_for_video(rois: dict, video_name: str) -> list[int] | None:
    """Get ``[x, y, w, h]`` for a video, falling back to ``_default``."""
    if video_name in rois:
        return rois[video_name]
    return rois.get("_default")


def load_calibration(path: Path) -> dict:
    """Load existing calibration JSON (or return skeleton)."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"_meta": {}}


def save_calibration(path: Path, data: dict) -> None:
    """Write calibration JSON (creates parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_date_from_filename(filename: str) -> str | None:
    """Try to pull a date from the filename.

    Looks for a six-digit group and interprets it as ``ddmmyy``.
    Returns ``YYYY-MM-DD`` on success, ``None`` on failure.
    """
    match = re.search(r"(\d{6})", filename)
    if match:
        try:
            dt = datetime.strptime(match.group(1), "%d%m%y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


# ══════════════════════════════════════════════════════════════════════════
#  Google Cloud Vision
# ══════════════════════════════════════════════════════════════════════════


def init_google_client():
    """Return a ``vision.ImageAnnotatorClient``, or exit with a clear error."""
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds or not Path(creds).exists():
        print("❌ Google Cloud Vision credentials not found.\n")
        print("   Set the environment variable:")
        print(
            "     export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json"
        )
        print()
        print("   Or create a service account key at:")
        print("     https://console.cloud.google.com/apis/credentials")
        sys.exit(1)

    from google.cloud import vision  # lazy import

    return vision.ImageAnnotatorClient()


def google_ocr(client, image_crop: np.ndarray) -> tuple[str, float]:
    """Run TEXT_DETECTION on a cropped image.

    Returns ``(raw_text, confidence)`` where confidence is 0–1.
    """
    from google.cloud import vision

    ok, encoded = cv2.imencode(".png", image_crop)
    if not ok:
        return "[encode error]", 0.0

    image = vision.Image(content=encoded.tobytes())
    try:
        response = client.text_detection(image=image)
        if response.error.message:
            return f"[API error: {response.error.message}]", 0.0

        texts = response.text_annotations
        if not texts:
            return "[no text]", 0.0

        raw = texts[0].description.replace("\n", " ").strip()
        conf = 1.0
        if (
            response.full_text_annotation
            and response.full_text_annotation.pages
        ):
            page = response.full_text_annotation.pages[0]
            if page.blocks:
                conf = page.blocks[0].confidence
        return raw, conf

    except Exception as e:
        return f"[error: {e}]", 0.0


# ══════════════════════════════════════════════════════════════════════════
#  Time parsing
# ══════════════════════════════════════════════════════════════════════════


def clean_ocr_text(text: str) -> str:
    """Fix common OCR misreads and strip timezone indicators.

    Timezone abbreviations (EST, EDT, CST, PDT …) and UTC/GMT offsets
    (UTC-5, GMT+8 …) are removed because the authoritative timezone comes
    from the YAML config (e.g. ``America/New_York``).
    """
    # Capital O → zero
    text = text.replace("O", "0")
    # Keep lowercase 'o' only when it's likely a zero (surrounded by digits)
    text = re.sub(r"(?<=\d)o(?=\d)", "0", text)
    # Lowercase l → one (only when surrounded by digits)
    text = re.sub(r"(?<=\d)l(?=\d)", "1", text)

    # ── Strip timezone indicators ────────────────────────────────
    # UTC±N / GMT±N  (e.g. "UTC-5", "GMT+08", "UTC-05:00")
    text = re.sub(r"\b(?:UTC|GMT)\s*[+-]?\d{1,2}(?::?\d{2})?\b", "", text,
                  flags=re.IGNORECASE)
    # Common North-American / European abbreviations
    _TZ_ABBRS = (
        "EST", "EDT", "CST", "CDT", "MST", "MDT", "PST", "PDT",
        "AST", "ADT", "HST", "AKST", "AKDT",
        "UTC", "GMT", "BST", "CET", "CEST", "IST", "JST", "KST",
        "AEST", "AEDT", "NZST", "NZDT",
    )
    # Build pattern: match these as whole words (case-insensitive)
    abbr_pat = r"\b(?:" + "|".join(_TZ_ABBRS) + r")\b"
    text = re.sub(abbr_pat, "", text, flags=re.IGNORECASE)

    # Collapse multiple spaces left behind and strip
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


# ── Auto-detect time format ──────────────────────────────────────────────

# Common burn-in timestamp formats (order: most specific → least specific)
_KNOWN_FORMATS = [
    # With date
    ("%I:%M:%S %p %d-%b-%y",  "08:35:30 PM 18-Jun-25"),
    ("%I:%M:%S %p %d-%b-%Y",  "08:35:30 PM 18-Jun-2025"),
    ("%I:%M:%S %p %d/%m/%Y",  "08:35:30 PM 18/06/2025"),
    ("%I:%M:%S %p %d/%m/%y",  "08:35:30 PM 18/06/25"),
    ("%I:%M:%S %p %m/%d/%Y",  "08:35:30 PM 06/18/2025"),
    ("%H:%M:%S %d-%b-%y",     "20:35:30 18-Jun-25"),
    ("%H:%M:%S %d-%b-%Y",     "20:35:30 18-Jun-2025"),
    ("%H:%M:%S %d/%m/%Y",     "20:35:30 18/06/2025"),
    ("%H:%M:%S %d/%m/%y",     "20:35:30 18/06/25"),
    # Time only (ROI cropped to just the time)
    ("%I:%M:%S %p",           "08:35:30 PM"),
    ("%H:%M:%S",              "20:35:30"),
    ("%I:%M %p",              "08:35 PM"),
    ("%H:%M",                 "20:35"),
]


def detect_time_format(text: str) -> tuple[str, bool]:
    """Try all known formats against *text* and return the first match.

    Parameters
    ----------
    text : str
        Raw or cleaned OCR text.

    Returns
    -------
    (format_str, has_date) : tuple[str, bool]
        The matching ``strptime`` format string, and whether the format
        includes a date component (``%d``, ``%b``, ``%Y``, etc.).
        If nothing matches, returns ``("", False)``.
    """
    cleaned = clean_ocr_text(text)
    for fmt, _ in _KNOWN_FORMATS:
        try:
            datetime.strptime(cleaned, fmt)
            has_date = any(c in fmt for c in ("%d", "%b", "%y", "%Y"))
            return fmt, has_date
        except ValueError:
            continue
    return "", False


def _fmt_description(fmt: str) -> str:
    """Return a human-readable example for a strptime format string."""
    for f, example in _KNOWN_FORMATS:
        if f == fmt:
            return example
    return fmt


def parse_time(
    text: str,
    time_format: str,
    base_date: str,
    tz: ZoneInfo,
    day_offset: int = 0,
) -> tuple[datetime | None, float]:
    """Parse OCR text into a tz-aware datetime.

    Parameters
    ----------
    text : str
        Raw OCR text (will be cleaned first).
    time_format : str
        ``strptime`` format for the burn-in (e.g. ``%I:%M:%S %p``).
        If the format includes date tokens (``%d``, ``%b``, ``%Y``, etc.),
        the date is extracted from the OCR text itself; otherwise from
        *base_date* + *day_offset*.
    base_date : str
        ``YYYY-MM-DD`` for the start date of the video.
    tz : ZoneInfo
        Target timezone.
    day_offset : int
        Number of midnights crossed since ``base_date``.

    Returns ``(datetime, unix_timestamp)`` or ``(None, 0.0)`` on failure.
    """
    cleaned = clean_ocr_text(text)
    try:
        parsed = datetime.strptime(cleaned, time_format)
    except ValueError:
        return None, 0.0

    # Check whether the format includes date components
    fmt_has_date = any(c in time_format for c in ("%d", "%b", "%y", "%Y"))

    if fmt_has_date:
        # Date comes from the parsed result itself
        dt_naive = parsed
    else:
        # Date comes from base_date + day_offset
        base = datetime.strptime(base_date, "%Y-%m-%d")
        dt_naive = base + timedelta(
            days=day_offset,
            hours=parsed.hour,
            minutes=parsed.minute,
            seconds=parsed.second,
        )

    dt_aware = dt_naive.replace(tzinfo=tz)
    return dt_aware, dt_aware.timestamp()


def detect_day_offset(
    prev_time: datetime | None,
    curr_time: datetime | None,
    current_offset: int,
) -> int:
    """Detect midnight crossing.

    If the parsed time jumped backward by more than 12 hours compared to the
    previous sample, assume we crossed midnight and increment the day offset.
    """
    if prev_time is None or curr_time is None:
        return current_offset
    diff = (curr_time - prev_time).total_seconds()
    if diff < -43_200:  # > 12 h backward jump
        return current_offset + 1
    return current_offset


# ══════════════════════════════════════════════════════════════════════════
#  Subcommand: roi  —  pick OCR region interactively (per-video)
#                      Professional PIL panel, click-to-zoom draw workflow
# ══════════════════════════════════════════════════════════════════════════

# ── Panel configuration ──────────────────────────────────────────────────

ROI_PANEL_W = 320   # width of info panel in pixels

# Professional colour palette (RGB for PIL, same scheme as pick_rois.py)
_CP = {
    "bg":       (20, 22, 30),
    "header":   (28, 32, 46),
    "accent":   (100, 160, 240),
    "text":     (180, 185, 205),
    "dim":      (80, 87, 110),
    "bright":   (220, 225, 240),
    "green":    (110, 200, 90),
    "amber":    (225, 185, 75),
    "save":     (110, 210, 130),
    "nav":      (130, 195, 240),
    "quit":     (210, 120, 130),
    "sep":      (40, 45, 60),
    "key":      (165, 172, 205),
    "roi_box":  (80, 230, 110),
}


def _load_font_roi(size: int):
    """Load a TrueType font at *size*; fall back to PIL default."""
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


def _render_roi_panel_pil(
    panel_h: int,
    video_name: str,
    video_idx: int,
    video_total: int,
    roi: list[int] | None,
    frame: np.ndarray | None,
):
    """Build the left info panel using PIL (crisp fonts, ROI preview)."""
    C = _CP
    img  = Image.new("RGB", (ROI_PANEL_W, panel_h), C["bg"])
    draw = ImageDraw.Draw(img)

    f_title = _load_font_roi(16)
    f_head  = _load_font_roi(14)
    f_body  = _load_font_roi(13)
    f_small = _load_font_roi(11)
    f_key   = _load_font_roi(12)

    y = 0

    # ── Title bar ─────────────────────────────────────────────────
    draw.rectangle([(0, 0), (ROI_PANEL_W, 38)], fill=C["header"])
    draw.text((14, 10), "AvisTrack  OCR ROI Picker", fill=C["accent"], font=f_title)
    y = 46

    # ── Video info ────────────────────────────────────────────────
    draw.text((14, y), f"VIDEO  {video_idx + 1} / {video_total}",
              fill=C["bright"], font=f_head)
    y += 24

    short = video_name if len(video_name) < 40 else "\u2026" + video_name[-37:]
    draw.text((14, y), short, fill=C["dim"], font=f_small)
    y += 18

    if roi:
        draw.text((14, y), f"Status:  Has ROI", fill=C["amber"], font=f_body)
    else:
        draw.text((14, y), f"Status:  No ROI", fill=C["green"], font=f_body)
    y += 26

    # ── Separator ─────────────────────────────────────────────────
    draw.line([(10, y), (ROI_PANEL_W - 10, y)], fill=C["sep"], width=1)
    y += 12

    # ── ROI info ──────────────────────────────────────────────────
    draw.text((14, y), "CURRENT ROI", fill=C["accent"], font=f_head)
    y += 24

    if roi:
        rx, ry, rw, rh = roi
        draw.text((14, y), f"Position:  ({rx}, {ry})", fill=C["text"], font=f_body)
        y += 20
        draw.text((14, y), f"Size:      {rw} \u00d7 {rh} px", fill=C["text"], font=f_body)
        y += 24

        # ── Zoomed ROI preview embedded in the panel ──────────────
        if frame is not None:
            crop = frame[ry : ry + rh, rx : rx + rw]
            if crop.size > 0:
                # Scale to fit panel width with padding
                max_w = ROI_PANEL_W - 28
                max_h = 80
                crop_h, crop_w = crop.shape[:2]
                s = min(max_w / max(crop_w, 1), max_h / max(crop_h, 1))
                pw = max(1, int(crop_w * s))
                ph = max(1, int(crop_h * s))
                pil_crop = Image.fromarray(crop[:, :, ::-1])  # BGR→RGB
                pil_crop = pil_crop.resize((pw, ph), Image.NEAREST)

                # Center in panel
                cx = (ROI_PANEL_W - pw) // 2
                # Green border
                draw.rectangle(
                    [(cx - 3, y - 3), (cx + pw + 2, y + ph + 2)],
                    outline=C["roi_box"], width=2,
                )
                img.paste(pil_crop, (cx, y))
                y += ph + 8

                draw.text((14, y), "ROI Preview", fill=C["dim"], font=f_small)
                y += 20
    else:
        draw.text((14, y), "(none — press D to draw)", fill=C["dim"], font=f_body)
        y += 28

    # ── Separator ─────────────────────────────────────────────────
    draw.line([(10, y), (ROI_PANEL_W - 10, y)], fill=C["sep"], width=1)
    y += 12

    # ── Controls ──────────────────────────────────────────────────
    draw.text((14, y), "CONTROLS", fill=C["accent"], font=f_head)
    y += 26

    controls = [
        ("D / N",       "Draw new ROI",       C["text"]),
        ("",            " click to zoom in",  C["dim"]),
        ("",            " drag on zoomed view",C["dim"]),
        None,
        ("Enter/Space", "Accept & next",      C["save"]),
        ("\u2190 / A",  "Prev video",         C["nav"]),
        ("\u2192",      "Next video",         C["nav"]),
        ("Q / Esc",     "Save & quit",        C["quit"]),
    ]
    for item in controls:
        if item is None:
            y += 8
            continue
        key, desc, clr = item
        if key:
            draw.text((18, y), key, fill=C["key"], font=f_key)
        draw.text((120, y), desc, fill=clr, font=f_key)
        y += 20

    # Convert PIL (RGB) → OpenCV (BGR)
    arr = np.array(img)
    return arr[:, :, ::-1].copy()


def _render_roi_panel_cv(
    panel_h: int,
    video_name: str,
    video_idx: int,
    video_total: int,
    roi: list[int] | None,
    frame: np.ndarray | None,
):
    """Fallback panel using cv2.putText (no Unicode, less polished)."""
    panel = np.zeros((panel_h, ROI_PANEL_W, 3), dtype=np.uint8)
    panel[:] = (30, 22, 20)

    y = 0
    def _t(txt, color=(205, 185, 180), sc=0.42, thick=1, gap=20):
        nonlocal y;  y += gap
        cv2.putText(panel, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    sc, color, thick, cv2.LINE_AA)

    _t("AvisTrack OCR ROI Picker", (240, 160, 100), 0.48, 1, 28)
    _t(f"Video {video_idx + 1}/{video_total}", (240, 225, 220), 0.46, 1, 24)
    short = video_name if len(video_name) < 34 else "..." + video_name[-31:]
    _t(short, (110, 87, 80), 0.34, 1, 18)
    if roi:
        _t(f"ROI: ({roi[0]},{roi[1]}) {roi[2]}x{roi[3]}", (75, 185, 225), 0.40, 1, 18)
    else:
        _t("ROI: (none)", (90, 200, 110), 0.40, 1, 18)

    y += 12
    _t("CONTROLS", (240, 160, 100), 0.46, 1, 20)
    for k, d in [("D/N","draw new ROI"),("","  click to zoom in"),
                 ("","  drag on zoomed view"),("",""),
                 ("Enter","accept & next"),("<-/A","prev video"),
                 ("->/D","next video"),("Q/Esc","save & quit")]:
        if not k and not d:
            y += 6; continue
        _t(f"  {k:11s} {d}", (205, 185, 180), 0.35, 1, 17)

    return panel


def _render_roi_panel(panel_h, video_name, video_idx, video_total, roi, frame):
    """Route to PIL renderer or cv2 fallback."""
    fn = _render_roi_panel_pil if _HAS_PIL else _render_roi_panel_cv
    return fn(panel_h, video_name, video_idx, video_total, roi, frame)


def _compose_roi(panel: np.ndarray, video_display: np.ndarray) -> np.ndarray:
    """Horizontally concatenate left panel + video frame."""
    ph = panel.shape[0]
    vh, vw = video_display.shape[:2]
    if vh != ph:
        sc = ph / vh
        video_display = cv2.resize(video_display, (int(vw * sc), ph),
                                   interpolation=cv2.INTER_AREA)
    return np.hstack([panel, video_display])


def _draw_roi_on_frame(frame: np.ndarray, roi: list[int] | None,
                       scale: float) -> np.ndarray:
    """Draw the ROI rectangle on the scaled video frame (no panel)."""
    display = cv2.resize(frame, None, fx=scale, fy=scale)
    if roi:
        rx, ry, rw, rh = roi
        sx, sy = int(rx * scale), int(ry * scale)
        sw, sh = int(rw * scale), int(rh * scale)
        cv2.rectangle(display, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
        cv2.putText(display, "ROI", (sx, sy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return display


def _pick_roi_for_frame(
    frame: np.ndarray,
    video_name: str,
    idx: int,
    total: int,
    prev_roi: list[int] | None,
    win_name: str,
    scale: float,
) -> tuple[list[int] | None, str]:
    """Show video frame + PIL panel, let user pick / keep / skip ROI.

    Press **D** → click near the timestamp → zoomed view appears → drag
    to draw the ROI rectangle.  Much easier than drawing on the tiny
    scaled-down overview.

    Returns ``(roi, action)`` where action is ``"next"`` | ``"prev"`` | ``"quit"``.
    """

    # Mutable click state for mouse callback
    _click: dict = {"pos": None}

    def _on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Offset by panel width so coords map to video area
            _click["pos"] = (x - ROI_PANEL_W, y)

    def _refresh(current_roi):
        vdisplay = _draw_roi_on_frame(frame, current_roi, scale)
        panel = _render_roi_panel(
            vdisplay.shape[0], video_name, idx, total, current_roi, frame
        )
        composite = _compose_roi(panel, vdisplay)
        cv2.imshow(win_name, composite)

    while True:
        _refresh(prev_roi)

        # Disable mouse callback during normal viewing
        cv2.setMouseCallback(win_name, lambda *_a: None)
        key = cv2.waitKeyEx(0)

        # ── Q / Esc → quit ──────────────────────────────────────
        if key in (ord("q"), ord("Q"), 27):
            return prev_roi, "quit"

        # ── Enter / Space / S → accept & next ───────────────────
        if key in (13, 32, ord("s"), ord("S")):
            return prev_roi, "next"

        # ── A / Left-arrow → prev ───────────────────────────────
        if key in (ord("a"), ord("A"), 65361, 2424832):
            return prev_roi, "prev"

        # ── Right-arrow → next (same as accept) ─────────────────
        if key in (65363, 2555904):
            return prev_roi, "next"

        # ── D / N → draw new ROI via click-to-zoom ──────────────
        if key in (ord("d"), ord("D"), ord("n"), ord("N")):

            # Phase 1: dim the video area, instruction overlay
            vdisplay = cv2.resize(frame, None, fx=scale, fy=scale)
            overlay = vdisplay.copy()
            cv2.rectangle(overlay, (0, 0),
                          (vdisplay.shape[1], vdisplay.shape[0]),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, vdisplay, 0.5, 0, vdisplay)

            msgs = [
                "Click near the timestamp to zoom in,",
                "then drag to draw the ROI rectangle.",
                "Press ESC to cancel.",
            ]
            cx_text = vdisplay.shape[1] // 2
            cy_text = vdisplay.shape[0] // 2 - 30
            for i, m in enumerate(msgs):
                sz = cv2.getTextSize(m, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.putText(vdisplay, m,
                            (cx_text - sz[0] // 2, cy_text + i * 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)

            # Show with panel
            panel = _render_roi_panel(
                vdisplay.shape[0], video_name, idx, total, prev_roi, frame
            )
            composite = _compose_roi(panel, vdisplay)

            _click["pos"] = None
            cv2.setMouseCallback(win_name, _on_click)
            cv2.imshow(win_name, composite)

            # Wait for click or ESC
            while _click["pos"] is None:
                k = cv2.waitKeyEx(30)
                if k == 27:
                    break

            cv2.setMouseCallback(win_name, lambda *_a: None)

            if _click["pos"] is None:
                continue  # ESC → cancelled, return to overview

            # Phase 2: crop around click, zoom in, selectROI
            dx, dy = _click["pos"]
            ox = int(dx / scale)
            oy = int(dy / scale)

            fh, fw = frame.shape[:2]
            crop_hw, crop_hh = 250, 150  # half-size in original pixels
            x1 = max(0, ox - crop_hw)
            y1 = max(0, oy - crop_hh)
            x2 = min(fw, ox + crop_hw)
            y2 = min(fh, oy + crop_hh)
            region = frame[y1:y2, x1:x2]

            # Adaptive zoom: fit zoomed crop to ~1400 px wide
            rw_region = max(x2 - x1, 1)
            zoom = min(3.0, 1400 / rw_region)
            zoomed = cv2.resize(region, None, fx=zoom, fy=zoom,
                                interpolation=cv2.INTER_LINEAR)

            roi_z = cv2.selectROI(
                win_name, zoomed, showCrosshair=True, fromCenter=False
            )

            if roi_z[2] > 0 and roi_z[3] > 0:
                # Convert zoomed-crop coords → original coords
                rx = x1 + int(round(roi_z[0] / zoom))
                ry = y1 + int(round(roi_z[1] / zoom))
                rw = int(round(roi_z[2] / zoom))
                rh = int(round(roi_z[3] / zoom))
                # Clamp to frame bounds
                rx = max(0, min(rx, fw - 1))
                ry = max(0, min(ry, fh - 1))
                rw = min(rw, fw - rx)
                rh = min(rh, fh - ry)
                new_roi = [rx, ry, rw, rh]

                # Show result in panel before continuing
                prev_roi = new_roi
                _refresh(new_roi)
                cv2.waitKey(800)   # brief flash so user sees result

                return new_roi, "next"

            # Selection cancelled — show frame again
            continue


def cmd_roi(args):
    """Interactive per-video OCR region picker with PIL info panel.

    Iterates through all videos (like pick_rois.py), letting you set or
    confirm the timestamp crop region for each one.  The previous video's
    ROI carries forward as default.
    """
    cfg = load_config(args.config)

    raw_dir = Path(cfg.drive.raw_videos)
    if not raw_dir.exists():
        print(f"❌ raw_videos not found: {raw_dir}")
        sys.exit(1)

    videos = find_videos(raw_dir, modality="rgb")
    if not videos:
        print(f"❌ No RGB videos found in {raw_dir}")
        sys.exit(1)

    # Determine save path
    roi_path = (
        Path(cfg.drive.ocr_roi)
        if cfg.drive.ocr_roi
        else Path(cfg.drive.metadata) / "ocr_roi.json"
    )

    # Load existing ROI data
    existing: dict = {}
    if roi_path.exists():
        with open(roi_path) as f:
            existing = json.load(f)

    # Compute display scale from first video
    cap0 = cv2.VideoCapture(str(videos[0]))
    ret0, frame0 = cap0.read()
    cap0.release()
    if not ret0:
        print("❌ Cannot read first video")
        sys.exit(1)
    h, w = frame0.shape[:2]
    display_w = min(1280, w)
    scale = display_w / w

    print()
    print(f"  📹 {len(videos)} RGB video(s) found")
    if _HAS_PIL:
        print(f"  🎨 PIL panel enabled — controls shown on-screen")
    else:
        print(f"  ⚠️  PIL not available, using basic fallback panel")
    print()

    win_name = "AvisTrack OCR ROI Picker"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, display_w + ROI_PANEL_W, int(h * scale))

    idx = 0
    prev_roi: list[int] | None = existing.get("_default")

    while 0 <= idx < len(videos):
        video_path = videos[idx]
        vname = video_path.name

        # If this video already has a saved ROI, use it; else carry forward
        current_roi = existing.get(vname, prev_roi)

        # Read first frame
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"  ⚠️  Cannot read {vname}, skipping")
            idx += 1
            continue

        roi, action = _pick_roi_for_frame(
            frame, vname, idx, len(videos), current_roi, win_name, scale
        )

        if action == "quit":
            # Save what we have and exit
            if roi:
                existing[vname] = roi
            break

        if action == "next":
            if roi:
                existing[vname] = roi
                prev_roi = roi  # carry forward
            idx += 1
            # Save after every forward move (crash-safe)
            roi_path.parent.mkdir(parents=True, exist_ok=True)
            with open(roi_path, "w") as f:
                json.dump(existing, f, indent=2)

        elif action == "prev":
            idx = max(0, idx - 1)

    cv2.destroyAllWindows()

    # Final save
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    with open(roi_path, "w") as f:
        json.dump(existing, f, indent=2)

    saved_count = sum(1 for k in existing if k != "_default")
    print(f"\n✅ Saved OCR ROI for {saved_count} video(s) → {roi_path}")



# ══════════════════════════════════════════════════════════════════════════
#  Subcommand: calibrate  —  sparse OCR → time_calibration.json
# ══════════════════════════════════════════════════════════════════════════


def _print_cost_box(video_info: list[dict], remaining: list[dict], interval: int):
    """Print the cost-estimate box and return (total_calls, remaining_calls)."""
    total_calls = sum(vi["n_calls"] for vi in video_info)
    remaining_calls = sum(vi["n_calls"] for vi in remaining)
    skip_count = len(video_info) - len(remaining)
    cost_no_free = remaining_calls / 1000 * VISION_PRICE_PER_1000
    cost_with_free = (
        max(0, remaining_calls - VISION_FREE_TIER) / 1000 * VISION_PRICE_PER_1000
    )

    W = 62  # inner width
    sep = "═" * W
    print()
    print(f"╔{sep}╗")
    print(f"║{'AvisTrack Time Calibration — Cost Estimate':^{W}}║")
    print(f"╠{sep}╣")
    print(f"║  Videos:            {len(video_info):>5}{'':<{W-27}}║")
    print(f"║  Sample interval:   every {interval} frames{'':<{W-33-len(str(interval))}}║")
    print(f"║{'':<{W}}║")

    # Video table (first 10)
    for vi in video_info[:10]:
        name = vi["path"].name
        if len(name) > 33:
            name = "…" + name[-32:]
        tag = " [done]" if vi["already_done"] else ""
        cell = f"  {name:<35s} {vi['total_frames']:>9,} fr {vi['n_calls']:>5} calls{tag}"
        print(f"║{cell:<{W}}║")
    if len(video_info) > 10:
        extra = f"  … and {len(video_info) - 10} more"
        print(f"║{extra:<{W}}║")

    print(f"║{'':<{W}}║")
    if skip_count:
        sk = f"  Already calibrated: {skip_count} (will skip)"
        print(f"║{sk:<{W}}║")
    rem = f"  Remaining: {len(remaining)} videos → {remaining_calls:,} API calls"
    print(f"║{rem:<{W}}║")
    print(f"║{'':<{W}}║")
    cost = f"  Est. cost: ${cost_with_free:.2f} (with free tier) – ${cost_no_free:.2f} (without)"
    print(f"║{cost:<{W}}║")
    tier = f"  (Free tier: {VISION_FREE_TIER}/mo, ${VISION_PRICE_PER_1000:.2f}/1000 after)"
    print(f"║{tier:<{W}}║")
    print(f"╚{sep}╝")
    print()

    return total_calls, remaining_calls


def cmd_calibrate(args):
    """Run sparse OCR calibration."""
    cfg = load_config(args.config)
    interval: int = args.interval

    # ── Resolve paths ────────────────────────────────────────────
    raw_dir = Path(cfg.drive.raw_videos)
    roi_path = Path(cfg.drive.ocr_roi) if cfg.drive.ocr_roi else None
    cal_path = (
        Path(cfg.drive.time_calibration) if cfg.drive.time_calibration else None
    )

    if not raw_dir.exists():
        print(f"❌ raw_videos not found: {raw_dir}")
        sys.exit(1)
    if not roi_path or not roi_path.exists():
        print("❌ OCR ROI not set. Run the `roi` subcommand first:")
        print(f"   python tools/calibrate_time.py roi --config {args.config}")
        sys.exit(1)
    if not cal_path:
        print("❌ time_calibration path not set in config")
        sys.exit(1)

    # ── Load auxiliary data ──────────────────────────────────────
    rois = load_ocr_roi(roi_path)
    if not rois:
        print("❌ ocr_roi.json is empty")
        sys.exit(1)

    tz_str = cfg.time.timezone
    t_fmt = cfg.time.time_format   # may be "auto"
    tz = ZoneInfo(tz_str)

    # ── Discover videos ──────────────────────────────────────────
    videos = find_videos(raw_dir, modality="rgb")
    if not videos:
        print("❌ No RGB videos found")
        sys.exit(1)

    # Load existing calibration (for resume / --force)
    calibration = load_calibration(cal_path)

    # ── Probe each video ─────────────────────────────────────────
    print("📹 Probing videos …")
    video_info: list[dict] = []
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total < 1:
            continue
        n_calls = (total // interval) + 1
        already = (v.name in calibration) and not args.force
        video_info.append(
            {
                "path": v,
                "fps": fps,
                "total_frames": total,
                "n_calls": n_calls,
                "already_done": already,
            }
        )

    if not video_info:
        print("❌ No valid videos found")
        sys.exit(1)

    remaining = [vi for vi in video_info if not vi["already_done"]]

    # ── Cost estimate box ────────────────────────────────────────
    _print_cost_box(video_info, remaining, interval)

    if not remaining:
        print("✅ All videos already calibrated. Use --force to redo.")
        return

    # ── First confirmation ───────────────────────────────────────
    ans = input("Continue with calibration? [y/N]: ").strip().lower()
    if ans != "y":
        print("Aborted.")
        return

    # ── Init Google Vision ───────────────────────────────────────
    client = init_google_client()

    # ── Preview (3 samples from first remaining video) ───────────
    #    Also auto-detects time_format if set to "auto" or if
    #    the configured format fails to parse.
    print()
    print("── Preview (3 samples from first remaining video) ──")
    preview_vi = remaining[0]
    roi = get_roi_for_video(rois, preview_vi["path"].name)
    if not roi:
        print("❌ No OCR ROI for this video")
        sys.exit(1)
    rx, ry, rw, rh = roi

    # Date for the preview video
    date_str = extract_date_from_filename(preview_vi["path"].name)
    if not date_str:
        date_str = input(
            f"  Cannot extract date from '{preview_vi['path'].name}'.\n"
            f"  Enter date (YYYY-MM-DD): "
        ).strip()

    cap = cv2.VideoCapture(str(preview_vi["path"]))
    preview_ok = 0
    preview_texts: list[str] = []   # collect OCR texts for auto-detect

    for pi in range(3):
        frame_idx = pi * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[ry : ry + rh, rx : rx + rw]
        text, conf = google_ocr(client, crop)
        preview_texts.append(text)
        dt, unix = parse_time(text, t_fmt, date_str, tz)
        if dt:
            preview_ok += 1
            dt_str = dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            print(f"  Frame {frame_idx:>8,}:  OCR=\"{text}\"  →  {dt_str}  ✓")
        else:
            print(f"  Frame {frame_idx:>8,}:  OCR=\"{text}\"  →  (parse failed)  ✗")
    cap.release()

    # ── Auto-detect format if preview failed or format is "auto" ─
    if preview_ok == 0 or t_fmt.lower() == "auto":
        print()
        detected_fmt = ""
        for ocr_text in preview_texts:
            detected_fmt, has_date = detect_time_format(ocr_text)
            if detected_fmt:
                break

        if detected_fmt:
            example = _fmt_description(detected_fmt)
            print(f"  🔍 Auto-detected time format: \"{detected_fmt}\"")
            print(f"     Example match: {example}")

            if detected_fmt != t_fmt:
                t_fmt = detected_fmt
                # Re-run preview with the detected format
                print()
                print("── Re-running preview with detected format ──")
                cap = cv2.VideoCapture(str(preview_vi["path"]))
                preview_ok = 0
                for pi in range(3):
                    frame_idx = pi * interval
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    crop = frame[ry : ry + rh, rx : rx + rw]
                    text, conf = google_ocr(client, crop)
                    dt, unix = parse_time(text, t_fmt, date_str, tz)
                    if dt:
                        preview_ok += 1
                        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                        print(f"  Frame {frame_idx:>8,}:  OCR=\"{text}\"  →  {dt_str}  ✓")
                    else:
                        print(f"  Frame {frame_idx:>8,}:  OCR=\"{text}\"  →  (parse failed)  ✗")
                cap.release()
        else:
            print("  ⚠️  Auto-detection failed — none of the known formats matched.")
            print(f'     OCR samples: {preview_texts}')
            print("     Set time_format manually in the YAML config.")
            ans = input("  Continue anyway? [y/N]: ").strip().lower()
            if ans != "y":
                print("Aborted.")
                return

    # ── Second confirmation ──────────────────────────────────────
    if preview_ok == 0:
        print()
        print("  ⚠️  All preview samples failed to parse!")
        print(f'     time_format: "{t_fmt}"')
        ans = input("  Continue anyway? [y/N]: ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return
    else:
        print()
        ans = input("  Preview looks correct? [y/N/q]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return

    # ── Store metadata ───────────────────────────────────────────
    calibration["_meta"] = {
        "version": 1,
        "ocr_engine": "google_vision",
        "time_format": t_fmt,
        "timezone": tz_str,
        "sample_interval": interval,
    }

    # ── Main calibration loop ────────────────────────────────────
    print()
    t0 = _time.monotonic()

    for vi_idx, vi in enumerate(remaining):
        video_path: Path = vi["path"]
        n_calls: int = vi["n_calls"]
        total: int = vi["total_frames"]
        fps: float = vi["fps"]

        print(f"  [{vi_idx+1}/{len(remaining)}] {video_path.name} ({n_calls} calls)")

        roi = get_roi_for_video(rois, video_path.name)
        if not roi:
            print("    ⚠️  No OCR ROI — skipping")
            continue
        rx, ry, rw, rh = roi

        # Resolve date
        date_str = extract_date_from_filename(video_path.name)
        if not date_str:
            date_str = input(
                f"    Enter date for {video_path.name} (YYYY-MM-DD): "
            ).strip()

        cap = cv2.VideoCapture(str(video_path))

        samples: list[dict] = []
        errors: list[dict] = []
        day_offset = 0
        prev_dt: datetime | None = None
        call_count = 0
        frame_idx = 0

        while frame_idx < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            crop = frame[ry : ry + rh, rx : rx + rw]
            text, conf = google_ocr(client, crop)

            dt, unix = parse_time(text, t_fmt, date_str, tz, day_offset)

            if dt:
                # Check midnight crossing
                new_offset = detect_day_offset(prev_dt, dt, day_offset)
                if new_offset != day_offset:
                    day_offset = new_offset
                    dt, unix = parse_time(text, t_fmt, date_str, tz, day_offset)

                samples.append(
                    {
                        "frame": frame_idx,
                        "ocr_raw": text,
                        "time_local": dt.strftime("%Y-%m-%dT%H:%M:%S"),
                        "unix": round(unix, 2),
                        "confidence": round(conf, 3),
                    }
                )
                prev_dt = dt
            else:
                errors.append(
                    {"frame": frame_idx, "ocr_raw": text, "error": "parse_failed"}
                )

            call_count += 1
            # Progress bar
            pct = call_count / n_calls * 100
            bar_len = 30
            filled = int(bar_len * call_count / n_calls)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r    {bar} {pct:5.1f}%  [{call_count}/{n_calls}]", end="")

            frame_idx += interval

        cap.release()

        # Finish line
        print(
            f"\r    {'█' * 30} 100.0%  [{n_calls}/{n_calls}]  "
            f"✓ {len(samples)} parsed, {len(errors)} errors"
        )

        # Store & save after every video (crash-safe)
        calibration[video_path.name] = {
            "fps_nominal": fps,
            "total_frames": total,
            "date": date_str,
            "samples": samples,
            "errors": errors,
        }
        save_calibration(cal_path, calibration)

    elapsed = _time.monotonic() - t0
    print()
    print(f"✅ Calibration complete → {cal_path}")
    print(f"   Elapsed: {elapsed/60:.1f} min")


# ══════════════════════════════════════════════════════════════════════════
#  Subcommand: postprocess  —  fix errors via hard rules + interpolation
# ══════════════════════════════════════════════════════════════════════════


def _try_fix_missing_seconds(
    text: str, time_format: str, base_date: str, tz: ZoneInfo, day_offset: int
) -> tuple[datetime | None, float, str]:
    """Try to recover timestamps missing seconds (e.g. '12:04 AM' → '12:04:00 AM').

    Returns ``(datetime, unix, fix_reason)`` or ``(None, 0.0, "")`` on failure.
    """
    cleaned = clean_ocr_text(text)
    # Pattern: HH:MM AM/PM (missing :SS)
    m = re.match(
        r"^(\d{1,2}:\d{2})\s+(AM|PM)\b(.*)$", cleaned, re.IGNORECASE
    )
    if m:
        fixed = f"{m.group(1)}:00 {m.group(2)}{m.group(3)}"
        dt, unix = parse_time(fixed, time_format, base_date, tz, day_offset)
        if dt:
            return dt, unix, "missing_seconds"
    return None, 0.0, ""


def _interpolate_gaps(
    points: list[dict],
) -> list[dict]:
    """Fill gaps using linear interpolation with monotonicity enforcement.

    ``points`` is a list of ``{"frame": int, "unix": float | None}``.
    Returns the same list with ``None`` values filled via linear interpolation
    between the nearest known neighbours.

    Also enforces monotonicity: if a known point violates monotonicity
    (timestamp goes backward when frame goes forward), it is treated as
    an outlier and replaced with an interpolated value.
    """
    # Separate known vs unknown
    known = [(p["frame"], p["unix"]) for p in points if p["unix"] is not None]
    if len(known) < 2:
        return points  # can't interpolate

    # Enforce monotonicity on the known points
    clean: list[tuple[int, float]] = [known[0]]
    for frame, unix in known[1:]:
        if unix > clean[-1][1]:  # strictly increasing
            clean.append((frame, unix))
        # else: skip — violates monotonicity

    if len(clean) < 2:
        return points

    frames_known = np.array([c[0] for c in clean], dtype=float)
    unix_known = np.array([c[1] for c in clean], dtype=float)

    for p in points:
        if p["unix"] is None:
            p["unix"] = float(np.interp(p["frame"], frames_known, unix_known))
            p["interpolated"] = True
        else:
            # Check monotonicity against known set
            interp_val = float(np.interp(p["frame"], frames_known, unix_known))
            # If this point's value is wildly off, replace it
            if abs(p["unix"] - interp_val) > 120:  # > 2 min deviation
                p["unix"] = interp_val
                p["interpolated"] = True

    return points


def cmd_postprocess(args):
    """Post-process calibration data: fix OCR errors and interpolate gaps.

    Strategy:
      1. Hard rule: fix "missing seconds" errors (HH:MM AM → HH:MM:00 AM)
      2. Merge fixed errors into samples
      3. Linear interpolation for remaining gaps using monotonicity constraint
      4. Report statistics
    """
    cfg = load_config(args.config)

    cal_path = (
        Path(cfg.drive.time_calibration) if cfg.drive.time_calibration else None
    )
    if not cal_path or not cal_path.exists():
        print("❌ time_calibration.json not found. Run `calibrate` first.")
        sys.exit(1)

    tz_str = cfg.time.timezone
    tz = ZoneInfo(tz_str)

    with open(cal_path) as f:
        calibration = json.load(f)

    meta = calibration.get("_meta", {})
    t_fmt = meta.get("time_format", cfg.time.time_format)
    interval = meta.get("sample_interval", 3000)

    videos = {k: v for k, v in calibration.items() if k != "_meta"}

    print()
    print("══════════════════════════════════════════════════════════")
    print("  Post-processing calibration data")
    print("══════════════════════════════════════════════════════════")
    print()

    total_fixed_sec = 0
    total_interpolated = 0
    total_errors_before = 0
    total_errors_after = 0
    total_samples_before = 0

    for vname in sorted(videos.keys()):
        vdata = videos[vname]
        samples = vdata.get("samples", [])
        errors = vdata.get("errors", [])
        date_str = vdata.get("date", None)

        if not errors:
            continue

        total_samples_before += len(samples)
        total_errors_before += len(errors)

        # ── Step 1: try to fix errors with hard rules ────────────
        still_errors = []
        fixed_count = 0

        for err in errors:
            text = err.get("ocr_raw", "")
            frame = err.get("frame", 0)

            # Estimate day_offset from nearby known samples
            day_offset = 0
            if samples:
                nearby = sorted(samples, key=lambda s: abs(s["frame"] - frame))
                if nearby:
                    ref_time = nearby[0].get("time_local", "")
                    if date_str and ref_time:
                        try:
                            ref_dt = datetime.strptime(ref_time, "%Y-%m-%dT%H:%M:%S")
                            base_dt = datetime.strptime(date_str, "%Y-%m-%d")
                            day_offset = (ref_dt.date() - base_dt.date()).days
                        except ValueError:
                            pass

            # Try missing-seconds fix
            dt, unix, reason = _try_fix_missing_seconds(
                text, t_fmt, date_str or "2025-01-01", tz, day_offset
            )
            if dt:
                samples.append({
                    "frame": frame,
                    "ocr_raw": text,
                    "time_local": dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    "unix": round(unix, 2),
                    "fix": reason,
                })
                fixed_count += 1
                total_fixed_sec += 1
            else:
                still_errors.append(err)

        # ── Step 2: sort all samples by frame ────────────────────
        samples.sort(key=lambda s: s["frame"])

        # ── Step 3: interpolate remaining gaps ───────────────────
        # Build a unified timeline: known samples + error frames
        all_frames = set(s["frame"] for s in samples)
        for err in still_errors:
            all_frames.add(err["frame"])
        all_frames = sorted(all_frames)

        # Create point list
        sample_map = {s["frame"]: s["unix"] for s in samples}
        points = [
            {"frame": f, "unix": sample_map.get(f, None)}
            for f in all_frames
        ]

        points = _interpolate_gaps(points)

        # Merge interpolated points back as samples
        interp_count = 0
        for p in points:
            if p.get("interpolated") and p["frame"] not in sample_map:
                dt_aware = datetime.fromtimestamp(p["unix"], tz=tz)
                samples.append({
                    "frame": p["frame"],
                    "ocr_raw": "",
                    "time_local": dt_aware.strftime("%Y-%m-%dT%H:%M:%S"),
                    "unix": round(p["unix"], 2),
                    "fix": "interpolated",
                })
                interp_count += 1
                total_interpolated += 1

        samples.sort(key=lambda s: s["frame"])

        # Update video data
        vdata["samples"] = samples
        vdata["errors"] = []  # all resolved
        remaining_errs = len(still_errors) - interp_count
        total_errors_after += max(0, remaining_errs)

        if fixed_count > 0 or interp_count > 0:
            print(
                f"  {vname}\n"
                f"    fixed (missing sec): {fixed_count:4d}  |  "
                f"interpolated: {interp_count:4d}  |  "
                f"was {len(errors)} errors → {max(0, remaining_errs)} remain"
            )

    # ── Monotonicity check across all videos ─────────────────────
    print()
    mono_violations = 0
    for vname in sorted(videos.keys()):
        samples = videos[vname].get("samples", [])
        for i in range(1, len(samples)):
            if samples[i]["unix"] < samples[i - 1]["unix"]:
                mono_violations += 1
    if mono_violations > 0:
        print(f"  ⚠️  {mono_violations} monotonicity violations remain")
    else:
        print("  ✅ All samples monotonically increasing (frame↑ → time↑)")

    # ── Save ─────────────────────────────────────────────────────
    calibration["_meta"]["postprocessed"] = True
    save_calibration(cal_path, calibration)

    print()
    print(f"  ── Summary ──────────────────────────────────────")
    print(f"  Errors before:      {total_errors_before}")
    print(f"  Fixed (missing s):  {total_fixed_sec}")
    print(f"  Interpolated:       {total_interpolated}")
    print(f"  Errors after:       {total_errors_after}")
    print(f"  Recovery rate:      {(total_fixed_sec + total_interpolated) / max(total_errors_before,1) * 100:.1f}%")
    print()
    print(f"  ✅ Saved → {cal_path}")


# ══════════════════════════════════════════════════════════════════════════
#  Subcommand: verify  —  spot-check calibration accuracy
# ══════════════════════════════════════════════════════════════════════════


def cmd_verify(args):
    """OCR random frames and compare with calibration predictions."""
    import random

    from avistrack.core.time_lookup import TimeLookup

    cfg = load_config(args.config)
    n_checks: int = args.n

    cal_path = Path(cfg.drive.time_calibration)
    roi_path = Path(cfg.drive.ocr_roi)
    raw_dir = Path(cfg.drive.raw_videos)

    if not cal_path.exists():
        print(f"❌ Calibration not found: {cal_path}")
        print("   Run `calibrate` first.")
        sys.exit(1)

    calibration = load_calibration(cal_path)
    rois = load_ocr_roi(roi_path)
    if not rois:
        print("❌ ocr_roi.json not found or empty")
        sys.exit(1)

    tz_str = cfg.time.timezone
    t_fmt = cfg.time.time_format
    tz = ZoneInfo(tz_str)

    cal_videos = [k for k in calibration if k != "_meta"]
    if not cal_videos:
        print("❌ No calibrated videos found")
        sys.exit(1)

    # ── Estimate cost ────────────────────────────────────────────
    total_verify_calls = len(cal_videos) * n_checks
    cost = total_verify_calls / 1000 * VISION_PRICE_PER_1000
    print(f"🔍 Verification plan: {n_checks} random frames × {len(cal_videos)} videos")
    print(f"   Total API calls: {total_verify_calls}  (est. ${cost:.2f})")
    print()
    ans = input("Continue? [y/N]: ").strip().lower()
    if ans != "y":
        print("Aborted.")
        return

    client = init_google_client()

    all_deltas: list[float] = []

    for vname in cal_videos:
        vdata = calibration[vname]
        if len(vdata.get("samples", [])) < 2:
            print(f"  {vname}: < 2 samples, skipping")
            continue

        # Find video file
        video_path = None
        for p in raw_dir.rglob("*"):
            if p.name == vname:
                video_path = p
                break
        if not video_path:
            print(f"  {vname}: file not found, skipping")
            continue

        roi = get_roi_for_video(rois, vname)
        if not roi:
            print(f"  {vname}: no ROI, skipping")
            continue
        rx, ry, rw, rh = roi

        lookup = TimeLookup.from_calibration(calibration, vname, tz_str)

        # Pick random frames NOT in the calibration set
        sample_frames = {s["frame"] for s in vdata["samples"]}
        total = vdata["total_frames"]
        candidates = [f for f in range(0, total, 100) if f not in sample_frames]
        check_frames = sorted(random.sample(candidates, min(n_checks, len(candidates))))

        print(f"  {vname}:")

        cap = cv2.VideoCapture(str(video_path))
        date_str = vdata.get("date", "")

        for fi in check_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue

            crop = frame[ry : ry + rh, rx : rx + rw]
            text, conf = google_ocr(client, crop)

            predicted_dt = lookup.frame_to_datetime(fi)
            predicted_str = predicted_dt.strftime(t_fmt)

            # Infer day_offset from calibration data
            day_offset = 0
            for s in vdata["samples"]:
                if s["frame"] <= fi:
                    base_dt = datetime.fromisoformat(s["time_local"])
                    base_date = datetime.strptime(date_str, "%Y-%m-%d")
                    day_offset = (base_dt.date() - base_date.date()).days

            actual_dt, _ = parse_time(text, t_fmt, date_str, tz, day_offset)

            if actual_dt:
                delta = abs((actual_dt - predicted_dt).total_seconds())
                all_deltas.append(delta)
                tag = "✓" if delta < 2.0 else "⚠️"
                print(
                    f"    Frame {fi:>8,}:  OCR=\"{text}\"  pred=\"{predicted_str}\"  "
                    f"Δ={delta:.1f}s  {tag}"
                )
            else:
                print(f"    Frame {fi:>8,}:  OCR=\"{text}\"  (parse failed)")

        cap.release()

    # ── Summary ──────────────────────────────────────────────────
    if all_deltas:
        arr = np.array(all_deltas)
        n_ok = int(np.sum(arr < 2.0))
        pct = n_ok / len(arr) * 100
        print()
        print(f"  Summary: {len(arr)} checks")
        print(f"    Mean Δ:    {arr.mean():.2f}s")
        print(f"    Median Δ:  {np.median(arr):.2f}s")
        print(f"    Max Δ:     {arr.max():.2f}s")
        print(f"    Within 2s: {n_ok}/{len(arr)} ({pct:.0f}%)")
        if pct >= 95:
            print("\n✅ Calibration accuracy: PASS")
        else:
            print(
                "\n⚠️  Calibration accuracy: MARGINAL "
                "— consider re-running with a smaller --interval"
            )
    else:
        print("\n⚠️  No successful comparisons were made.")


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Time calibration tool — build frame↔time mapping via OCR.\n\n"
            "Subcommands:\n"
            "  roi          Pick the OCR text region interactively\n"
            "  calibrate    Run sparse OCR → time_calibration.json\n"
            "  postprocess  Fix OCR errors + interpolate gaps\n"
            "  verify       Spot-check calibration accuracy"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command")

    # ── roi ───────────────────────────────────────────────────────
    p_roi = sub.add_parser("roi", help="Pick OCR region for each video (per-video)")
    p_roi.add_argument("--config", required=True, help="AvisTrack YAML config")

    # ── calibrate ─────────────────────────────────────────────────
    p_cal = sub.add_parser("calibrate", help="Run sparse OCR calibration")
    p_cal.add_argument("--config", required=True, help="AvisTrack YAML config")
    p_cal.add_argument(
        "--interval",
        type=int,
        default=1000,
        help="Sample every N frames (default: 1000)",
    )
    p_cal.add_argument(
        "--force",
        action="store_true",
        help="Re-calibrate videos that already have data",
    )

    # ── verify ────────────────────────────────────────────────────
    p_ver = sub.add_parser("verify", help="Spot-check calibration accuracy")
    p_ver.add_argument("--config", required=True, help="AvisTrack YAML config")
    p_ver.add_argument(
        "--n",
        type=int,
        default=5,
        help="Random frames to check per video (default: 5)",
    )

    # ── postprocess ───────────────────────────────────────────────
    p_pp = sub.add_parser(
        "postprocess",
        help="Fix OCR errors (missing seconds, garbled) via rules + interpolation",
    )
    p_pp.add_argument("--config", required=True, help="AvisTrack YAML config")

    args = ap.parse_args()

    if args.command == "roi":
        cmd_roi(args)
    elif args.command == "calibrate":
        cmd_calibrate(args)
    elif args.command == "postprocess":
        cmd_postprocess(args)
    elif args.command == "verify":
        cmd_verify(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
