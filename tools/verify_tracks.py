#!/usr/bin/env python3
import os; os.environ.setdefault("PYTHONUTF8", "1")
"""
tools/verify_tracks.py
────────────────────────────────────────────────────────────────────────
Visual spot-checker for production tracking parquets.

On first run: extracts short clip segments from each video into a local
cache (tools/.verify_cache/), showing a progress dialog.  Subsequent runs
skip extraction and open the viewer instantly.

Usage
-----
    python tools/verify_tracks.py \
        --parquet-dir "E:/Wave2/05_Tracking_Output/top9_interp" \
        --videos-dir  "E:/Wave2/00_raw_videos" \
        --roi-file    "E:/Wave2/02_Global_Metadata/camera_rois.json" \
        [--n-segments 5] [--segment-len 150]

Keyboard
--------
    ←/→           prev / next frame
    Ctrl+←/→      jump 10 frames
    ↑/↓           prev / next segment
    [ / ]         prev / next video
    Space         play / pause
    R             resample this video  (new random segments → re-extract)
    Shift+R       resample ALL videos
    B/I/C/T       toggle boxes / IDs / confidence / trails
    Q             quit
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QShortcut,
    QSlider, QVBoxLayout, QWidget,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from avistrack.core.transformer import PerspectiveTransformer

# ── Constants ─────────────────────────────────────────────────────────────────

VIDEO_EXTS     = {".mkv", ".mp4", ".avi", ".mov"}
TRAIL_LEN      = 20
DISPLAY_SIZE   = 640
FPS_PLAY       = 30
CACHE_DIR      = Path(__file__).parent / ".verify_cache"


_TRACK_COLORS_BGR = [
    (255,  80,  80), ( 80, 255,  80), ( 80,  80, 255),
    (255, 200,  50), (200,  80, 255), ( 50, 220, 255),
    (255, 255,  80), ( 80, 255, 220), (180, 180, 255),
]

def _track_color(tid: int) -> tuple:
    return _TRACK_COLORS_BGR[(int(tid) - 1) % len(_TRACK_COLORS_BGR)]


# ── ffmpeg auto-detection ─────────────────────────────────────────────────────

def _find_ffmpeg() -> Optional[str]:
    """Find ffmpeg: PATH → same conda env → conda pkgs (newest first)."""
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")
    # Walk up from python executable to locate conda root
    p = Path(sys.executable)
    for _ in range(7):
        p = p.parent
        if (p / "pkgs").is_dir():
            # prefer newest pkg
            pkgs = sorted(p.glob("pkgs/ffmpeg-*/Library/bin/ffmpeg.exe"), reverse=True)
            if pkgs:
                return str(pkgs[0])
            # fallback: any env that has it
            envs = sorted(p.glob("envs/*/Library/bin/ffmpeg.exe"))
            if envs:
                return str(envs[0])
            break
    return None


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class VideoEntry:
    name:         str
    parquet_path: Path
    video_path:   Optional[Path]
    transformer:  Optional[PerspectiveTransformer]


@dataclass
class Segment:
    start: int   # parquet frame number (1-indexed, inclusive)
    end:   int


# ── CacheManager ─────────────────────────────────────────────────────────────

class CacheManager:
    """Manages tools/.verify_cache/ — manifest + mp4 clips."""

    def __init__(self, cache_dir: Path, params: dict):
        self._dir    = cache_dir
        self._params = params        # {n_segments, segment_len, seed}
        self._dir.mkdir(parents=True, exist_ok=True)
        self._manifest: dict = self._load_manifest()

    # ── internal ─────────────────────────────────────────────────────────

    def _manifest_path(self) -> Path:
        return self._dir / "manifest.json"

    def _load_manifest(self) -> dict:
        mp = self._manifest_path()
        if mp.exists():
            try:
                with open(mp) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"params": self._params, "entries": {}}

    def _save(self):
        self._manifest["params"] = self._params
        with open(self._manifest_path(), "w") as f:
            json.dump(self._manifest, f, indent=2)

    def _entries(self) -> dict:
        return self._manifest.setdefault("entries", {})

    # ── validation ────────────────────────────────────────────────────────

    def _params_ok(self) -> bool:
        return self._manifest.get("params", {}) == self._params

    def is_valid(self, name: str, video_mtime: float, parquet_mtime: float) -> bool:
        if not self._params_ok():
            return False
        e = self._entries().get(name)
        if not e:
            return False
        if abs(e.get("video_mtime", 0) - video_mtime) > 2:
            return False
        if abs(e.get("parquet_mtime", 0) - parquet_mtime) > 2:
            return False
        return all((self._dir / s["file"]).exists() for s in e.get("segments", []))

    # ── getters ───────────────────────────────────────────────────────────

    def get_segments(self, name: str) -> Optional[list[Segment]]:
        e = self._entries().get(name)
        if not e:
            return None
        return [Segment(s["start"], s["end"]) for s in e["segments"]]

    def get_clip_path(self, name: str, seg_idx: int) -> Optional[Path]:
        e = self._entries().get(name)
        if not e or seg_idx >= len(e["segments"]):
            return None
        p = self._dir / e["segments"][seg_idx]["file"]
        return p if p.exists() else None

    def clip_filename(self, name: str, idx: int, start: int, end: int) -> str:
        return f"{name}_seg{idx:02d}_f{start}-{end}.mp4"

    @property
    def seed(self) -> int:
        return self._params.get("seed", 42)

    # ── setters ───────────────────────────────────────────────────────────

    def store_entry(self, name: str, video_mtime: float, parquet_mtime: float, segs: list[Segment]):
        self._entries()[name] = {
            "video_mtime":   video_mtime,
            "parquet_mtime": parquet_mtime,
            "segments": [
                {"start": s.start, "end": s.end,
                 "file":  self.clip_filename(name, i, s.start, s.end)}
                for i, s in enumerate(segs)
            ],
        }
        self._save()

    def invalidate(self, name: str):
        e = self._entries().pop(name, None)
        if e:
            for s in e.get("segments", []):
                p = self._dir / s["file"]
                if p.exists():
                    p.unlink()
        self._save()

    def invalidate_all(self):
        for mp4 in self._dir.glob("*.mp4"):
            mp4.unlink()
        self._manifest = {"params": self._params, "entries": {}}
        self._save()

    def update_seed(self, seed: int):
        self._params["seed"] = seed

    @property
    def dir(self) -> Path:
        return self._dir


# ── TrackStore ────────────────────────────────────────────────────────────────

class TrackStore:
    def __init__(self, path: Path, frames: Optional[set[int]] = None):
        """
        frames: if given, only keep those frame numbers before groupby.
                Viewer passes a ~750-frame set so we avoid groupby-ing 1M rows.
                ExtractionWorker passes None (needs full range for sampling).
        """
        df = pd.read_parquet(path)
        df["frame"] = df["frame"].astype(int)
        self.frame_min = int(df["frame"].min())
        self.frame_max = int(df["frame"].max())
        if frames is not None:
            df = df[df["frame"].isin(frames)]
        # call to_dict once on the (small) filtered df, then bucket by frame
        self._by_frame: dict[int, list[dict]] = {}
        for r in df.to_dict("records"):
            self._by_frame.setdefault(r["frame"], []).append(r)

    def get(self, frame_no: int) -> list[dict]:
        return self._by_frame.get(frame_no, [])

    def sample_segments(self, n: int, seg_len: int, seed: int) -> list[Segment]:
        rng  = random.Random(seed)
        span = self.frame_max - self.frame_min + 1
        if span <= seg_len:
            return [Segment(self.frame_min, self.frame_max)]
        candidates = list(range(self.frame_min, self.frame_max - seg_len + 2))
        rng.shuffle(candidates)
        segs:  list[Segment]      = []
        taken: list[tuple[int, int]] = []
        for start in candidates:
            end = start + seg_len - 1
            if all(end < ts or start > te for ts, te in taken):
                segs.append(Segment(start, end))
                taken.append((start, end))
                if len(segs) >= n:
                    break
        segs.sort(key=lambda s: s.start)
        return segs


# ── Extraction worker ─────────────────────────────────────────────────────────

class ExtractionWorker(QThread):
    """Background worker: loads parquet, samples segments, extracts clips.

    Takes raw VideoEntry objects so the main thread never touches parquets.
    grand_total is estimated as n_entries × n_segments (exact count unknown
    until parquets are loaded, but close enough for the progress bar).
    """
    # video_name, seg_idx, n_segs_this_video, done_total, grand_total
    progress    = pyqtSignal(str, int, int, int, int)
    finished_ok = pyqtSignal()

    def __init__(
        self,
        entries:    list[VideoEntry],
        cache:      CacheManager,
        n_segments: int,
        seg_len:    int,
        ffmpeg:     Optional[str],
    ):
        super().__init__()
        self._entries   = entries
        self._cache     = cache
        self._n_seg     = n_segments
        self._seg_len   = seg_len
        self._ffmpeg    = ffmpeg
        self._cancelled = False
        self._grand_total = len(entries) * n_segments   # estimate
        self._done        = 0

    def cancel(self):
        self._cancelled = True

    def run(self):
        for entry in self._entries:
            if self._cancelled:
                break
            try:
                store = TrackStore(entry.parquet_path)
                segs  = store.sample_segments(self._n_seg, self._seg_len, self._cache.seed)
                fps, w, h = self._probe(entry.video_path)
            except Exception as exc:
                print(f"[ERROR] load {entry.name}: {exc}", file=sys.stderr)
                continue

            for seg_idx, seg in enumerate(segs):
                if self._cancelled:
                    break
                self.progress.emit(entry.name, seg_idx, len(segs),
                                   self._done, self._grand_total)
                fname    = self._cache.clip_filename(entry.name, seg_idx, seg.start, seg.end)
                out_path = self._cache.dir / fname
                try:
                    if self._ffmpeg:
                        self._via_ffmpeg(entry.video_path, seg, fps, out_path)
                    else:
                        self._via_opencv(entry.video_path, seg, fps, w, h, out_path)
                except Exception as exc:
                    print(f"[ERROR] extract seg{seg_idx} {entry.name}: {exc}", file=sys.stderr)
                self._done += 1

            if not self._cancelled:
                vm = entry.video_path.stat().st_mtime if entry.video_path else 0.0
                pm = entry.parquet_path.stat().st_mtime
                self._cache.store_entry(entry.name, vm, pm, segs)

        if not self._cancelled:
            self.finished_ok.emit()

    def _probe(self, vp: Optional[Path]) -> tuple[float, int, int]:
        if not vp or not vp.exists():
            return 30.0, 640, 640
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return fps, w, h

    def _via_ffmpeg(self, vp: Path, seg: Segment, fps: float, out: Path):
        start_s = (seg.start - 1) / fps
        dur_s   = (seg.end - seg.start + 1) / fps
        subprocess.run(
            [self._ffmpeg,
             "-ss", f"{start_s:.6f}", "-i", str(vp),
             "-t",  f"{dur_s:.6f}",
             "-c:v", "libx264", "-crf", "23", "-an", "-y", str(out)],
            capture_output=True,
        )

    def _via_opencv(self, vp: Path, seg: Segment, fps: float, w: int, h: int, out: Path):
        cap    = cv2.VideoCapture(str(vp))
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, seg.start - 1)
        for _ in range(seg.end - seg.start + 1):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        writer.release()
        cap.release()


# ── Progress dialog ───────────────────────────────────────────────────────────

class ProgressDialog(QDialog):
    def __init__(self, worker: ExtractionWorker, total: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pre-extracting Segments")
        self.setMinimumWidth(500)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self._worker    = worker
        self._cancelled = False

        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        self._lbl_video = QLabel("Initializing…")
        self._lbl_video.setFont(QFont("Consolas", 9))
        lay.addWidget(self._lbl_video)

        self._lbl_seg = QLabel("")
        lay.addWidget(self._lbl_seg)

        self._bar = QProgressBar()
        self._bar.setRange(0, total)
        self._bar.setValue(0)
        lay.addWidget(self._bar)

        self._lbl_count = QLabel(f"0 / {total} segments")
        lay.addWidget(self._lbl_count)

        note = QLabel("(You can Cancel to skip; uncached videos will fall back to slower live seek.)")
        note.setStyleSheet("color: gray; font-size: 8pt;")
        lay.addWidget(note)

        btn = QPushButton("Cancel")
        btn.clicked.connect(self._cancel)
        lay.addWidget(btn, alignment=Qt.AlignRight)

        worker.progress.connect(self._on_progress)
        worker.finished_ok.connect(self.accept)
        worker.start()

    def _on_progress(self, name: str, seg_idx: int, n_segs: int, done: int, total: int):
        self._lbl_video.setText(name)
        self._lbl_seg.setText(f"Seg {seg_idx + 1} / {n_segs}")
        self._bar.setValue(done)
        self._lbl_count.setText(f"{done} / {total} segments")

    def _cancel(self):
        self._cancelled = True
        self._worker.cancel()
        self.reject()

    def was_cancelled(self) -> bool:
        return self._cancelled


# ── Rendering ─────────────────────────────────────────────────────────────────

def _render(
    raw_bgr:     np.ndarray,
    transformer: Optional[PerspectiveTransformer],
    rows:        list[dict],
    trail:       dict[int, deque],
    show_boxes:  bool = True,
    show_ids:    bool = True,
    show_conf:   bool = True,
    show_trails: bool = True,
    target_n:    int  = 9,
) -> np.ndarray:
    frame = transformer.transform(raw_bgr) if transformer else raw_bgr.copy()
    if frame.shape[0] != DISPLAY_SIZE or frame.shape[1] != DISPLAY_SIZE:
        frame = cv2.resize(frame, (DISPLAY_SIZE, DISPLAY_SIZE))

    if show_trails:
        for tid, pts in trail.items():
            pts_list = list(pts)
            color    = _track_color(tid)
            for i in range(1, len(pts_list)):
                a = i / len(pts_list)
                c = tuple(int(v * a) for v in color)
                cv2.line(frame, pts_list[i - 1], pts_list[i], c, 1, cv2.LINE_AA)

    n_det = len(rows)
    for row in rows:
        tid   = int(row["id"])
        x, y  = int(row["x"]), int(row["y"])
        w, h  = int(row["w"]), int(row["h"])
        conf  = float(row["conf"])
        color = _track_color(tid)
        if show_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if show_ids or show_conf:
            parts = []
            if show_ids:  parts.append(f"#{tid}")
            if show_conf: parts.append(f"{conf:.2f}")
            lbl = " ".join(parts)
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ly = max(y - 4, th + 6)
            cv2.rectangle(frame, (x, ly - th - 4), (x + tw + 4, ly), color, -1)
            cv2.putText(frame, lbl, (x + 2, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    badge = (70, 180, 70) if n_det == target_n else \
            (60, 145, 240) if n_det == target_n - 1 else (60, 60, 210)
    cv2.rectangle(frame, (4, 4), (78, 30), badge, -1)
    cv2.putText(frame, f"N={n_det}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def _update_trail(trail: dict, rows: list[dict]):
    for row in rows:
        tid = int(row["id"])
        cx  = int(row["x"]) + int(row["w"]) // 2
        cy  = int(row["y"]) + int(row["h"]) // 2
        if tid not in trail:
            trail[tid] = deque(maxlen=TRAIL_LEN)
        trail[tid].append((cx, cy))


def _bgr_to_pixmap(bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888))


def _unix_to_str(v) -> str:
    try:
        ts = float(v)
        if np.isnan(ts):
            return "—"
        return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "—"


# ── Main window ───────────────────────────────────────────────────────────────

class VerifyWindow(QMainWindow):

    def __init__(
        self,
        entries:     list[VideoEntry],
        cache:       CacheManager,
        n_segments:  int,
        segment_len: int,
        ffmpeg_exe:  Optional[str],
    ):
        super().__init__()
        self.setWindowTitle("AvisTrack — Track Verifier")
        self.entries     = entries
        self._cache      = cache
        self.n_segments  = n_segments
        self.segment_len = segment_len
        self._ffmpeg     = ffmpeg_exe

        self._entry:      Optional[VideoEntry] = None
        self._store:      Optional[TrackStore] = None
        self._vid_idx:    int = 0
        self._segments:   list[Segment] = []
        self._seg_idx:    int = 0
        self._frame_no:   int = 0
        self._playing:    bool = False
        self._trail:      dict[int, deque] = {}

        # VideoCapture state — keyed by clip path so segment switches reopen automatically
        self._cap:            Optional[cv2.VideoCapture] = None
        self._cap_clip_path:  Optional[Path] = None
        self._cap_cursor:     int = -1

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        self._build_ui()
        if entries:
            self._load_entry(0)
        self.show()

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        cw   = QWidget()
        root = QVBoxLayout(cw)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        self.setCentralWidget(cw)

        # Row 1 — video selector + segment nav
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Video:"))
        self._combo = QComboBox()
        self._combo.setMinimumWidth(280)
        for e in self.entries:
            self._combo.addItem(e.name)
        self._combo.currentIndexChanged.connect(lambda i: self._load_entry(i))
        row1.addWidget(self._combo, 3)
        row1.addSpacing(16)

        self._seg_lbl = QLabel("Seg —/—")
        self._seg_lbl.setMinimumWidth(80)
        row1.addWidget(self._seg_lbl)

        btn_ps = QPushButton("◀ Prev");  btn_ps.clicked.connect(self._prev_segment)
        btn_ns = QPushButton("Next ▶");  btn_ns.clicked.connect(self._next_segment)
        btn_rs = QPushButton("↺ Resample")
        btn_rs.setToolTip("R = resample this video\nShift+R = resample ALL videos")
        btn_rs.clicked.connect(lambda: self._do_resample(all_videos=False))
        for b in (btn_ps, btn_ns, btn_rs):
            row1.addWidget(b)
        root.addLayout(row1)

        # Display
        self._disp = QLabel()
        self._disp.setFixedSize(DISPLAY_SIZE, DISPLAY_SIZE)
        self._disp.setStyleSheet("background: #111;")
        self._disp.setAlignment(Qt.AlignCenter)
        root.addWidget(self._disp, alignment=Qt.AlignHCenter)

        # Info bar
        info = QHBoxLayout()
        self._lbl_frame = QLabel("Frame: —")
        self._lbl_time  = QLabel("Time: —")
        mono = QFont("Consolas", 9)
        self._lbl_frame.setFont(mono)
        self._lbl_time.setFont(mono)
        info.addWidget(self._lbl_frame)
        info.addStretch()
        info.addWidget(self._lbl_time)
        root.addLayout(info)

        # Slider
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setTracking(False)
        self._slider.valueChanged.connect(self._on_slider)
        root.addWidget(self._slider)

        # Playback
        ctrl = QHBoxLayout()
        btn_back = QPushButton("◀");     btn_back.setFixedWidth(40); btn_back.clicked.connect(self._step_back)
        self._btn_play = QPushButton("▶ Play"); self._btn_play.setFixedWidth(90); self._btn_play.clicked.connect(self._toggle_play)
        btn_fwd  = QPushButton("▶");     btn_fwd.setFixedWidth(40);  btn_fwd.clicked.connect(self._step_fwd)
        ctrl.addStretch()
        for b in (btn_back, self._btn_play, btn_fwd):
            ctrl.addWidget(b)
        ctrl.addStretch()
        root.addLayout(ctrl)

        # Toggles
        tgl = QHBoxLayout()
        self._cb_boxes  = QCheckBox("Box [B]");   self._cb_boxes.setChecked(True)
        self._cb_ids    = QCheckBox("ID [I]");    self._cb_ids.setChecked(True)
        self._cb_conf   = QCheckBox("Conf [C]");  self._cb_conf.setChecked(True)
        self._cb_trails = QCheckBox("Trail [T]"); self._cb_trails.setChecked(True)
        for cb in (self._cb_boxes, self._cb_ids, self._cb_conf, self._cb_trails):
            cb.stateChanged.connect(lambda _: self._redraw())
            tgl.addWidget(cb)
        root.addLayout(tgl)

        # Shortcuts
        def _sc(key, fn):
            QShortcut(QKeySequence(key), self).activated.connect(fn)

        _sc(Qt.Key_Space,              self._toggle_play)
        _sc(Qt.Key_Left,               self._step_back)
        _sc(Qt.Key_Right,              self._step_fwd)
        _sc(Qt.CTRL + Qt.Key_Left,     lambda: self._jump(-10))
        _sc(Qt.CTRL + Qt.Key_Right,    lambda: self._jump(10))
        _sc(Qt.Key_Up,                 self._prev_segment)
        _sc(Qt.Key_Down,               self._next_segment)
        _sc(Qt.Key_BracketLeft,        self._prev_video)
        _sc(Qt.Key_BracketRight,       self._next_video)
        _sc(Qt.Key_R,                  lambda: self._do_resample(all_videos=False))
        _sc(Qt.SHIFT + Qt.Key_R,       lambda: self._do_resample(all_videos=True))
        _sc(Qt.Key_B,                  self._cb_boxes.toggle)
        _sc(Qt.Key_I,                  self._cb_ids.toggle)
        _sc(Qt.Key_C,                  self._cb_conf.toggle)
        _sc(Qt.Key_T,                  self._cb_trails.toggle)
        _sc(Qt.Key_Q,                  self.close)

    # ── Video navigation ──────────────────────────────────────────────────

    def _prev_video(self):
        if self._vid_idx > 0:
            self._combo.setCurrentIndex(self._vid_idx - 1)

    def _next_video(self):
        if self._vid_idx < len(self.entries) - 1:
            self._combo.setCurrentIndex(self._vid_idx + 1)

    # ── Entry loading (instant — reads from cache) ────────────────────────

    def _load_entry(self, idx: int):
        self._stop_play()
        self._vid_idx = idx
        self._entry   = self.entries[idx]
        self._trail.clear()
        self._store   = None
        self._segments = []

        segs = self._cache.get_segments(self._entry.name)

        if segs:
            # Cache hit — fast path: only load the ~750 rows we need
            needed: set[int] = set()
            for s in segs:
                needed.update(range(s.start, s.end + 1))
            self._store    = TrackStore(self._entry.parquet_path, frames=needed)
            self._segments = segs
            self._seg_idx  = 0
            self._go_to_segment(0)
        else:
            # Cache miss — extract this video in background, show placeholder
            self._show_canvas_msg(f"Extracting {self._entry.name}…")
            self._start_ondemand_extract(self._entry)

    def _show_canvas_msg(self, msg: str):
        canvas = np.zeros((DISPLAY_SIZE, DISPLAY_SIZE, 3), dtype=np.uint8)
        cv2.putText(canvas, msg, (30, DISPLAY_SIZE // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 140, 140), 2, cv2.LINE_AA)
        self._disp.setPixmap(_bgr_to_pixmap(canvas))
        self._lbl_frame.setText("Frame: —")
        self._lbl_time.setText("Time: —")

    def _start_ondemand_extract(self, entry: VideoEntry):
        """Extract a single video's segments in background; reload when done."""
        if hasattr(self, "_ondemand_worker") and self._ondemand_worker is not None:
            self._ondemand_worker.cancel()
        worker = ExtractionWorker([entry], self._cache, self.n_segments,
                                  self.segment_len, self._ffmpeg)
        worker.finished_ok.connect(lambda: self._on_ondemand_done(entry.name))
        worker.start()
        self._ondemand_worker = worker

    def _on_ondemand_done(self, name: str):
        self._ondemand_worker = None
        # Reload only if the user is still on the same video
        if self._entry and self._entry.name == name:
            self._load_entry(self._vid_idx)

    # ── Resample ─────────────────────────────────────────────────────────

    def _do_resample(self, all_videos: bool):
        self._stop_play()
        new_seed = random.randint(1, 99999)
        self._cache.update_seed(new_seed)

        if all_videos:
            self._cache.invalidate_all()
            targets = [e for e in self.entries if e.video_path and e.video_path.exists()]
        else:
            self._cache.invalidate(self._entry.name)
            targets = [self._entry] if self._entry.video_path and self._entry.video_path.exists() else []

        if targets:
            total  = len(targets) * self.n_segments   # estimate
            worker = ExtractionWorker(targets, self._cache, self.n_segments,
                                      self.segment_len, self._ffmpeg)
            dlg    = ProgressDialog(worker, total, parent=self)
            dlg.exec_()

        # reload current entry from fresh cache
        self._load_entry(self._vid_idx)

    # ── Segment navigation ────────────────────────────────────────────────

    def _go_to_segment(self, idx: int):
        if not self._segments:
            return
        self._stop_play()
        self._seg_idx = max(0, min(idx, len(self._segments) - 1))
        self._trail.clear()
        seg = self._segments[self._seg_idx]
        self._slider.blockSignals(True)
        self._slider.setMinimum(seg.start)
        self._slider.setMaximum(seg.end)
        self._slider.blockSignals(False)
        self._seg_lbl.setText(f"Seg {self._seg_idx + 1}/{len(self._segments)}")
        self._seek(seg.start)

    def _prev_segment(self):
        self._go_to_segment(self._seg_idx - 1)

    def _next_segment(self):
        self._go_to_segment(self._seg_idx + 1)

    # ── Frame navigation ──────────────────────────────────────────────────

    def _seek(self, frame_no: int):
        self._trail.clear()
        self._frame_no = frame_no
        self._do_draw(sequential=False, update_trail=True)
        self._sync_slider()

    def _step_fwd(self):
        seg = self._segments[self._seg_idx]
        if self._frame_no < seg.end:
            self._frame_no += 1
            self._do_draw(sequential=True, update_trail=True)
            self._sync_slider()

    def _step_back(self):
        seg = self._segments[self._seg_idx]
        if self._frame_no > seg.start:
            self._trail.clear()
            self._frame_no -= 1
            self._do_draw(sequential=False, update_trail=True)
            self._sync_slider()

    def _jump(self, delta: int):
        seg    = self._segments[self._seg_idx]
        target = max(seg.start, min(seg.end, self._frame_no + delta))
        if target != self._frame_no:
            self._trail.clear()
            self._frame_no = target
            self._do_draw(sequential=False, update_trail=True)
            self._sync_slider()

    def _on_slider(self, val: int):
        if val != self._frame_no:
            self._trail.clear()
            self._frame_no = val
            self._do_draw(sequential=False, update_trail=True)

    # ── Playback ──────────────────────────────────────────────────────────

    def _toggle_play(self):
        if self._playing:
            self._stop_play()
        else:
            self._playing = True
            self._btn_play.setText("⏸ Pause")
            self._timer.start(max(1, int(1000 / FPS_PLAY)))

    def _stop_play(self):
        self._playing = False
        self._btn_play.setText("▶ Play")
        self._timer.stop()

    def _tick(self):
        seg = self._segments[self._seg_idx]
        if self._frame_no >= seg.end:
            self._stop_play()
            return
        self._frame_no += 1
        self._do_draw(sequential=True, update_trail=True)
        self._sync_slider()

    # ── Rendering ─────────────────────────────────────────────────────────

    def _get_raw_frame(self, frame_no: int, sequential: bool) -> Optional[np.ndarray]:
        if not self._entry or not self._segments:
            return None

        seg       = self._segments[self._seg_idx]
        clip_path = self._cache.get_clip_path(self._entry.name, self._seg_idx)

        if clip_path:
            # ── read from cache clip ──────────────────────────────────
            cv2_idx = frame_no - seg.start          # 0-indexed within clip
            source  = clip_path
        else:
            # ── fallback: original video ──────────────────────────────
            vp = self._entry.video_path
            if not vp or not vp.exists():
                return None
            cv2_idx = frame_no - 1                  # parquet is 1-indexed
            source  = vp

        # Open / reopen cap if source changed
        if self._cap is None or self._cap_clip_path != source:
            if self._cap:
                self._cap.release()
            self._cap          = cv2.VideoCapture(str(source))
            self._cap_clip_path = source
            self._cap_cursor   = -1

        if not self._cap.isOpened():
            return None

        if not sequential or self._cap_cursor != cv2_idx:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, cv2_idx)
            self._cap_cursor = cv2_idx

        ret, frame = self._cap.read()
        self._cap_cursor = (cv2_idx + 1) if ret else -1
        return frame if ret else None

    def _do_draw(self, sequential: bool, update_trail: bool):
        if not self._segments or self._store is None:
            return
        raw  = self._get_raw_frame(self._frame_no, sequential)
        rows = self._store.get(self._frame_no)

        if update_trail:
            _update_trail(self._trail, rows)

        if raw is None:
            canvas = np.zeros((DISPLAY_SIZE, DISPLAY_SIZE, 3), dtype=np.uint8)
            cv2.putText(canvas, "Video not found", (140, DISPLAY_SIZE // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
        else:
            canvas = _render(
                raw, self._entry.transformer, rows, self._trail,
                show_boxes  = self._cb_boxes.isChecked(),
                show_ids    = self._cb_ids.isChecked(),
                show_conf   = self._cb_conf.isChecked(),
                show_trails = self._cb_trails.isChecked(),
            )

        self._disp.setPixmap(_bgr_to_pixmap(canvas))
        self._update_info(rows)

    def _redraw(self):
        self._do_draw(sequential=False, update_trail=False)

    def _update_info(self, rows: list[dict]):
        seg = self._segments[self._seg_idx] if self._segments else None
        if seg:
            pos   = self._frame_no - seg.start + 1
            total = seg.end - seg.start + 1
            self._lbl_frame.setText(
                f"Frame {self._frame_no}  [{pos}/{total}]"
                f"  (seg {self._seg_idx + 1}: {seg.start}–{seg.end})"
            )
        else:
            self._lbl_frame.setText("Frame: —")

        ts = rows[0].get("unix_time") if rows else None
        self._lbl_time.setText(f"Time: {_unix_to_str(ts) if ts is not None else '—'}")

    def _sync_slider(self):
        self._slider.blockSignals(True)
        self._slider.setValue(self._frame_no)
        self._slider.blockSignals(False)

    def closeEvent(self, event):
        self._stop_play()
        if self._cap:
            self._cap.release()
        super().closeEvent(event)


# ── Entry scanning ────────────────────────────────────────────────────────────

def _scan_entries(parquet_dir: Path, videos_dir: Path, roi_file: Optional[Path]) -> list[VideoEntry]:
    parquets = sorted(parquet_dir.glob("*.parquet"))
    if not parquets:
        print(f"ERROR: no parquets in {parquet_dir}", file=sys.stderr)
        sys.exit(1)

    video_lookup: dict[str, Path] = {}
    for ext in VIDEO_EXTS:
        for vp in videos_dir.rglob(f"*{ext}"):
            video_lookup[vp.stem] = vp

    roi_data: dict = {}
    if roi_file and roi_file.exists():
        with open(roi_file) as f:
            roi_data = json.load(f)

    entries: list[VideoEntry] = []
    for pq in parquets:
        stem    = pq.stem
        vid     = video_lookup.get(stem)
        xformer = None
        if vid and roi_data:
            key = vid.name if vid.name in roi_data else stem
            if key in roi_data:
                try:
                    xformer = PerspectiveTransformer(corners=roi_data[key],
                                                     target_size=(DISPLAY_SIZE, DISPLAY_SIZE))
                except Exception as e:
                    print(f"  [WARN] ROI failed {key}: {e}", file=sys.stderr)
        if vid is None:
            print(f"  [WARN] no video for {stem}", file=sys.stderr)
        entries.append(VideoEntry(name=stem, parquet_path=pq, video_path=vid, transformer=xformer))

    print(f"Found {len(entries)} parquets, {sum(1 for e in entries if e.video_path)} with video.")
    return entries


# ── Startup: fast mtime-only stale check (no parquet loading) ─────────────────

def _find_stale(entries: list[VideoEntry], cache: CacheManager) -> list[VideoEntry]:
    """Return entries whose cache is missing or stale — mtime check only, no I/O."""
    stale = []
    for entry in entries:
        if not entry.video_path or not entry.video_path.exists():
            continue
        vm = entry.video_path.stat().st_mtime
        pm = entry.parquet_path.stat().st_mtime
        if not cache.is_valid(entry.name, vm, pm):
            stale.append(entry)
    return stale


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Visual parquet track verifier.")
    ap.add_argument("--parquet-dir",  required=True)
    ap.add_argument("--videos-dir",   required=True)
    ap.add_argument("--roi-file",     default=None)
    ap.add_argument("--n-segments",   type=int, default=5)
    ap.add_argument("--segment-len",  type=int, default=150)
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    parquet_dir = Path(args.parquet_dir)
    videos_dir  = Path(args.videos_dir)
    roi_file    = Path(args.roi_file) if args.roi_file else None

    ffmpeg_exe = _find_ffmpeg()
    if ffmpeg_exe:
        print(f"ffmpeg: {ffmpeg_exe}")
    else:
        print("ffmpeg not found — will use OpenCV fallback (slower)")

    entries = _scan_entries(parquet_dir, videos_dir, roi_file)

    params  = {"n_segments": args.n_segments, "segment_len": args.segment_len, "seed": args.seed}
    cache   = CacheManager(CACHE_DIR, params)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    stale = _find_stale(entries, cache)   # fast: mtime only, no parquet I/O
    if stale:
        total  = len(stale) * args.n_segments   # estimate for progress bar
        worker = ExtractionWorker(stale, cache, args.n_segments,
                                  args.segment_len, ffmpeg_exe)
        dlg    = ProgressDialog(worker, total)
        dlg.exec_()   # shows immediately; parquet loading happens in worker thread

    win = VerifyWindow(entries, cache, args.n_segments, args.segment_len, ffmpeg_exe)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
