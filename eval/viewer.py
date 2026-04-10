#!/usr/bin/env python3
"""
eval/viewer.py
──────────────
Interactive PyQt5 viewer for comparing tracking methods against ground truth.

Layout: GT panel | Method-A panel | Method-B panel
        Metrics table (HOTA / IDF1 / MOTA / IDSW per method)
        Frame slider + playback controls + overlay toggles

Usage:
    python eval/viewer.py \
        --clips-dir  "E:/Wave2/01_Dataset_MOT_Format/test_golden" \
        --tracks-dir "E:/Wave2/04_Evaluation/W2_Collective/tracks" \
        --scores     "E:/Wave2/04_Evaluation/W2_Collective/reports/scores.json"

Keyboard shortcuts:
    ← / →       previous / next frame
    Ctrl+← / →  jump 10 frames
    ↑ / ↓       previous / next clip
    Space       play / pause
    B           toggle boxes
    I           toggle IDs
    C           toggle confidence
    T           toggle trails
    Q           quit
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QFont, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QComboBox,
    QSlider, QPushButton, QCheckBox, QHBoxLayout, QVBoxLayout,
    QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QSizePolicy, QFrame, QShortcut, QSplitter,
)


# ── Color palette for 9 track IDs ────────────────────────────────────────────

_TRACK_COLORS_BGR = [
    (255,  80,  80),  # 1  blue
    ( 80, 255,  80),  # 2  green
    ( 80,  80, 255),  # 3  red
    (255, 200,  50),  # 4  cyan-yellow
    (200,  80, 255),  # 5  purple
    ( 50, 220, 255),  # 6  orange
    (255, 255,  80),  # 7  aqua
    ( 80, 255, 220),  # 8  lime
    (180, 180, 255),  # 9  pink-ish
]

def _track_color(tid: int) -> tuple:
    return _TRACK_COLORS_BGR[(int(tid)-1) % len(_TRACK_COLORS_BGR)]

_GT_COLOR   = (100, 230, 100)   # bright green for GT
_TRAIL_LEN  = 20


# ── Data loading ──────────────────────────────────────────────────────────────

def load_mot(txt_path: Path) -> dict[int, list[tuple]]:
    """MOT txt → {frame(1-based): [(id, x1,y1,x2,y2)]}"""
    by_frame: dict[int, list] = defaultdict(list)
    with open(txt_path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6: continue
            frame = int(p[0]); tid = int(p[1])
            x,y,w,h = float(p[2]),float(p[3]),float(p[4]),float(p[5])
            by_frame[frame].append((tid, x, y, x+w, y+h))
    return dict(by_frame)


def load_scores(path: Path) -> dict:
    if not path or not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── Frame renderer ────────────────────────────────────────────────────────────

class Renderer:
    def __init__(self, panel_size: int = 480):
        self.size = panel_size

    def render(self, img: np.ndarray,
               tracks: dict[int, list[tuple]],   # {frame: [(id,x1,y1,x2,y2)]}
               frame_1based: int,
               show_boxes: bool, show_ids: bool,
               show_conf: bool,                  # not used here (no conf in tracks)
               show_trails: bool,
               gt_style: bool = False) -> np.ndarray:

        out = cv2.resize(img, (self.size, self.size))
        scale = self.size / max(img.shape[:2])

        dets = tracks.get(frame_1based, [])

        # Draw trails
        if show_trails and dets:
            for tid, *_ in dets:
                pts = []
                for f in range(max(1, frame_1based-_TRAIL_LEN), frame_1based):
                    for d in tracks.get(f, []):
                        if d[0] == tid:
                            cx = int((d[1]+d[3])/2 * scale)
                            cy = int((d[2]+d[4])/2 * scale)
                            pts.append((cx, cy))
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    col = _GT_COLOR if gt_style else _track_color(tid)
                    faded = tuple(int(c*alpha) for c in col)
                    cv2.line(out, pts[i-1], pts[i], faded, 1)

        # Draw boxes + IDs
        if show_boxes and dets:
            for tid, x1, y1, x2, y2 in dets:
                col = _GT_COLOR if gt_style else _track_color(tid)
                x1s,y1s,x2s,y2s = (int(v*scale) for v in (x1,y1,x2,y2))
                cv2.rectangle(out, (x1s,y1s), (x2s,y2s), col, 2)
                if show_ids:
                    label = str(int(tid))
                    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    bx1 = max(x1s,0); by1 = max(y1s-th-4,0)
                    cv2.rectangle(out,(bx1,by1),(bx1+tw+4,by1+th+4),col,-1)
                    cv2.putText(out, label, (bx1+2,by1+th+1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1)

        return out

    @staticmethod
    def to_pixmap(img_bgr: np.ndarray) -> QPixmap:
        h, w, ch = img_bgr.shape
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        qi  = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        return QPixmap.fromImage(qi)


# ── Video panel widget ────────────────────────────────────────────────────────

class VideoPanel(QFrame):
    def __init__(self, title: str, panel_size: int = 480):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { border: 1px solid #555; border-radius: 4px; }")

        self._label_img = QLabel()
        self._label_img.setAlignment(Qt.AlignCenter)
        self._label_img.setMinimumSize(panel_size, panel_size)
        self._label_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._title = QLabel(title)
        self._title.setAlignment(Qt.AlignCenter)
        self._title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self._title.setStyleSheet("color:#aaa; padding:4px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.addWidget(self._title)
        layout.addWidget(self._label_img, stretch=1)

    def set_title(self, t: str):
        self._title.setText(t)

    def set_pixmap(self, pm: QPixmap):
        self._label_img.setPixmap(
            pm.scaled(self._label_img.width(), self._label_img.height(),
                      Qt.KeepAspectRatio, Qt.SmoothTransformation))


# ── Metrics table ─────────────────────────────────────────────────────────────

METRIC_KEYS = ["HOTA", "IDF1", "MOTA", "IDSW"]

class MetricsTable(QTableWidget):
    def __init__(self):
        super().__init__(len(METRIC_KEYS), 2)
        self.setVerticalHeaderLabels(METRIC_KEYS)
        self.setHorizontalHeaderLabels(["Method A", "Method B"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setDefaultSectionSize(28)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setFixedHeight(len(METRIC_KEYS)*30 + 26)
        self.setStyleSheet("""
            QTableWidget { background:#2a2a2a; color:#ddd; gridline-color:#444;
                           border:none; font-size:13px; }
            QHeaderView::section { background:#333; color:#aaa;
                                   padding:4px; border:1px solid #444; }
        """)

    def update_scores(self, scores_a: dict, scores_b: dict):
        for row, key in enumerate(METRIC_KEYS):
            for col, scores in enumerate([scores_a, scores_b]):
                val = scores.get(key, "—")
                if isinstance(val, float):
                    text = f"{val:.3f}"
                else:
                    text = str(val) if val != "—" else "—"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.setItem(row, col, item)


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    PANEL_SIZE = 480
    PLAY_FPS   = 10

    def __init__(self, clips_dir: Path, tracks_dir: Path, scores: dict):
        super().__init__()
        self.setWindowTitle("AvisTrack — Tracking Method Comparison")
        self.setMinimumSize(1560, 800)

        self.clips_dir  = clips_dir
        self.tracks_dir = tracks_dir
        self.scores     = scores

        # Discover clips and methods
        ann_dir = clips_dir / "annotations"
        self.clip_names = sorted(d.name for d in ann_dir.iterdir() if d.is_dir())
        self.methods    = sorted(d.name for d in tracks_dir.iterdir() if d.is_dir())

        if not self.clip_names:
            raise RuntimeError(f"No clips in {ann_dir}")
        if not self.methods:
            raise RuntimeError(f"No tracking methods in {tracks_dir}")

        self.renderer = Renderer(self.PANEL_SIZE)
        self._build_ui()
        self._connect_signals()
        self._setup_shortcuts()
        self._apply_dark_theme()

        # State
        self.clip_idx    = 0
        self.frame_idx   = 0   # 0-based
        self.show_boxes  = True
        self.show_ids    = True
        self.show_conf   = False
        self.show_trails = False
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(1000 // self.PLAY_FPS)
        self._play_timer.timeout.connect(self._next_frame)

        self._load_clip(0)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6); root.setContentsMargins(8,8,8,8)

        # ── Top controls bar ──────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.setSpacing(12)

        ctrl.addWidget(QLabel("Clip:"))
        self.combo_clip = QComboBox()
        # Shorten names for display
        for name in self.clip_names:
            short = name.split("_s")[0].replace("Wave2_CollectiveDanger_","") + \
                    ("_s"+name.split("_s")[-1] if "_s" in name else "")
            self.combo_clip.addItem(short, userData=name)
        self.combo_clip.setMinimumWidth(280)
        ctrl.addWidget(self.combo_clip)

        ctrl.addWidget(_vsep())

        ctrl.addWidget(QLabel("Method A:"))
        self.combo_a = QComboBox()
        for m in self.methods: self.combo_a.addItem(m)
        self.combo_a.setMinimumWidth(160)
        ctrl.addWidget(self.combo_a)

        ctrl.addWidget(QLabel("Method B:"))
        self.combo_b = QComboBox()
        for m in self.methods: self.combo_b.addItem(m)
        if len(self.methods) > 1:
            self.combo_b.setCurrentIndex(1)
        self.combo_b.setMinimumWidth(160)
        ctrl.addWidget(self.combo_b)

        ctrl.addStretch()

        self.lbl_frame = QLabel("Frame 0/0")
        self.lbl_frame.setMinimumWidth(100)
        ctrl.addWidget(self.lbl_frame)

        root.addLayout(ctrl)

        # ── Video panels ──────────────────────────────────────────────────────
        panels_row = QHBoxLayout()
        panels_row.setSpacing(6)

        self.panel_gt = VideoPanel("Ground Truth")
        self.panel_a  = VideoPanel("Method A")
        self.panel_b  = VideoPanel("Method B")
        for p in (self.panel_gt, self.panel_a, self.panel_b):
            panels_row.addWidget(p, stretch=1)

        root.addLayout(panels_row, stretch=1)

        # ── Metrics table ─────────────────────────────────────────────────────
        self.metrics_table = MetricsTable()
        root.addWidget(self.metrics_table)

        # ── Slider + playback ────────────────────────────────────────────────
        slider_row = QHBoxLayout()

        self.btn_prev_clip = QPushButton("◀◀ Prev Clip")
        self.btn_prev      = QPushButton("◀ Prev")
        self.btn_play      = QPushButton("▶  Play")
        self.btn_next      = QPushButton("Next ▶")
        self.btn_next_clip = QPushButton("Next Clip ▶▶")
        for btn in (self.btn_prev_clip, self.btn_prev,
                    self.btn_play, self.btn_next, self.btn_next_clip):
            btn.setFixedHeight(32)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)

        slider_row.addWidget(self.btn_prev_clip)
        slider_row.addWidget(self.btn_prev)
        slider_row.addWidget(self.btn_play)
        slider_row.addWidget(self.slider, stretch=1)
        slider_row.addWidget(self.btn_next)
        slider_row.addWidget(self.btn_next_clip)

        root.addLayout(slider_row)

        # ── Overlay toggles ───────────────────────────────────────────────────
        toggle_row = QHBoxLayout()
        self.chk_boxes  = QCheckBox("Boxes [B]");   self.chk_boxes.setChecked(True)
        self.chk_ids    = QCheckBox("IDs [I]");     self.chk_ids.setChecked(True)
        self.chk_conf   = QCheckBox("Confidence [C]")
        self.chk_trails = QCheckBox("Trails [T]")
        for chk in (self.chk_boxes, self.chk_ids, self.chk_conf, self.chk_trails):
            toggle_row.addWidget(chk)
        toggle_row.addStretch()

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color:#888; font-size:11px;")
        toggle_row.addWidget(self.lbl_status)

        root.addLayout(toggle_row)

    def _connect_signals(self):
        self.combo_clip.currentIndexChanged.connect(self._on_clip_changed)
        self.combo_a.currentIndexChanged.connect(self._refresh)
        self.combo_b.currentIndexChanged.connect(self._refresh)
        self.slider.valueChanged.connect(self._on_slider)
        self.btn_prev.clicked.connect(self._prev_frame)
        self.btn_next.clicked.connect(self._next_frame)
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_prev_clip.clicked.connect(lambda: self._step_clip(-1))
        self.btn_next_clip.clicked.connect(lambda: self._step_clip(+1))
        for chk, attr in [(self.chk_boxes,"show_boxes"),(self.chk_ids,"show_ids"),
                          (self.chk_conf,"show_conf"),(self.chk_trails,"show_trails")]:
            chk.stateChanged.connect(lambda _, a=attr, c=chk:
                                     setattr(self,a,c.isChecked()) or self._refresh())

    def _setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Left),        self).activated.connect(self._prev_frame)
        QShortcut(QKeySequence(Qt.Key_Right),       self).activated.connect(self._next_frame)
        QShortcut(QKeySequence("Ctrl+Left"),        self).activated.connect(lambda: self._jump(-10))
        QShortcut(QKeySequence("Ctrl+Right"),       self).activated.connect(lambda: self._jump(+10))
        QShortcut(QKeySequence(Qt.Key_Up),          self).activated.connect(lambda: self._step_clip(-1))
        QShortcut(QKeySequence(Qt.Key_Down),        self).activated.connect(lambda: self._step_clip(+1))
        QShortcut(QKeySequence(Qt.Key_Space),       self).activated.connect(self._toggle_play)
        QShortcut(QKeySequence("B"),                self).activated.connect(
            lambda: self.chk_boxes.setChecked(not self.chk_boxes.isChecked()))
        QShortcut(QKeySequence("I"),                self).activated.connect(
            lambda: self.chk_ids.setChecked(not self.chk_ids.isChecked()))
        QShortcut(QKeySequence("C"),                self).activated.connect(
            lambda: self.chk_conf.setChecked(not self.chk_conf.isChecked()))
        QShortcut(QKeySequence("T"),                self).activated.connect(
            lambda: self.chk_trails.setChecked(not self.chk_trails.isChecked()))
        QShortcut(QKeySequence("Q"),                self).activated.connect(self.close)

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background:#1e1e1e; color:#ddd; }
            QLabel { color:#ccc; font-size:12px; }
            QComboBox {
                background:#2d2d2d; color:#ddd; border:1px solid #555;
                border-radius:3px; padding:3px 8px; font-size:12px;
            }
            QComboBox QAbstractItemView { background:#2d2d2d; color:#ddd; }
            QPushButton {
                background:#3a3a3a; color:#ddd; border:1px solid #555;
                border-radius:3px; padding:4px 12px; font-size:12px;
            }
            QPushButton:hover  { background:#4a4a4a; }
            QPushButton:pressed{ background:#2a2a2a; }
            QSlider::groove:horizontal { height:6px; background:#444; border-radius:3px; }
            QSlider::handle:horizontal {
                width:14px; height:14px; margin:-4px 0;
                background:#5a8fd8; border-radius:7px;
            }
            QCheckBox { color:#ccc; spacing:6px; font-size:12px; }
            QCheckBox::indicator { width:16px; height:16px; }
        """)

    # ── Clip loading ──────────────────────────────────────────────────────────

    def _load_clip(self, idx: int):
        self.clip_idx  = idx
        self.combo_clip.blockSignals(True)
        self.combo_clip.setCurrentIndex(idx)
        self.combo_clip.blockSignals(False)

        clip_name = self.clip_names[idx]
        img_dir   = self.clips_dir / "annotations" / clip_name / "img1"
        self._frames = sorted(img_dir.glob("frame_*.png"))
        if not self._frames:
            self._frames = sorted(img_dir.glob("frame_*.jpg"))

        self.n_frames = len(self._frames)
        self.slider.setMaximum(max(0, self.n_frames - 1))
        self.slider.blockSignals(True)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.frame_idx = 0

        # Load GT
        gt_txt = self.clips_dir/"annotations"/clip_name/"gt"/"gt.txt"
        self._gt = load_mot(gt_txt) if gt_txt.exists() else {}

        # Load method tracks (lazy-cache per clip)
        self._method_tracks: dict[str, dict] = {}
        for method in self.methods:
            t_path = self.tracks_dir / method / clip_name / "gt.txt"
            self._method_tracks[method] = load_mot(t_path) if t_path.exists() else {}

        self._update_metrics()
        self._refresh()

    def _get_tracks(self, method: str) -> dict:
        return self._method_tracks.get(method, {})

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _refresh(self):
        if not self._frames:
            return
        fi = self.frame_idx
        img = cv2.imread(str(self._frames[fi]))
        if img is None:
            return

        frame_1based = fi + 1
        method_a = self.combo_a.currentText()
        method_b = self.combo_b.currentText()

        kw = dict(frame_1based=frame_1based,
                  show_boxes=self.show_boxes, show_ids=self.show_ids,
                  show_conf=self.show_conf,   show_trails=self.show_trails)

        self.panel_gt.set_pixmap(self.renderer.to_pixmap(
            self.renderer.render(img, self._gt,                       gt_style=True, **kw)))
        self.panel_a.set_pixmap(self.renderer.to_pixmap(
            self.renderer.render(img, self._get_tracks(method_a),     gt_style=False, **kw)))
        self.panel_b.set_pixmap(self.renderer.to_pixmap(
            self.renderer.render(img, self._get_tracks(method_b),     gt_style=False, **kw)))

        self.panel_a.set_title(f"Method A — {method_a}")
        self.panel_b.set_title(f"Method B — {method_b}")
        self.lbl_frame.setText(f"Frame {fi+1} / {self.n_frames}")

        self.slider.blockSignals(True)
        self.slider.setValue(fi)
        self.slider.blockSignals(False)

    def _update_metrics(self):
        clip_name = self.clip_names[self.clip_idx]
        method_a  = self.combo_a.currentText()
        method_b  = self.combo_b.currentText()

        def get_clip_scores(method: str) -> dict:
            return (self.scores.get(method, {})
                               .get("clips", {})
                               .get(clip_name, {}))

        sa = get_clip_scores(method_a)
        sb = get_clip_scores(method_b)
        self.metrics_table.setHorizontalHeaderLabels([method_a, method_b])
        self.metrics_table.update_scores(sa, sb)

    # ── Playback / navigation ─────────────────────────────────────────────────

    def _prev_frame(self):
        self._jump(-1)

    def _next_frame(self):
        if self.frame_idx >= self.n_frames - 1:
            self._play_timer.stop()
            self.btn_play.setText("▶  Play")
        else:
            self._jump(+1)

    def _jump(self, delta: int):
        self.frame_idx = max(0, min(self.n_frames-1, self.frame_idx+delta))
        self._refresh()

    def _on_slider(self, value: int):
        self.frame_idx = value
        self._refresh()

    def _toggle_play(self):
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.btn_play.setText("▶  Play")
        else:
            self._play_timer.start()
            self.btn_play.setText("⏸  Pause")

    def _step_clip(self, delta: int):
        new_idx = (self.clip_idx + delta) % len(self.clip_names)
        self._load_clip(new_idx)

    def _on_clip_changed(self, idx: int):
        self._load_clip(idx)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _vsep() -> QFrame:
    sep = QFrame(); sep.setFrameShape(QFrame.VLine)
    sep.setStyleSheet("color:#555;"); sep.setFixedWidth(2)
    return sep


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir",  required=True)
    ap.add_argument("--tracks-dir", required=True)
    ap.add_argument("--scores",     default=None)
    args = ap.parse_args()

    scores = load_scores(Path(args.scores)) if args.scores else {}

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    win = MainWindow(
        clips_dir  = Path(args.clips_dir),
        tracks_dir = Path(args.tracks_dir),
        scores     = scores,
    )
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
