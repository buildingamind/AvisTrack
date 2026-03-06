# AvisTrack 🐦

> A permanent, model-agnostic tracking engine for avian behavioral experiments.
> Researchers should spend time analyzing behavior — not rewriting video processing loops.

---

## Table of Contents

1. [Background & Motivation](#1-background--motivation)
2. [Repository Ecosystem](#2-repository-ecosystem)
3. [Architecture](#3-architecture)
4. [Project Structure](#4-project-structure)
5. [Setup](#5-setup)
6. [Configuration](#6-configuration)
7. [Tools](#7-tools)
8. [Offline Batch Processing](#8-offline-batch-processing)
9. [Model Evaluation](#9-model-evaluation)
10. [Real-time Integration (ChamberBroadcaster)](#10-real-time-integration-chamberbroadcaster)
11. [Dataset & Sampling Workflow](#11-dataset--sampling-workflow)
12. [Running Tests](#12-running-tests)

---

## 1. Background & Motivation

Our primary animal model is the chick. While the subject is constant, our experimental environments and analytical needs are highly variable — from open-field tests to complex social interaction paradigms in custom chambers.

We frequently transition between:

- **Real-time vs. Offline analysis** — low-latency live interventions vs. high-fidelity post-hoc processing
- **Single vs. Multi-bird tracking** — one subject vs. managing occlusions and ID assignment in groups of 9
- **Macro vs. Micro movements** — bounding-box spatial positioning vs. keypoint-level ethological observations

Because setups vary so drastically, we rely on YOLO, DeepLabCut (DLC), and Vision Transformers (ViT). Historically, every new paradigm required writing a new tracking script from scratch — rewriting video reading logic, re-implementing coordinate smoothing, and hardcoding model paths. Comparing fine-tuned models became an administrative nightmare.

**AvisTrack** is the permanent, stable foundation. It treats the core mechanics of tracking (reading frames, handling outputs, smoothing data) as a constant, and treats the experimental variables (the chamber, the model, the weights) as easily swappable configuration.

---

## 2. Repository Ecosystem

AvisTrack is one of three purpose-specific repositories. Each has a single, well-defined responsibility.

```
┌─────────────────────────────────────────────────────────────┐
│  ChamberBroadcaster                                         │
│  Recording: camera management, video saving, streaming      │
│  (uses AvisTrack for tracking via a thin processor wrapper) │
└───────────────────────┬─────────────────────────────────────┘
                        │ frames (live)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  AvisTrack   ◄── THIS REPO                                  │
│  Tracking engine: YOLO / DLC / ViT backends                 │
│  Input:  raw or perspective-corrected video frames          │
│  Output: MOT-format .txt or Parquet tracking data           │
└───────────────────────┬─────────────────────────────────────┘
                        │ MOT / Parquet files
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Collective-Chamber-Metrics                                 │
│  Analysis: NNI, convex hull, huddle count, dashboard        │
│  (reads AvisTrack output — no changes needed)               │
└─────────────────────────────────────────────────────────────┘
```

**Data always lives on the external drive / NAS.** Config files in this repo contain paths pointing to the drive; no actual video or tracking data is committed to git.

---

## 3. Architecture

### 3.1 Design Principles

| Principle | Implication |
|-----------|-------------|
| **Backends are pluggable** | YOLO, DLC, ViT all expose the same `.update(frame) → list[Detection]` interface. The rest of the codebase never needs to know which model is running. |
| **Config-driven** | All paths, model settings, and chamber parameters live in a YAML file. Nothing is hardcoded. |
| **Pipeline steps are optional** | Perspective transform (`transformer.py`) is a pipeline step that can be enabled or disabled per experiment. Not all models or setups need it. |
| **Frame source is abstracted** | `frame_source.py` makes the tracking loop identical whether the input is a video file (offline) or a live camera feed from ChamberBroadcaster (real-time). |
| **Data lives on the drive** | Manifests and ROI JSON files on the external drive are the source of truth. AvisTrack only stores configs and code. |

### 3.2 Data Flow

```
External Drive / NAS
  00_raw_videos/           ←─ source videos (.mkv)
  01_Dataset_MOT_Format/
    train/ val/ test/      ←─ sampled clips (.mp4)
  02_Global_Metadata/
    camera_rois.json       ←─ 4-corner ROI per video
    *_manifest.txt         ←─ clip registry (prevents re-sampling)
  03_Model_Training/
    Yolo11n/.../best.pt    ←─ trained weights

         │  (paths referenced in configs/wave3_collective.yaml)
         ▼

  [AvisTrack pipeline]
    FrameSource            reads video file or live camera
    PerspectiveTransformer warp frame to top-down view (optional)
    YoloOfflineTracker     detect + IoU match + interpolate
         │
         ▼
  MOT .txt  →  Collective-Chamber-Metrics
  Parquet   →  Collective-Chamber-Metrics
```

### 3.3 Backend Interface

Every backend returns a list of `Detection` objects:

```python
@dataclass
class Detection:
    track_id:   int           # persistent ID across frames
    x: float                  # top-left x  (perspective-corrected)
    y: float                  # top-left y
    w: float                  # bounding box width
    h: float                  # bounding box height
    confidence: float
    keypoints:  list[dict]    # [{"label", "x", "y", "likelihood"}]
                              # empty for YOLO; filled for DLC / ViT
```

YOLO fills `x/y/w/h`; DLC and ViT fill `keypoints`. The downstream code handles both uniformly.

### 3.4 YOLO — Two Modes

| Mode | Class | When to use |
|------|-------|-------------|
| `offline` | `YoloOfflineTracker` | Batch-processing stored videos. Multi-subject. IoU + Hungarian matching. Linear interpolation for gap-filling. |
| `realtime` | `YoloRealtimeTracker` | Live camera feed from ChamberBroadcaster. Single-subject. Non-blocking background thread. Kalman filter smoothing. |

---

## 4. Project Structure

```
AvisTrack/
│
├── avistrack/                    # Installable Python package
│   ├── __init__.py               # load_tracker() factory entry point
│   │
│   ├── backends/
│   │   ├── base.py               # Abstract TrackerBackend + Detection dataclass
│   │   ├── yolo/
│   │   │   ├── realtime.py       # YOLO + Kalman (real-time, single-subject)
│   │   │   └── offline.py        # YOLO + IoU + interpolation (batch, multi-subject)
│   │   ├── dlc.py                # DeepLabCut keypoint backend
│   │   └── vit.py                # DorsalVentralNet (ViT) keypoint backend
│   │
│   ├── core/
│   │   ├── transformer.py        # Perspective transform (optional pipeline step)
│   │   └── frame_source.py       # Unified frame iterator (file or live camera)
│   │
│   └── config/
│       ├── schema.py             # Pydantic v2 config schema
│       └── loader.py             # YAML loader with {root} placeholder resolution
│
├── configs/                      # Per-experiment YAML configs (no data, only paths)
│   ├── template.yaml             # Annotated template — start here
│   ├── wave3_collective.yaml     # Wave 3 Collective Chamber
│   └── wave3_vr.yaml             # Wave 3 VR Chamber
│
├── tools/
│   ├── pick_rois.py              # Interactive ROI corner picker → writes camera_rois.json
│   └── sample_clips.py          # Sample clips from raw videos, check manifest for overlaps
│
├── eval/
│   └── run_eval.py               # Compare multiple weight files on test set → CSV report
│
├── cli/
│   └── run_batch.py              # Full-data batch processing with resume / multiprocessing
│
├── tests/
│   ├── test_config.py
│   └── test_transformer.py
│
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

---

## 5. Setup

### 5.1 Create the Conda environment

```bash
conda create -n avistrack python=3.11 -y
conda activate avistrack
```

### 5.2 Install AvisTrack

From the repo root — use editable mode so code changes take effect immediately:

```bash
cd /path/to/AvisTrack
pip install -e .
pip install -r requirements.txt
```

### 5.3 (Optional) Install backend-specific dependencies

| Backend | Extra install |
|---------|---------------|
| YOLO    | `pip install ultralytics filterpy` (included in requirements.txt) |
| DLC     | `pip install deeplabcut` |
| ViT     | `pip install -e /path/to/ChamberBroadcaster` (for DorsalVentralNet) |

### 5.4 Verify

```bash
conda run -n avistrack python -m pytest tests/ -v
# Expected: 6 passed
```

---

## 6. Configuration

All experiments are driven by a single YAML file. Copy `configs/template.yaml` and fill in your paths.

```yaml
experiment: "Wave3_Collective"

# All '{root}' placeholders are auto-resolved at load time
drive:
  root:           "/media/woodlab/104-A/Wave3"
  raw_videos:     "{root}/00_raw_videos"
  roi_file:       "{root}/02_Global_Metadata/camera_rois.json"
  train_manifest: "{root}/02_Global_Metadata/train_data_manifest.txt"
  val_manifest:   "{root}/02_Global_Metadata/val_tuning_manifest.txt"
  test_manifest:  "{root}/02_Global_Metadata/test_golden_manifest.txt"

chamber:
  n_subjects:  9
  fps:         30
  target_size: [640, 640]   # inference resolution; null = preserve aspect ratio

model:
  backend: "yolo"           # "yolo" | "dlc" | "vit"
  mode:    "offline"        # "offline" | "realtime"  (YOLO only)
  weights: "{root}/03_Model_Training/Yolo11n/Train_v1_best/weights/best.pt"

tracking:
  conf_threshold: 0.2
  max_gap_frames: 30

pipeline:
  - step: transform         # remove this line if video is already cropped
  - step: track

output:
  format: "mot"             # "mot" (txt) or "parquet"
  dir:    "{root}/../outputs/W3_COLL"
```

### Configuration key reference

| Key | Description |
|-----|-------------|
| `drive.root` | Mount point of the external drive / NAS |
| `chamber.n_subjects` | Number of animals; caps the number of tracked IDs |
| `chamber.target_size` | `[w, h]` for perspective-corrected output; `null` = auto |
| `model.backend` | Which tracker to load |
| `model.mode` | `offline` = batch processing; `realtime` = live CB feed |
| `tracking.max_gap_frames` | Frames to interpolate across a detection gap |
| `pipeline` | List of steps; remove `transform` if no ROI warp needed |

---

## 7. Tools

### 7.1 Pick ROI corners — `tools/pick_rois.py`

Scans a folder for videos, shows the first frame of each one that has no ROI entry yet, and lets you click the 4 chamber corners interactively. Results are appended to `camera_rois.json` on the drive after each video (crash-safe).

```bash
conda activate avistrack
python tools/pick_rois.py \
    --video-dir /media/woodlab/104-A/Wave3/00_raw_videos \
    --roi-file  /media/woodlab/104-A/Wave3/02_Global_Metadata/camera_rois.json
```

**Controls in the window:**

| Key | Action |
|-----|--------|
| Left-click | Place a corner (order: TL → TR → BR → BL) |
| `R` | Reset all corners for this video |
| `S` / Enter | Save and move to next video |
| `Q` / Esc | Skip this video |

Add `--force` to re-pick ROIs for videos that already have an entry.

---

### 7.2 Sample clips — `tools/sample_clips.py`

Randomly samples short clips from raw videos and saves them to the drive. Reads all three existing manifests (train / val / test) first, so new clips are guaranteed not to overlap with anything already sampled.

```bash
python tools/sample_clips.py \
    --config     configs/wave3_collective.yaml \
    --split      train \
    --n          20 \
    --duration   3 \
    --output-dir /media/woodlab/104-A/Wave3/01_Dataset_MOT_Format/train
```

Each new clip is appended to the corresponding manifest file immediately (crash-safe). The manifest format matches what already exists on the drive:

```
Clip_Filename, Original_Video_Path, Start_Time, Duration
Wave3_..._TRAIN_s49508.mp4, /media/.../Day12_RGB.mkv, 49508.33, 3
```

---

## 8. Offline Batch Processing

Processes **all** raw videos with a single command. Supports:
- Automatic resume (skips files whose output already exists)
- Configurable multiprocessing (`--workers`)
- Per-video progress logging

```bash
conda activate avistrack
python cli/run_batch.py --config configs/wave3_collective.yaml
```

```bash
# Control parallelism (reduce if GPU runs out of memory)
python cli/run_batch.py --config configs/wave3_collective.yaml --workers 2

# Force re-run even if output already exists
python cli/run_batch.py --config configs/wave3_collective.yaml --force
```

Output is written to the `output.dir` path set in the config. Each video produces one `.txt` file in MOT format, ready to be consumed by Collective-Chamber-Metrics.

---

## 9. Model Evaluation

Compare multiple trained weight files on the same test set. For each clip in `01_Dataset_MOT_Format/test_golden/`, a matching `.txt` ground-truth file must exist in the same folder (MOT format).

```bash
python eval/run_eval.py \
    --config  configs/wave3_collective.yaml \
    --weights \
        /media/.../Train_v1_best/weights/best.pt \
        /media/.../Train_v2_best/weights/best.pt \
    --output  eval/reports/wave3_coll_20260306.csv
```

**Output CSV example:**

| Weights | Precision | Recall | F1 | ID_Switches | FPS |
|---------|-----------|--------|----|-------------|-----|
| v1_best.pt | 0.912 | 0.883 | 0.897 | 14 | 43.2 |
| v2_best.pt | 0.941 | 0.921 | 0.931 | 7  | 41.8 |

### Typical iteration cycle

```
Run eval → performance not satisfactory?
    │
    ▼
sample_clips.py --split train --n 20   (new clips, no overlap with existing)
    │
    ▼
Retrain YOLO with expanded dataset
    │
    ▼
Run eval again → compare v1 vs v2 vs v3 in one command
    │
    ▼
Pick best weights → run_batch.py on full dataset
```

---

## 10. Real-time Integration (ChamberBroadcaster)

AvisTrack is installed as a package in ChamberBroadcaster's environment:

```bash
# One-time setup, run from CB's machine / conda env
pip install -e /path/to/AvisTrack
```

Then add a single thin wrapper file to ChamberBroadcaster — **no other CB files need to change**:

```python
# ChamberBroadcaster/chamber_broadcaster/processors/avistrack_processor.py

import avistrack
from .tracking import TrackingProcessor

class AvisTrackProcessor(TrackingProcessor):
    """
    Thin wrapper: delegates all tracking logic to AvisTrack.
    CB only passes frames in and receives Detection lists out.
    """
    def __init__(self, config: dict, **kwargs):
        cfg_path = config["avistrack_config"]   # path to a configs/*.yaml
        self.tracker = avistrack.load_tracker(cfg_path)

    def process(self, context: dict) -> dict:
        frame = context.get("frame")
        if frame is None:
            return context
        detections = self.tracker.update(frame)
        # Convert to CB's expected tracking_data format
        context["tracking_data"] = {
            "animals": [
                [{"label": kp["label"], "x": kp["x"], "y": kp["y"],
                  "likelihood": kp["likelihood"]}
                 for kp in d.keypoints]
                for d in detections
            ],
            "tracker": "AvisTrack",
        }
        return context

    def release(self):
        self.tracker.release()
```

In `config.yaml` for ChamberBroadcaster:

```yaml
- name: "AvisTrackProcessor"
  config:
    avistrack_config: "/path/to/AvisTrack/configs/wave3_vr.yaml"
```

The existing `YoloKalmanProcessor`, `JoshuTrackingProcessor`, and `DLCTrackingProcessor` in CB remain untouched. New experiments simply use `AvisTrackProcessor` instead.

---

## 11. Dataset & Sampling Workflow

Data on the external drive follows this layout (established in Wave 3):

```
/media/woodlab/104-A/Wave3/
├── 00_raw_videos/                 continuous .mkv recordings
├── 01_Dataset_MOT_Format/
│   ├── train/                     sampled .mp4 clips (training set)
│   ├── val_tuning/                sampled .mp4 clips (validation set)
│   └── test_golden/               sampled .mp4 clips + .txt ground truth
├── 02_Global_Metadata/
│   ├── camera_rois.json           {video_name: [[x,y],[x,y],[x,y],[x,y]]}
│   ├── train_data_manifest.txt    Clip_Filename, Original_Video_Path, Start_Time, Duration
│   ├── val_tuning_manifest.txt
│   ├── test_golden_manifest.txt
│   ├── valid_intervals.json
│   └── danger_intervals.json
└── 03_Model_Training/
    └── Yolo11n/
        └── Train_v1_best/weights/best.pt
```

**The manifests are the source of truth for what has been sampled.** `sample_clips.py` reads all three before picking any new clips, ensuring zero time-overlap across the entire dataset.

---

## 12. Running Tests

```bash
conda activate avistrack
cd /path/to/AvisTrack
python -m pytest tests/ -v
```

```
tests/test_config.py::test_load_minimal_config              PASSED
tests/test_config.py::test_root_placeholder_resolution      PASSED
tests/test_transformer.py::test_transform_output_size_fixed PASSED
tests/test_transformer.py::test_transform_output_size_auto  PASSED
tests/test_transformer.py::test_from_roi_file               PASSED
tests/test_transformer.py::test_from_roi_file_missing_key   PASSED

6 passed in 0.20s
```
