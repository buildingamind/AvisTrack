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
6. [Quick Start (New Experiment)](#6-quick-start-new-experiment)
7. [Configuration](#7-configuration)
8. [Tools](#8-tools)
9. [Time Calibration](#9-time-calibration)
10. [Dataset & Sampling Workflow](#10-dataset--sampling-workflow)
11. [Offline Batch Processing](#11-offline-batch-processing)
12. [Model Evaluation](#12-model-evaluation)
13. [Real-time Integration (ChamberBroadcaster)](#13-real-time-integration-chamberbroadcaster)
14. [Running Tests](#14-running-tests)
15. [Common Recipes](#15-common-recipes)

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
│   │   ├── frame_source.py       # Unified frame iterator (file or live camera)
│   │   └── time_lookup.py        # Frame↔time bidirectional conversion (OCR calibration)
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
│   ├── init_config.py            # ★ Interactive config wizard — creates YAML configs
│   ├── pick_rois.py              # ROI picker + validator → camera_rois.json
│   ├── sample_clips.py           # Sample clips from raw videos with ROI transform
│   └── calibrate_time.py         # OCR-based frame↔time calibration (Google Vision)
│
├── eval/
│   └── run_eval.py               # Compare multiple weight files on test set → CSV report
│
├── cli/
│   └── run_batch.py              # Full-data batch processing with resume / multiprocessing
│
├── tests/
│   ├── test_config.py
│   ├── test_transformer.py
│   └── test_time_lookup.py       # TimeLookup interpolation + round-trip tests
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
# Expected: 25 passed
```

---

## 6. Quick Start (New Experiment)

This section walks through the **complete workflow** from zero to processed data. Every command below assumes `conda activate avistrack` has been run.

### Step 0: Prepare your external drive

Create the standard directory structure on your drive:

```
/media/woodlab/<DRIVE_NAME>/<WaveX>/
├── 00_raw_videos/            ← put your .mkv recordings here
├── 01_Dataset_MOT_Format/
│   ├── train/
│   ├── val_tuning/
│   └── test_golden/
├── 02_Global_Metadata/       ← manifests + ROIs will go here
└── 03_Model_Training/        ← put trained weights here
```

You can create only the folders you need — the tools will create missing ones for you.

### Step 1: Create a config file (GUI wizard)

```bash
python tools/init_config.py
```

A GUI window opens with:
- **Browse buttons** for every path (folders and files) — no manual path typing
- **Dropdowns** for backend, mode, and output format
- **Spinboxes** for numeric fields (subjects, FPS, threshold)
- **Auto-detect** button that scans the drive root for standard directories and weight files
- **Save dialog** to pick where the YAML goes
- Path validation with warnings shown before saving

You can pre-fill the root:

```bash
python tools/init_config.py --root /media/woodlab/104-A/Wave2 --name Wave2_Collective
```

If tkinter is not available (e.g. headless server), fall back to terminal mode:

```bash
python tools/init_config.py --cli
```

> **Alternatively**, copy and edit the template manually:
> ```bash
> cp configs/template.yaml configs/wave2_collective.yaml
> # Edit with your favorite editor
> ```

### Step 2: Pick ROI corners (if using perspective transform)

```bash
python tools/pick_rois.py --config configs/wave2_collective.yaml
```

The config provides `raw_videos` and `roi_file` automatically.  
Click the 4 chamber corners in each video frame. Results are saved after every video (crash-safe).

Then validate that every video is covered:

```bash
python tools/pick_rois.py validate --config configs/wave2_collective.yaml
```

### Step 3: Calibrate frame↔time mapping (if videos have burn-in timestamps)

If your videos have a burn-in timestamp overlay (e.g. security camera style), you can build a frame → real-time mapping using sparse OCR. **Do this before sampling clips** so that time-based exclusions and intervals can be applied.

```bash
# 3a. Pick the timestamp region on each video
python tools/calibrate_time.py roi --config configs/wave2_collective.yaml

# 3b. Run calibration (shows cost estimate, preview, auto-detects format, then asks to confirm)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
python tools/calibrate_time.py calibrate --config configs/wave2_collective.yaml

# 3c. Post-process: fix OCR errors using monotonicity + interpolation
python tools/calibrate_time.py postprocess --config configs/wave2_collective.yaml

# 3d. Verify accuracy
python tools/calibrate_time.py verify --config configs/wave2_collective.yaml
```

The `calibrate` step auto-detects the timestamp format from the OCR text (e.g. `08:35:30 PM 18-Jun-25`). Timezone indicators like `EST`, `UTC-5` are automatically stripped — the authoritative timezone comes from the YAML config.

The `postprocess` step fixes common OCR failures (missing seconds, garbled digits, API errors) using monotonicity constraints and linear interpolation between good samples. Typically recovers 95%+ of failed samples.

This creates `time_calibration.json` on the drive. Use `TimeLookup` in code to convert between frames and wall-clock time:

```python
from avistrack.core.time_lookup import TimeLookup
tl = TimeLookup.load("time_calibration.json", "Video_RGB.mkv")
tl.frame_to_datetime(5000)   # → timezone-aware datetime
tl.unix_to_frame(1730264567) # → frame index
```

### Step 4: Sample training / val / test clips

```bash
# Training clips: 20 clips × 3 seconds each
python tools/sample_clips.py \
    --config     configs/wave2_collective.yaml \
    --split      train \
    --n          20 \
    --duration   3 \
    --output-dir /media/woodlab/104-A/Wave2/01_Dataset_MOT_Format/train

# Validation clips
python tools/sample_clips.py \
    --config     configs/wave2_collective.yaml \
    --split      val \
    --n          10 \
    --duration   3 \
    --output-dir /media/woodlab/104-A/Wave2/01_Dataset_MOT_Format/val_tuning

# Golden test clips: 20 clips × 20 seconds (600 frames @ 30fps)
python tools/sample_clips.py \
    --config     configs/wave2_collective.yaml \
    --split      test \
    --n          20 \
    --duration   20 \
    --output-dir /media/woodlab/104-A/Wave2/01_Dataset_MOT_Format/test_golden
```

> **Important:** `--duration` is in **seconds**, not frames.
> To get a specific number of frames, divide by FPS:
>
> | Desired frames | FPS | `--duration` |
> |---------------|-----|-------------|
> | 90 frames     | 30  | `3`         |
> | 300 frames    | 30  | `10`        |
> | 600 frames    | 30  | `20`        |
> | 900 frames    | 30  | `30`        |

### Step 5: Train your model (outside AvisTrack)

Label the sampled clips in CVAT / Roboflow / Label Studio, then train YOLO:

```bash
yolo detect train data=dataset.yaml model=yolo11n.pt epochs=100 imgsz=640
```

Put the resulting `best.pt` into `03_Model_Training/` on the drive and update the `model.weights` path in your config.

### Step 6: Evaluate model performance

```bash
python eval/run_eval.py \
    --config  configs/wave2_collective.yaml \
    --weights /media/.../weights/best.pt \
    --output  eval/reports/wave2_coll.csv
```

### Step 7: Run batch processing on all raw videos

```bash
python cli/run_batch.py --config configs/wave2_collective.yaml
```

Done! MOT output files will appear in the `output.dir` specified in your config.

---

## 7. Configuration

All experiments are driven by a single YAML file. You can create one in three ways:

| Method | When to use |
|--------|-------------|
| `python tools/init_config.py` | **Recommended.** Interactive wizard with validation and auto-detection. |
| Copy `configs/template.yaml` | When you want full manual control. |
| Copy an existing wave config | When the new experiment is similar to one you've already set up. |

### 7.1 Example config

```yaml
experiment: "Wave3_Collective"

# All '{root}' placeholders are auto-resolved at load time
drive:
  root:           "/media/woodlab/104-A/Wave3"
  raw_videos:     "{root}/00_raw_videos"
  roi_file:       "{root}/02_Global_Metadata/camera_rois.json"
  ocr_roi:        "{root}/02_Global_Metadata/ocr_roi.json"
  time_calibration: "{root}/02_Global_Metadata/time_calibration.json"
  train_manifest: "{root}/02_Global_Metadata/train_data_manifest.txt"
  val_manifest:   "{root}/02_Global_Metadata/val_tuning_manifest.txt"
  test_manifest:  "{root}/02_Global_Metadata/test_golden_manifest.txt"

chamber:
  n_subjects:  9
  fps:         30
  target_size: [640, 640]

model:
  backend: "yolo"
  mode:    "offline"
  weights: "{root}/03_Model_Training/Yolo11n/Train_v1_best/weights/best.pt"

tracking:
  conf_threshold: 0.2
  max_gap_frames: 30

pipeline:
  - step: transform
  - step: track

output:
  format: "mot"
  dir:    "{root}/../outputs/W3_COLL"

# Frame↔time mapping (burn-in timestamp OCR)
time:
  timezone:    "America/New_York"
  time_format: "auto"
```

### 7.2 Key reference

| Key | Type | Description |
|-----|------|-------------|
| `experiment` | string | Human-readable name for this experiment |
| `drive.root` | string | Mount point of the external drive / NAS |
| `drive.raw_videos` | string | Directory containing source `.mkv` / `.mp4` recordings |
| `drive.roi_file` | string | Path to `camera_rois.json` (4-corner ROI per video) |
| `drive.train_manifest` | string | CSV tracking which clips have been sampled for training |
| `drive.val_manifest` | string | CSV tracking which clips have been sampled for validation |
| `drive.test_manifest` | string | CSV tracking which clips have been sampled for testing |
| `drive.ocr_roi` | string | Path to `ocr_roi.json` (burn-in timestamp crop region) |
| `drive.time_calibration` | string | Path to `time_calibration.json` (frame↔time mapping) |
| `drive.exclusions` | string | Path to `exclusions.json` (invalid intervals) |
| `chamber.n_subjects` | int | Number of animals; caps tracked IDs |
| `chamber.fps` | int | Video framerate |
| `chamber.target_size` | [w, h] | Inference resolution after perspective warp; `null` = auto |
| `model.backend` | string | `"yolo"` \| `"dlc"` \| `"vit"` |
| `model.mode` | string | `"offline"` = batch; `"realtime"` = live CB feed |
| `model.weights` | string | Path to the trained weight file |
| `tracking.conf_threshold` | float | Detection confidence threshold (0–1) |
| `tracking.max_gap_frames` | int | Max frames to interpolate across a detection gap |
| `pipeline` | list | Steps to run: `transform` (optional) and `track` |
| `output.format` | string | `"mot"` (txt) or `"parquet"` |
| `output.dir` | string | Where to write tracking output files |
| `time.timezone` | string | IANA timezone name (e.g. `America/New_York`), handles DST |
| `time.time_format` | string | `"auto"` (detect from OCR text) or explicit `strptime` format |

### 7.3 How `{root}` resolution works

Any string value in the YAML containing `{root}` is replaced with the value of `drive.root` at load time. This keeps configs DRY — change the drive mount point in one place and all paths update automatically.

```yaml
drive:
  root: "/media/woodlab/104-A/Wave3"
  raw_videos: "{root}/00_raw_videos"
  # → resolved to: /media/woodlab/104-A/Wave3/00_raw_videos
```

---

## 8. Tools

### 8.1 Config wizard — `tools/init_config.py`

GUI-based config creator with folder/file pickers, validation, and auto-detection. See [Quick Start](#6-quick-start-new-experiment) for details.

```bash
# Launch GUI wizard (default)
python tools/init_config.py

# Pre-fill drive root and experiment name
python tools/init_config.py --root /media/woodlab/104-A/Wave2 --name Wave2_Collective

# Terminal-only mode (no GUI)
python tools/init_config.py --cli

# Specify output path (CLI mode)
python tools/init_config.py --cli -o configs/my_experiment.yaml
```

**What the GUI provides:**
- **Browse…** buttons for every path field — pick folders and files from the OS file dialog
- **Auto-detect** button scans drive root for `00_raw_videos/`, `02_Global_Metadata/`, etc.
- **Dropdown** for backend (`yolo` / `dlc` / `vit`), mode, and output format
- **Spinboxes** for numeric values (no typos possible)
- Auto-finds `.pt` weight files under `03_Model_Training/` and lists them in a dropdown
- Validates all paths and shows warnings before saving
- **Save As** dialog for the output YAML file
- Falls back to CLI mode automatically if `tkinter` is not installed

---

### 8.2 ROI tool — `tools/pick_rois.py`

Two modes: **pick** (interactive corner picker, **default**) and **validate** (check format + coverage).

#### `pick` — Interactive 4-corner ROI picker (default)

Scans a folder for videos and opens a single OpenCV window with a **left info panel** and the **video frame** on the right. A random frame is shown by default. Corners from the previous video carry forward as defaults. You can navigate back and forth between videos with arrow keys.

```bash
# Default mode is pick — no subcommand needed:
python tools/pick_rois.py --config configs/wave3_collective.yaml

# Explicit subcommand also works:
python tools/pick_rois.py pick --config configs/wave3_collective.yaml

# IR only:
python tools/pick_rois.py --config configs/wave3_collective.yaml --modality ir

# All modalities (RGB first, then IR):
python tools/pick_rois.py --config configs/wave3_collective.yaml --modality all
```

| Flag | Description |
|------|-------------|
| `--config` | YAML config to auto-resolve `raw_videos` and `roi_file` |
| `--video-dir` | Override video directory (or use without config) |
| `--roi-file` | Override ROI output path (or use without config) |
| `--modality` | `rgb` (default), `ir`, or `all` (RGB first then IR) |

**Controls** (shown in the left info panel):

| Key | Action |
|-----|--------|
| Left-click | Place next corner (order: ① upper-left → ② upper-right → ③ lower-right → ④ lower-left) |
| `Z` / Ctrl+Z | Undo last corner |
| `R` | Reset all 4 corners |
| `N` | Jump to a random frame |
| ← / `A` | Go to **previous video** |
| → / `D` | Save corners & go to **next video** |
| `S` / Enter | Save corners & next video |
| `Q` / Esc | Quit the picker entirely |

**UI layout:**
- Single window: dark left panel (info + controls + corner status) + video frame on right
- Lines are drawn **progressively** as you place corners (2 corners = 1 line, 3 = 2 lines, 4 = closed polygon)
- Corner dots are color-coded with numbered labels (①②③④)

**Behavior notes:**
- All videos are shown, including ones that already have a saved ROI (marked 🟡 for review, 🟢 for new)
- Each video starts with the previous video's corners pre-loaded (press `R` to clear)
- If a video already has a saved ROI, those saved corners are loaded instead
- Use ← / → to navigate between videos — you can go back to review/edit previous ones
- Results are saved to JSON after every forward move (crash-safe)

#### `validate` — Check ROI file before sampling

Run this before `sample_clips.py` to catch problems early.  Checks:
1. File exists
2. Valid JSON
3. Every entry has exactly 4 `[x, y]` corner points (numbers)
4. Every target video (filtered by `--modality`) has an ROI entry

```bash
# Using a config (auto-resolves roi_file + raw_videos paths):
python tools/pick_rois.py validate --config configs/wave3_collective.yaml

# Or specify paths directly:
python tools/pick_rois.py validate \
    --roi-file  /media/woodlab/104-A/Wave3/02_Global_Metadata/camera_rois.json \
    --video-dir /media/woodlab/104-A/Wave3/00_raw_videos \
    --modality  rgb
```

Example output (all good):
```
📂 Checking ROI for 17 RGB video(s) in .../00_raw_videos
📄 ROI file: .../camera_rois.json

  ✅ ROI file has 18 entries
  ✅ All 18 entries have valid 4-corner format
  ✅ All 17 videos have ROI entries

✅ ROI validation passed.
```

Example output (problems):
```
  ✅ ROI file has 15 entries
  ✅ All 15 entries have valid 4-corner format
  ❌ 2/17 videos have NO ROI entry:
     Wave3_CollectiveDanger_3_Day15_Camera1_RGB.mkv
     Wave3_CollectiveDanger_3_Day16_Camera1_RGB.mkv

❌ ROI validation FAILED.
```

#### ROI file format (`camera_rois.json`)

```json
{
    "Wave3_CollectiveChamber_Day1_Cam1_RGB.mkv": [
        [315, 61],   // TL
        [889, 63],   // TR
        [870, 629],  // BR
        [327, 622]   // BL
    ],
    "Wave3_CollectiveDanger_2_Day2_Camera1_RGB.mkv": [
        [314, 104],
        [887, 103],
        [870, 664],
        [328, 657]
    ]
}
```

Keys are video **basenames** (with extension). Each value is a list of exactly 4 `[x, y]` integer pairs representing chamber corners.

---

### 8.3 Sample clips — `tools/sample_clips.py`

Randomly samples short clips from raw **RGB** videos (default), applies ROI perspective-correction on each frame, and appends records to the manifest.  Uses frame-count-weighted sampling so longer videos get proportionally more clips.

Before sampling, the tool runs `validate_roi_file()` to ensure the ROI file exists, is well-formed, and covers every target video. If validation fails, it prints the exact `pick_rois.py` command to fix it.

```bash
python tools/sample_clips.py \
    --config     configs/wave3_collective.yaml \
    --split      train \
    --n          20 \
    --duration   3 \
    --output-dir /media/woodlab/104-A/Wave3/01_Dataset_MOT_Format/train
```

**All flags:**

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--config` | Yes | — | Path to YAML config file |
| `--split` | Yes | — | `train` \| `val` \| `test` — determines which manifest to append to |
| `--n` | Yes | — | Number of clips to sample |
| `--duration` | No | `3` | Clip length in **seconds** |
| `--output-dir` | Yes | — | Where to write the `.mp4` clip files |
| `--seed` | No | `42` | Random seed for reproducibility |
| `--modality` | No | `rgb` | `rgb` or `ir` — filter videos by filename keyword |
| `--min-gap` | No | `5` | Minimum gap in **minutes** between clips from the same video |
| `--no-transform` | No | — | Skip ROI crop / perspective-correction (extract raw clips) |

**Duration is in seconds — conversion table:**

| Frames | @ 30 fps | `--duration` |
|--------|----------|-------------|
| 90     | 3s       | `3`         |
| 300    | 10s      | `10`        |
| 600    | 20s      | `20`        |
| 900    | 30s      | `30`        |

Each new clip is appended to the corresponding manifest file immediately (crash-safe). The manifest CSV format:

```
Clip_Filename, Original_Video_Path, Start_Time, Duration
Wave3_..._TRAIN_s49508.mp4, /media/.../Day12_RGB.mkv, 49508.33, 3
```

---

### 8.4 Time calibration — `tools/calibrate_time.py`

Builds a mapping from video frame numbers to wall-clock time using sparse OCR
of the burn-in timestamp overlay. Uses the **Google Cloud Vision** API.

Four subcommands:

| Subcommand | Purpose |
|------------|---------|
| `roi` | Iterate through all videos — pick/confirm the timestamp crop region for each one |
| `calibrate` | Run sparse OCR across all videos → `time_calibration.json` (auto-detects timestamp format) |
| `postprocess` | Fix OCR errors using monotonicity constraints + linear interpolation |
| `verify` | Spot-check calibration accuracy against OCR ground truth |

```bash
# 1. Set Google Cloud credentials
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# 2. Pick the timestamp region (per-video, previous carries forward)
python tools/calibrate_time.py roi --config configs/wave3_collective.yaml

# 3. Run calibration (cost estimate → auto-detect format → preview → confirmation → full run)
python tools/calibrate_time.py calibrate --config configs/wave3_collective.yaml

# 4. Post-process: fix OCR errors using monotonicity + interpolation
python tools/calibrate_time.py postprocess --config configs/wave3_collective.yaml

# 5. Verify accuracy (OCR random frames, compare with interpolation)
python tools/calibrate_time.py verify --config configs/wave3_collective.yaml
```

**Calibrate flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | (required) | YAML config file |
| `--interval` | `1000` | OCR every N frames (lower = more accurate, more API calls) |
| `--force` | `false` | Re-calibrate videos that already have data |

**Calibrate workflow:**

1. **Probe** all videos → estimate total API calls
2. **Cost estimate box** — shows per-video call counts, estimated USD cost
3. **First confirmation** — user approves the plan
4. **Auto-detect format** — if `time_format` is `"auto"` (default), tries common formats against the OCR text. Strips timezone indicators (`EST`, `UTC-5`, etc.) automatically.
5. **Preview** — OCR 3 frames from the first video, show parsed results
6. **Second confirmation** — user verifies the preview looks correct
7. **Full run** — OCR all videos with progress bar, save after each video (crash-safe)
8. **Midnight crossing** — auto-detected when clock jumps backward > 12h

**Postprocess workflow:**

After `calibrate` runs, some OCR samples may fail (garbled text, missing seconds, API errors). The `postprocess` subcommand fixes these:

1. **Missing seconds** — `12:04 AM` → `12:04:00 AM` (re-parsed with `:00` appended)
2. **Monotonicity filter** — removes samples where time goes backward (impossible)
3. **Gap interpolation** — fills removed/failed samples using linear interpolation between valid neighbors
4. **Statistics report** — shows before/after counts, error categories, per-video recovery rate

```bash
python tools/calibrate_time.py postprocess --config configs/wave2_collective.yaml
# Saves cleaned data back to time_calibration.json (original backed up as .json.bak)
```

**Verify flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--n` | `5` | Random frames to check per video |

Verify reports per-check delta (Δ seconds between OCR and interpolation), and
prints PASS if ≥95% of checks are within 2 seconds.

**Using the calibration in code:**

```python
from avistrack.core.time_lookup import TimeLookup

tl = TimeLookup.load("time_calibration.json", "Day12_RGB.mkv")

# Frame → time
tl.frame_to_datetime(5000)       # timezone-aware datetime
tl.frame_to_unix(5000)           # float Unix timestamp
tl.frame_to_timestr(5000)        # "2025-10-30 00:02:47"

# Time → frame
tl.unix_to_frame(1730264567.0)   # 5000
tl.datetime_to_frame(some_dt)    # nearest frame

# Diagnostics
tl.actual_fps(0, 3000)           # effective FPS between two frames
tl.duration_seconds              # total calibrated span
```

---

## 9. Time Calibration

Many lab video systems have unstable FPS — the nominal 30 fps can drift
significantly. However, most cameras burn a human-readable timestamp into
the video frame (typically in the upper-right corner). AvisTrack uses sparse
OCR to read these timestamps and build a frame↔time mapping.

### How it works

```
Frame indices:     0        1000      2000      3000      ...
                   │         │         │         │
     OCR →        12:00:00  12:00:34  12:01:07  12:01:40  ...
                   │         │         │         │
                   └─────────┼─────────┼─────────┘
                         piecewise-linear interpolation
                              (numpy.interp)
```

1. **`roi`** — for each video, you draw a rectangle around the timestamp text
   (previous video's region carries forward as default, just like `pick_rois.py`)
2. **`calibrate`** — the tool OCRs every N-th frame (default 1000), parsing
   the text into a datetime. The timestamp format is auto-detected from the
   OCR text (supports `08:35:30 PM EST 18-Jun-25`, `20:35:30 UTC-5`, etc.).
   These sparse anchor points are saved to `time_calibration.json`.
3. **`postprocess`** — fixes OCR errors (missing seconds, garbled digits, API
   failures) using monotonicity constraints and linear interpolation.
4. **`TimeLookup`** uses `numpy.interp` to interpolate between anchors,
   giving sub-second accuracy for any frame in between.

### Data files

| File | Location | Content |
|------|----------|---------|
| `ocr_roi.json` | `02_Global_Metadata/` | `{"video_name": [x, y, w, h], ...}` — per-video crop region for the timestamp |
| `time_calibration.json` | `02_Global_Metadata/` | Per-video arrays of `{frame, unix, ocr_raw, time_local}` samples |

### Google Cloud Vision setup

1. Create a Google Cloud project and enable the **Cloud Vision API**
2. Create a **service account key** (JSON) at [Cloud Console](https://console.cloud.google.com/apis/credentials)
3. Set the environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
   ```
4. Pricing: **1,000 free** TEXT_DETECTION calls/month, then $1.50 / 1,000

---

## 10. Dataset & Sampling Workflow

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
│   ├── ocr_roi.json               {video_name: [x, y, w, h]}  (timestamp crop per video)
│   ├── time_calibration.json      frame↔unix mapping per video
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

### Typical iteration cycle

```
┌─────────────────────────────────┐
│  1. Create config               │  python tools/init_config.py
│     (or copy template.yaml)     │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  2. Pick ROIs                   │  python tools/pick_rois.py --config ...
│     + Validate coverage         │  python tools/pick_rois.py validate --config ...
│     (skip if no transform)      │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  3. Calibrate frame↔time       │  python tools/calibrate_time.py roi ...
│     (if burn-in timestamps)    │  python tools/calibrate_time.py calibrate ...
│     + Post-process OCR errors   │  python tools/calibrate_time.py postprocess ...
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  4. Sample train clips          │  python tools/sample_clips.py --split train ...
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  5. Label clips in CVAT         │  (external tool)
│     + Train YOLO                │  yolo detect train ...
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  6. Sample golden test clips    │  python tools/sample_clips.py --split test ...
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  7. Label test clips GT         │  (manual annotation in MOT format)
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  8. Evaluate model              │  python eval/run_eval.py ...
│     Not good enough? → go to 4  │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  9. Batch process all videos    │  python cli/run_batch.py --config ...
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  9. Analyze with CCM            │  (Collective-Chamber-Metrics repo)
└─────────────────────────────────┘
```

---

## 11. Offline Batch Processing

Processes **all** raw videos with a single command. Supports:
- Automatic resume (skips files whose output already exists)
- Configurable multiprocessing (`--workers`)
- Per-video progress logging

```bash
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

## 12. Model Evaluation

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

---

## 13. Real-time Integration (ChamberBroadcaster)

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

## 14. Running Tests

```bash
conda activate avistrack
cd /path/to/AvisTrack
python -m pytest tests/ -v
```

```
tests/test_config.py::test_load_minimal_config              PASSED
tests/test_config.py::test_root_placeholder_resolution      PASSED
tests/test_time_lookup.py::TestInterpolation::...           PASSED  (x3)
tests/test_time_lookup.py::TestReverse::...                 PASSED  (x2)
tests/test_time_lookup.py::TestDatetime::...                PASSED  (x4)
tests/test_time_lookup.py::TestProperties::...              PASSED  (x4)
tests/test_time_lookup.py::TestConstruction::...            PASSED  (x6)
tests/test_transformer.py::test_transform_output_size_fixed PASSED
tests/test_transformer.py::test_transform_output_size_auto  PASSED
tests/test_transformer.py::test_from_roi_file               PASSED
tests/test_transformer.py::test_from_roi_file_missing_key   PASSED

25 passed in 0.25s
```

---

## 15. Common Recipes

### Sample 20 golden-test clips with 600 frames each

At 30 fps, 600 frames = 20 seconds:

```bash
python tools/sample_clips.py \
    --config     configs/wave2_collective.yaml \
    --split      test \
    --n          20 \
    --duration   20 \
    --output-dir /media/woodlab/104-A/Wave2/01_Dataset_MOT_Format/test_golden
```

### Set up a brand-new wave from scratch

```bash
# 1. Create config interactively
python tools/init_config.py --root /media/woodlab/NEW_DRIVE/Wave4

# 2. Pick ROI corners for all videos
python tools/pick_rois.py --config configs/wave4_collective.yaml

# 2b. Validate ROIs
python tools/pick_rois.py validate --config configs/wave4_collective.yaml

# 3. Calibrate frame↔time mapping (if burn-in timestamps exist)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
python tools/calibrate_time.py roi --config configs/wave4_collective.yaml
python tools/calibrate_time.py calibrate --config configs/wave4_collective.yaml

# 4. Sample training clips
python tools/sample_clips.py \
    --config configs/wave4_collective.yaml --split train --n 30 --duration 3 \
    --output-dir /media/woodlab/NEW_DRIVE/Wave4/01_Dataset_MOT_Format/train

# 5. (Label in CVAT → Train YOLO → put best.pt on drive)

# 6. Sample test clips (600 frames = 20 sec @ 30fps)
python tools/sample_clips.py \
    --config configs/wave4_collective.yaml --split test --n 20 --duration 20 \
    --output-dir /media/woodlab/NEW_DRIVE/Wave4/01_Dataset_MOT_Format/test_golden

# 7. Evaluate
python eval/run_eval.py --config configs/wave4_collective.yaml \
    --weights /media/.../best.pt --output eval/reports/wave4.csv

# 8. Batch process everything
python cli/run_batch.py --config configs/wave4_collective.yaml
```

### Compare two model versions

```bash
python eval/run_eval.py \
    --config  configs/wave3_collective.yaml \
    --weights \
        /media/.../Train_v1_best/weights/best.pt \
        /media/.../Train_v2_best/weights/best.pt \
    --output  eval/reports/comparison.csv
```

### Re-sample more training data after eval shows low recall

```bash
# This will NOT overlap with any existing train/val/test clips
python tools/sample_clips.py \
    --config     configs/wave3_collective.yaml \
    --split      train \
    --n          20 \
    --duration   3 \
    --seed       123 \
    --output-dir /media/woodlab/104-A/Wave3/01_Dataset_MOT_Format/train
```
