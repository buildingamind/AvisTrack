# Collective Chamber W2 — YOLO Model Training Experiment Plan

## Objective

Train and select the best YOLO detection model for the **Wave 2 Collective Chamber** experiment.
- **Subject**: 9 White Leghorn chicks (lighter coloration than W3 Rhode Island Red mix)
- **Camera**: Fixed overhead, 640×640 after ROI perspective correction
- **Task**: Single-class detection (`chick`, class 0) for downstream multi-animal tracking

---

## Available Resources

| Resource | Details |
|----------|---------|
| W2 raw annotations | `E:\Wave2\01_Dataset_MOT_Format\train\annotations\` — 50 clips, read-only |
| W3 raw annotations | `E:\Wave3\01_Dataset_MOT_Format\train\clip_merge_train\` — 1 merged clip, 6265 frames, read-only |
| W2 curated dataset | `E:\Wave2\03_Model_Training\Datasets\YOLO\W2_iou096_v1\` — 1495 frames (1198 train / 297 val) |
| W3 curated dataset | `E:\Wave3\03_Model_Training\Datasets\YOLO\W3_iou092_v1\` — 2950 frames (2508 train / 442 val) |
| W3 trained weights | `E:\Wave3\03_Model_Training\Yolo11n\Train_v1_best\weights\best.pt` |
| Local GPU | RTX 3070 (8 GB VRAM) — Phase 1 architecture scan and iteration |
| Remote GPU | 8× A10 — Phase 2/3 full-length runs |
| n_subjects | 9 (both W2 and W3) |
| Breed difference | W2 = White Leghorn (white/pale), W3 = different breed (darker) — color domain gap |

---

## Dataset Directory Layout

All curated datasets live under `03_Model_Training\Datasets\` — the raw annotation
folders (`01_Dataset_MOT_Format\`) are **never modified**.

```
E:\Wave2\03_Model_Training\
  Datasets\
    YOLO\
      W2_iou096_v1\            <- W2 train only, IoU=0.96 [Phase 1/2 baseline, superseded]
        images\train\  (1198 .jpg)
        images\val\    (297  .jpg)
        data.yaml
        curation_report.json
      W2W3_combined_iou096_092_v1\   <- no images; data.yaml only [superseded]
        data.yaml
      W2_iou094_v1\            <- W2 train/annotations, IoU=0.94 [Phase 3+]
        images\train\  (983  .jpg)
        images\val\    (275  .jpg)
        data.yaml
        curation_report.json
      W2_vt_iou094_v1\         <- W2 val_tuning/annotations, IoU=0.94 [Phase 3+]
        images\train\  (654  .jpg)
        images\val\    (106  .jpg)
        data.yaml
        curation_report.json
      W2_combined_iou094_v1\   <- combined W2 (72 clips), IoU=0.94 [Phase 3+]
        data.yaml               <- points to both W2_iou094_v1 + W2_vt_iou094_v1
    COCO\                      <- reserved for future format
    MOT_filtered\              <- reserved for future format
  W2_Collective\               <- training run outputs (Phases 1-4)

E:\Wave3\03_Model_Training\
  Datasets\
    YOLO\
      W3_iou092_v1\            <- W3 only, IoU filter=0.92
        images\train\  (~1481 .jpg)
        images\val\    (~262  .jpg)
        labels\train\
        labels\val\
        data.yaml
        curation_report.json
```

### Versioning convention
`{source}_iou{threshold*100}_v{N}` — increment `vN` when re-curating with a new
threshold or seed. Old versions are kept for reproducibility.

### Combined data.yaml
```yaml
# E:\Wave2\03_Model_Training\Datasets\YOLO\W2W3_combined_iou096_092_v1\data.yaml
train:
  - E:/Wave2/03_Model_Training/Datasets/YOLO/W2_iou096_v1/images/train
  - E:/Wave3/03_Model_Training/Datasets/YOLO/W3_iou092_v1/images/train
val:
  - E:/Wave2/03_Model_Training/Datasets/YOLO/W2_iou096_v1/images/val
  - E:/Wave3/03_Model_Training/Datasets/YOLO/W3_iou092_v1/images/val
nc: 1
names: ['chick']
```

---

## Raw Data Layout (MOT format — read-only)

Both W2 and W3 use **MOT directory format** (not flat `.mp4 + .txt` pairs).
Frames are pre-extracted to `img1/`; no video decode is needed.
Frame index convention: gt.txt is **1-based**, filenames are **0-based**
→ `frame 1 in gt.txt = frame_000000.png`, `frame N = frame_{N-1:06d}.png`.

### W2 — multi-clip layout (50 clips)

```
E:\Wave2\01_Dataset_MOT_Format\train\annotations\
  <clip_name>\
    gt\gt.txt        MOT labels (frame,id,x,y,w,h,conf,class,vis)
    img1\frame_XXXXXX.png
```

Clips: 50 total — mix of 45-frame and 90-frame clips.
Many are "CollectiveDanger" clips (threat stimulus); several are nearly static
(chicks freeze). Those contribute only 1 frame each after IoU filtering.

### W3 — single merged-clip layout

```
E:\Wave3\01_Dataset_MOT_Format\train\clip_merge_train\
  gt\gt.txt     frames 1-6265, ~55 490 rows (~8.9 animals/frame)
  img1\frame_XXXXXX.png    6265 frames total
```

### MOT gt.txt column layout

```
frame, id, x, y, w, h, conf, class, vis
```
- `x, y` = top-left corner in pixels
- `class` = `1` in gt.txt (MOT convention) → exported as `0` in YOLO labels (single class)

---

## Phase 0: Data Curation  ✓ DONE

**Tool**: `tools/curate_frames.py`

### Filter method: IoU (recommended)

Keep frame when `min IoU across shared tracks vs last kept frame < iou_threshold`.
- IoU = 1.0 → box identical (skip); IoU = 0.0 → no overlap (definitely keep)
- W2 uses **0.96** (higher threshold → more frames from the white Leghorn breed)
- W3 uses **0.92** (standard threshold)
- W2 has many static "danger" clips (chicks freeze); those contribute 1 frame each

### IoU threshold selection (from preview)

```
Threshold    W2 kept    W3 kept    Total    Train~85%
  0.92           694       1743     2437        2071
  0.94           843       1743     2586        2198
  0.96          1109       1743     2852        2424  <- chosen
  0.98          1663       1743     3406        2895
```

W2 IoU distribution: Min=0.000  P25=0.927  Median=0.973  P95=1.000
W3 IoU distribution: Min=0.000  P25=0.913  Median=0.957  P95=0.994

### Export commands (already run)

```bash
PYTHON=/c/Users/OpenChickStudio/miniconda3/envs/AvisTrack/python.exe

# W2 — iou=0.96, clip-level split (50 clips → 43 train / 7 val)
$PYTHON tools/curate_frames.py export \
    --clips-dir "E:/Wave2/01_Dataset_MOT_Format/train/annotations" \
    --output-dir "E:/Wave2/03_Model_Training/Datasets/YOLO/W2_iou096_v1" \
    --filter-method iou --iou-threshold 0.96 \
    --val-split 0.15 --seed 42
# Result: 1495 total (1198 train / 297 val)

# W3 — iou=0.92, frame-level split (single clip → frame shuffle)
$PYTHON tools/curate_frames.py export \
    --clips-dir "E:/Wave3/01_Dataset_MOT_Format/train/clip_merge_train" \
    --output-dir "E:/Wave3/03_Model_Training/Datasets/YOLO/W3_iou092_v1" \
    --filter-method iou --iou-threshold 0.92 \
    --val-split 0.15 --seed 42
# Result: ~1743 total (~1481 train / ~262 val)
```

> Note: single-clip sources (W3) use automatic frame-level split instead of
> clip-level split to avoid all frames landing in val.

### Preview command (re-run anytime)

```bash
$PYTHON tools/curate_frames.py preview \
    --clips-dir "E:/Wave2/01_Dataset_MOT_Format/train/annotations" \
                "E:/Wave3/01_Dataset_MOT_Format/train/clip_merge_train" \
    --labels W2 W3
```

---

## Phase 1: Architecture Scan

**Goal**: Run 6 architectures to convergence; pick top-3 by mAP50-95 for Phase 1.5.
**Status**: ✓ DONE

| Candidate | Weights | Params | Epochs | Best mAP50 | Best mAP50-95 | → Phase 1.5 |
|-----------|---------|--------|--------|-----------|--------------|------------|
| `yolo11s` | yolo11s.pt | ~9.4M  | 44 | 0.938 | **0.549** | ✓ #1 |
| `yolo11n` | yolo11n.pt | ~2.6M  | 50 | 0.945 | **0.543** | ✓ #2 |
| `yolo26s` | yolo26s.pt | ~7.2M  | 38 | **0.949** | 0.539 | ✓ #3 (highest mAP50) |
| `yolo26n` | yolo26n.pt | ~2.2M  | 52 | 0.944 | 0.529 | — |
| `yolo8n`  | yolov8n.pt | ~3.2M  | 31 | 0.924 | 0.536 | — |
| `yolo8s`  | yolov8s.pt | ~11M   | 33 | 0.941 | 0.533 | — |

All 6 run **sequentially and automatically** — one finishes, next starts.
Each runs to convergence (patience=20, max 300 epochs).

**Settings**:
```
epochs:   300   # patience=20 stops early; typically 100-150 epochs
imgsz:    640
batch:    16
device:   0     # RTX 3070
patience: 20
data:     E:/Wave2/03_Model_Training/Datasets/YOLO/W2_iou096_v1/data.yaml
project:  E:/Wave2/03_Model_Training/W2_Collective/Phase1_ArchScan
```

**Decision rule**: Pick top-3 by mAP50-95 on val; carry all into Phase 1.5.
`run_pipeline.py` reads results automatically and patches Phase 2 configs.

**Output**: `E:\Wave2\03_Model_Training\W2_Collective\Phase1_ArchScan\`

---

## Phase 1.5: Architecture Verification on Combined Data

**Goal**: Confirm Phase 1 ranking holds when W3 data is added. Architecture rankings
can shift with dataset size/diversity — a larger model may underfit W2-only but excel
on combined data.

**Status**: ✓ DONE — ranking unchanged, yolo11s confirmed as winner

**Inputs**: top-3 from Phase 1 (W2-only ranking) — yolo11s, yolo11n, yolo26s

**Settings**:
```
epochs:   300   # patience=20 stops early
imgsz:    640
batch:    16
device:   0
patience: 20
data:     E:/Wave2/03_Model_Training/Datasets/YOLO/W2W3_combined_iou096_092_v1/data.yaml
            └─ train: W2 (1198) + W3 (1481 frames)
            └─ val:   W2 only (297 frames)  ← W2-only eval for apples-to-apples comparison
project:  E:/Wave2/03_Model_Training/W2_Collective/Phase1_5_ArchVerify
```

| Candidate | Best mAP50 | Best mAP50-95 | Best Epoch | → Phase 2 |
|-----------|-----------|--------------|------------|-----------|
| `yolo11s` | 0.941 | **0.549** | 10 | ✓ winner |
| `yolo26s` | 0.940 | 0.532 | 6 | — |
| `yolo11n` | 0.937 | 0.527 | 17 | — |

Ranking unchanged from Phase 1 → **yolo11s confirmed for Phase 2**.

**Decision rule**:
- If ranking is unchanged → use Phase 1 winner for Phase 2 (confident choice)
- If ranking flips → use the combined-data winner for Phase 2

**Output**: `E:\Wave2\03_Model_Training\W2_Collective\Phase1_5_ArchVerify\`

---

## Phase 2: Training Strategy

**Status**: ✓ DONE — w2_scratch is the clear winner

**Goal**: Compare scratch vs. fine-tune and data mixing using the Phase 1 winner.

| Candidate | Model Init | Data | Best mAP50-95 | F1 | Decision |
|-----------|-----------|------|-------------|-----|---------|
| `w2_scratch` | COCO weights | W2_only | **0.553** | **0.911** | ✓ Winner |
| `combined_finetune` | W3 best.pt | W2W3_combined | 0.527 | 0.899 | — |
| `combined_scratch` | COCO weights | W2W3_combined | 0.538 | 0.866 | — |
| `w3_finetune_full` | W3 best.pt | W2_only | 0.515 | 0.881 | — |
| `w3_finetune_freeze` | W3 best.pt | W2_only | 0.488 | 0.844 | — |

**Key finding**: W3 transfer learning hurt performance in all variants. Root cause: W2 = White
Leghorn (white/pale); W3 = different breed (darker) — colour domain gap is too large. W2-only
scratch training consistently outperforms any W3-initialised approach.

**Phase 2 hit a ceiling at mAP50-95 ≈ 0.55.** Root cause: insufficient training data diversity
(1198 frames from 50 clips, many "CollectiveDanger" freeze clips with near-identical frames).

**Settings**:
```
epochs:   100
imgsz:    640
batch:    16
patience: 20
data:     (per candidate above)
project:  E:/Wave2/03_Model_Training/W2_Collective/Phase2_Strategy
```

**Output**: `E:\Wave2\03_Model_Training\W2_Collective\Phase2_Strategy\`

---

## Dataset Expansion (between Phase 2 and Phase 3)

**Status**: ✓ DONE

**Motivation**: Phase 2 plateau at mAP50-95=0.55 was attributed to limited data diversity.
`val_tuning/annotations` (22 additional W2 clips, 1440 raw frames) was added to the training pool.
IoU threshold lowered from 0.96 → 0.94 to reduce near-duplicate frames from freeze clips.
W3 data deliberately excluded (colour domain gap confirmed harmful in Phase 2).

| Dataset | Clips | IoU threshold | Train frames | Val frames |
|---------|-------|--------------|-------------|-----------|
| `W2_iou096_v1` (old) | 50 | 0.96 | 1198 | 297 |
| `W2_iou094_v1` (new) | 50 | 0.94 | 983 | 275 |
| `W2_vt_iou094_v1` (new) | 22 | 0.94 | 654 | 106 |
| **`W2_combined_iou094_v1`** (combined) | **72** | **0.94** | **1637** | **381** |

**Export commands** (already run):
```bash
PYTHON=/c/Users/OpenChickStudio/miniconda3/envs/AvisTrack/python.exe

PYTHONIOENCODING=utf-8 $PYTHON tools/curate_frames.py export \
    --clips-dir "E:/Wave2/01_Dataset_MOT_Format/train/annotations" \
    --output-dir "E:/Wave2/03_Model_Training/Datasets/YOLO/W2_iou094_v1" \
    --filter-method iou --iou-threshold 0.94 --val-split 0.15 --seed 42
# Result: 983 train / 275 val

PYTHONIOENCODING=utf-8 $PYTHON tools/curate_frames.py export \
    --clips-dir "E:/Wave2/01_Dataset_MOT_Format/val_tuning/annotations" \
    --output-dir "E:/Wave2/03_Model_Training/Datasets/YOLO/W2_vt_iou094_v1" \
    --filter-method iou --iou-threshold 0.94 --val-split 0.15 --seed 42
# Result: 654 train / 106 val
```

Combined data.yaml: `E:\Wave2\03_Model_Training\Datasets\YOLO\W2_combined_iou094_v1\data.yaml`

---

## Phase 3: Augmentation Tuning

**Status**: ✓ DONE — aug_default and aug_minimal are the winners

**Goal**: Find the best augmentation profile for fixed overhead camera with 9 chicks.
**Model**: yolo11s.pt (COCO weights, w2_scratch strategy)
**Data**: `W2_combined_iou094_v1` (1637 train / 381 val — 72 clips, IoU=0.94)

| Candidate | Total Epochs | Best Epoch | mAP50 | mAP50-95 | F1 | Decision |
|-----------|-------------|-----------|-------|----------|----|---------|
| `aug_minimal` | 77 | 57 | **0.966** | 0.567 | **0.930** | ✓ Winner (F1/mAP50) |
| `aug_default` | 76 | 56 | 0.953 | **0.576** | 0.918 | ✓ Winner (mAP50-95) |
| `aug_adamw` | 73 | 53 | 0.956 | 0.563 | 0.915 | — |
| `aug_sgd_cos` | 48 | 28 | 0.949 | 0.543 | 0.911 | early stop |
| `aug_overhead` | 38 | 18 | 0.954 | 0.541 | 0.910 | early stop |
| `aug_heavy` | 44 | 24 | 0.948 | 0.541 | 0.900 | early stop |

**Key findings**:
- Overhead-specific augmentation (flipud+degrees+scale) **hurt** performance — all three variants
  early-stopped at 38-48 epochs vs 73-77 for default/minimal. Scale augmentation likely harmful
  for small overhead targets.
- Best augmentation is either none (default) or minimal rotation+conservative colour jitter.
- mAP50=0.966 is excellent; the mAP50-95 gap (0.966-0.576=0.39) reflects annotation box
  precision limits from MOT tracking labels, not model detection failure.

**Progress**: Phase 2 best (0.553) → Phase 3 best (0.576) — +0.023 from dataset expansion.
yolo11s capacity appears to be near its ceiling on this dataset.

**Output**: `E:\Wave2\03_Model_Training\W2_Collective\Phase3_Augmentation\`

---

## Phase 3.5: Model Capacity Scaling

**Status**: ✓ DONE — model capacity is NOT the bottleneck; plateau confirmed

**Goal**: Test whether a larger backbone can push mAP50-95 above the yolo11s plateau (~0.576).
**Augmentation**: aug_minimal profile (flipud=0.5, degrees=180, hsv_h=0.02, hsv_s=0.3).
**Data**: `W2_combined_iou094_v1` (unchanged). batch=8 (batch=16 OOM on RTX 3070 8GB).

| Candidate | Params | Best Epoch | Total Epochs | mAP50 | mAP50-95 | F1 |
|-----------|--------|-----------|-------------|-------|----------|----|
| yolo11s aug_minimal (Ph3 ref) | 9.4M | 57 | 77 | 0.966 | 0.567 | 0.930 |
| yolo11m | 20.1M | 33 | 53 | 0.962 | 0.567 | 0.922 |
| yolo11l | 25.3M | 35 | 55 | 0.963 | 0.569 | 0.923 |

**Key finding**: Scaling from 9.4M → 25.3M params yields **zero improvement** in mAP50-95
(0.567 → 0.567 → 0.569, within noise). Model capacity is definitively not the bottleneck.

**Root cause of the ~0.57 ceiling**: MOT tracking annotations are not tightly hand-labelled
boxes. At high IoU thresholds (0.75, 0.85, 0.95) the annotations themselves are imprecise,
creating a hard ceiling that no model size can overcome.

**Conclusion**: mAP50-95=0.75 is **not achievable** with the current annotation quality.
Decision required — see Next Step below.

**Output**: `E:\Wave2\03_Model_Training\W2_Collective\Phase3_5_ModelScale\`

---

## Phase 4: Tracking Method Evaluation

**Status**: READY TO RUN

**Selected model**: `aug_minimal` (yolo11s, Phase 3 winner)
- Path: `E:\Wave2\03_Model_Training\W2_Collective\Phase3_Augmentation\aug_minimal\weights\best.pt`
- Rationale: best Precision (0.961) + Recall (0.900) + F1 (0.930); smallest/fastest

**Decision on mAP50-95=0.75 target**: Not achievable with current MOT tracking annotations
(annotation box precision is the hard ceiling, not model capacity). Proceeding with
mAP50=0.966 model which is excellent for downstream tracking (IoU≥0.5 matching).

### Tracking Methods (all use top-9 selection — 9 animals always present)

| Method | Description |
|--------|-------------|
| `top9_hungarian` | Top-9 by conf + Hungarian IoU matching — clean baseline |
| `top9_kalman` | Top-9 + Kalman constant-velocity prediction + Hungarian — better occlusion handling |
| `top9_interp` | top9_hungarian + linear interpolation for gaps ≤10 frames |
| `bytetrack` | Ultralytics built-in ByteTrack — industry reference |

**Decision rule**: Keep methods where HOTA is competitive; drop underperformers from UI.

### Eval Pipeline (4 scripts)

| Script | Purpose |
|--------|---------|
| `eval/infer_clips.py` | YOLO inference → `detections/<clip>/dets.csv` (shared by all methods) |
| `eval/trackers.py` | Run 4 tracking methods → `tracks/<method>/<clip>/gt.txt` (MOT format) |
| `eval/score.py` | Compute HOTA/IDF1/MOTA → `reports/scores.json` |
| `eval/viewer.py` | PyQt5 interactive comparison UI (GT + Method A + Method B) |

### Run Commands

```bash
cd /c/Users/OpenChickStudio/Documents/GitHub/AvisTrack
PYTHON=/c/Users/OpenChickStudio/miniconda3/envs/AvisTrack/python.exe
WEIGHTS="E:/Wave2/03_Model_Training/W2_Collective/Phase3_Augmentation/aug_minimal/weights/best.pt"
CLIPS="E:/Wave2/01_Dataset_MOT_Format/test_golden"
OUTDIR="E:/Wave2/04_Evaluation/W2_Collective"

# Step 1: YOLO inference (once, reused by all trackers)
PYTHONIOENCODING=utf-8 $PYTHON eval/infer_clips.py \
    --weights "$WEIGHTS" --clips-dir "$CLIPS" --output-dir "$OUTDIR"

# Step 2: All 4 tracking methods
PYTHONIOENCODING=utf-8 $PYTHON eval/trackers.py \
    --detections-dir "$OUTDIR/detections" --clips-dir "$CLIPS" \
    --output-dir "$OUTDIR/tracks" --weights "$WEIGHTS"

# Step 3: Score (HOTA / IDF1 / MOTA)
PYTHONIOENCODING=utf-8 $PYTHON eval/score.py \
    --tracks-dir "$OUTDIR/tracks" --gt-dir "$CLIPS" \
    --output "$OUTDIR/reports/scores.json"

# Step 4: Open comparison viewer
PYTHONIOENCODING=utf-8 $PYTHON eval/viewer.py \
    --clips-dir "$CLIPS" --tracks-dir "$OUTDIR/tracks" \
    --scores "$OUTDIR/reports/scores.json"
```

### Viewer UI Features
- 3 panels: GT (green) | Method A | Method B — methods selectable via dropdown
- Frame slider + play (10 fps) + prev/next clip
- Metrics table: HOTA / IDF1 / MOTA / IDSW per method per clip
- Toggles: Boxes [B] / IDs [I] / Confidence [C] / Trails [T]
- Keyboard: ←/→ frame, ↑/↓ clip, Space play, Ctrl+←/→ jump 10 frames, Q quit

**Output**: `E:\Wave2\04_Evaluation\W2_Collective\`

---

## Key Metrics (Priority Order)

1. **Precision** — false positives cause ghost tracks (highest priority)
2. **Recall** — missed detections cause track fragmentation
3. **F1** — balanced summary
4. **ID-Switches** — tracking stability (measured in `eval/run_eval.py`)
5. **FPS** — must sustain ≥ 15 FPS on RTX 3070 for offline use

Evaluation threshold: IoU ≥ 0.5 (standard COCO).

---

## Output Directory Structure

```
E:\Wave2\03_Model_Training\
  Datasets\
    YOLO\
      W2_iou096_v1\
      W2W3_combined_iou096_092_v1\
    COCO\
    MOT_filtered\
  W2_Collective\
    Phase1_ArchScan\
      yolo8n\  yolo8s\  yolo11n\  yolo11s\  yolo26n\  yolo26s\
    Phase1_5_ArchVerify\
      <arch_A>\  <arch_B>\         <- top-2 from Phase 1 on combined data
    Phase2_Strategy\
      w2_scratch\  w3_finetune_full\  w3_finetune_freeze\
      combined_scratch\  combined_finetune\
    Phase3_Augmentation\
      aug_default\  aug_overhead\  aug_heavy\
      aug_minimal\  aug_adamw\  aug_sgd_cos\
    Phase4_Final\
      summary.csv

E:\Wave3\03_Model_Training\
  Datasets\
    YOLO\
      W3_iou092_v1\
```

---

## Compute Budget Estimate

| Phase | Runs | Epochs | Est. Time (RTX 3070) |
|-------|------|--------|----------------------|
| Phase 1 | 6 | ~100-150 (patience=20) | ~8-12 hr |
| Phase 1.5 | 3 (top-3 archs × combined data) | ~100-150 | ~4-6 hr |
| Phase 2 | 5 | 100 | ~12 hr |
| Phase 3 | 6×2 | 100 | ~14 hr |
| **Total** | | | **~37-42 hr** |

Phases 2 and 3 can be run on the 8×A10 remote cluster (~4× speedup).

---

## Tools

| Tool | Purpose | Status |
|------|---------|--------|
| `tools/curate_frames.py` | Frame dedup + YOLO dataset export (IoU & displacement methods) | Done |
| `train/run_train.py` | Batch training runner (skip completed, dry-run, --eval) | TODO |
| `eval/run_eval.py` | Final test-set evaluation | TODO |

---

## Next Step

**Phase 4 — run the eval pipeline.**
All training is complete. Selected model: `aug_minimal` yolo11s (best P/R/F1).
Run Steps 1–4 from the Phase 4 section above, then review the viewer to assess
whether tracking quality is acceptable for downstream behavioural analysis.

If HOTA for `top9_interp` is not competitive with `top9_kalman` or `bytetrack`,
it can be dropped from the UI (viewer supports any subset via dropdown).
