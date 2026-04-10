# W2 Collective Chamber — YOLO Model Selection Report

## Overview

This report summarizes the process of selecting the best YOLO detection model for
the Wave 2 Collective Chamber experiment. The task is single-class chick detection
(9 White Leghorn chicks per session) from a fixed overhead 640×640 camera, used as
input to a downstream multi-animal tracker.

---

## Phase 0 — Data Curation

Raw annotations were in MOT format across 50 clips (W2) and one merged clip (W3).
Many W2 clips contained "CollectiveDanger" stimuli where chicks freeze — causing
long runs of near-identical frames that would skew training.

**Solution:** IoU-based frame deduplication. A frame is kept only when the minimum
bounding-box IoU against the last kept frame falls below a threshold.

| Threshold | W2 frames kept | W3 frames kept |
|-----------|---------------|---------------|
| 0.92 | 694 | 1743 |
| 0.94 | 843 | 1743 |
| **0.96** | **1109** | **1743** ← chosen |
| 0.98 | 1663 | 1743 |

W2 used threshold **0.96** (higher, to preserve more frames from freeze-heavy clips);
W3 used **0.92** (standard). Final curated sets: **W2 = 1495 frames**, **W3 ≈ 1743 frames**.

---

## Phase 1 — Architecture Scan

Six architectures were trained to convergence (patience = 20, max 300 epochs) on
W2-only data using a single RTX 3070.

| Architecture | Params | mAP50 | mAP50-95 | Advance |
|-------------|--------|-------|----------|---------|
| `yolo11s` | 9.4 M | 0.938 | **0.549** | ✓ |
| `yolo11n` | 2.6 M | 0.945 | 0.543 | ✓ |
| `yolo26s` | 7.2 M | **0.949** | 0.539 | ✓ |
| `yolo26n` | 2.2 M | 0.944 | 0.529 | — |
| `yolo8n` | 3.2 M | 0.924 | 0.536 | — |
| `yolo8s` | 11 M | 0.941 | 0.533 | — |

Top 3 by mAP50-95 (`yolo11s`, `yolo11n`, `yolo26s`) advanced to Phase 1.5.

---

## Phase 1.5 — Architecture Verification on Combined Data

**Concern:** rankings on W2-only data might reverse on a larger, more diverse dataset
(larger models can underfit small datasets but excel with more data).

All three candidates were retrained on W2 + W3 combined training data; validation
remained W2-only for a fair comparison.

| Architecture | mAP50-95 | Result |
|-------------|----------|--------|
| `yolo11s` | **0.549** | ✓ still #1 |
| `yolo26s` | 0.532 | — |
| `yolo11n` | 0.527 | — |

Ranking unchanged. **`yolo11s` confirmed as the architecture** going forward.

---

## Phase 2 — Training Strategy

**Question:** fine-tune from W3 weights, or train from scratch on W2 data?

| Strategy | Init weights | Data | mAP50-95 | F1 | Result |
|----------|-------------|------|----------|-----|--------|
| `w2_scratch` | COCO | W2 only | **0.553** | **0.911** | ✓ winner |
| `combined_scratch` | COCO | W2 + W3 | 0.538 | 0.866 | — |
| `combined_finetune` | W3 best.pt | W2 + W3 | 0.527 | 0.899 | — |
| `w3_finetune_full` | W3 best.pt | W2 only | 0.515 | 0.881 | — |
| `w3_finetune_freeze` | W3 best.pt | W2 only | 0.488 | 0.844 | — |

**Finding:** every W3-initialized variant underperformed. Root cause: W2 birds are
White Leghorn (white/pale); W3 birds are darker. The colour domain gap is large enough
that W3 pretraining actively hurts W2 performance. W2-only scratch training is the
clear winner.

**Ceiling observed:** mAP50-95 ≈ 0.55, attributed to low data diversity — many
freeze-clip frames are near-identical even after IoU filtering.

---

## Dataset Expansion (between Phase 2 and Phase 3)

To break the 0.55 ceiling, 22 additional W2 clips from `val_tuning/annotations` were
added to the training pool. The IoU threshold was also lowered from 0.96 → **0.94**
to reduce freeze-frame duplicates. W3 data was excluded entirely.

| Dataset | Clips | IoU threshold | Train frames | Val frames |
|---------|-------|--------------|-------------|-----------|
| Old `W2_iou096_v1` | 50 | 0.96 | 1198 | 297 |
| **New `W2_combined_iou094_v1`** | **72** | **0.94** | **1637** | **381** |

---

## Phase 3 — Augmentation Tuning

Fixed: `yolo11s` + COCO weights + `W2_combined_iou094_v1` dataset.

| Profile | Epochs | mAP50 | mAP50-95 | F1 | Result |
|---------|--------|-------|----------|----|--------|
| `aug_minimal` | 77 | **0.966** | 0.567 | **0.930** | ✓ best F1 |
| `aug_default` | 76 | 0.953 | **0.576** | 0.918 | ✓ best mAP50-95 |
| `aug_adamw` | 73 | 0.956 | 0.563 | 0.915 | — |
| `aug_sgd_cos` | 48 | 0.949 | 0.543 | 0.911 | early stop |
| `aug_overhead` | 38 | 0.954 | 0.541 | 0.910 | early stop |
| `aug_heavy` | 44 | 0.948 | 0.541 | 0.900 | early stop |

**Finding:** augmentation profiles designed specifically for overhead cameras
(large rotations, scale jitter, vertical flips) **all early-stopped** at 38–48 epochs,
while default/minimal ran to 73–77. Scale augmentation is particularly harmful for
small overhead targets. The best strategy is no augmentation or minimal rotation
with conservative colour jitter.

Phase 2 → Phase 3 gain: **mAP50-95: 0.553 → 0.576 (+0.023)** from data expansion.

---

## Phase 3.5 — Model Capacity Scaling

**Question:** is the ~0.57 plateau caused by insufficient model capacity?

| Model | Params | mAP50-95 |
|-------|--------|----------|
| `yolo11s` | 9.4 M | 0.567 |
| `yolo11m` | 20.1 M | 0.567 |
| `yolo11l` | 25.3 M | 0.569 |

Scaling from 9.4 M to 25.3 M parameters yields **zero improvement**.
Model capacity is not the bottleneck.

**Root cause of the ceiling:** MOT tracking labels are not hand-drawn tight boxes.
At high IoU thresholds (0.75, 0.85, 0.95) the annotations themselves are imprecise,
creating a hard ceiling that no model can overcome. mAP50-95 = 0.75 is not
achievable with the current annotation quality.

---

## Final Model

**`aug_minimal` — yolo11s (COCO init, W2-only scratch)**

| Metric | Value |
|--------|-------|
| Precision | 0.961 |
| Recall | 0.900 |
| F1 | **0.930** |
| mAP50 | **0.966** |
| mAP50-95 | 0.567 |

**Weights:** `E:\Wave2\03_Model_Training\W2_Collective\Phase3_Augmentation\aug_minimal\weights\best.pt`

Selected because it has the highest Precision (fewest ghost detections → cleaner
downstream tracks), highest F1, and the smallest/fastest footprint of the top
candidates. The mAP50 of 0.966 is excellent for IoU ≥ 0.5 tracking matching.

---

## Decision Path Summary

```
Phase 1  — Architecture scan (6 models)
              → yolo11s leads on mAP50-95

Phase 1.5 — Verify ranking on combined data
              → ranking unchanged; yolo11s confirmed

Phase 2  — Training strategy (5 variants)
              → W2 scratch wins; W3 transfer hurts (colour domain gap)
              → ceiling: mAP50-95 ≈ 0.55 (data diversity limited)

Data expansion — 50 clips → 72 clips; IoU 0.96 → 0.94

Phase 3  — Augmentation tuning (6 profiles)
              → overhead-specific augmentation hurts; minimal is best
              → ceiling rises to mAP50-95 ≈ 0.576

Phase 3.5 — Capacity scaling (9 M → 25 M params)
              → zero improvement; annotation precision is the hard ceiling

Final    — yolo11s + aug_minimal  (P=0.961, R=0.900, F1=0.930)
```

---

## Next Step — Phase 4: Tracking Evaluation

The selected model feeds four tracking algorithms evaluated on a held-out test set:

| Method | Description |
|--------|-------------|
| `top9_hungarian` | Top-9 detections by confidence + Hungarian IoU matching |
| `top9_kalman` | Top-9 + Kalman filter for occlusion handling |
| `top9_interp` | Hungarian baseline + linear gap interpolation (≤ 10 frames) |
| `bytetrack` | Ultralytics built-in ByteTrack (industry reference) |

Primary metric: **HOTA** (balances detection and association accuracy).
