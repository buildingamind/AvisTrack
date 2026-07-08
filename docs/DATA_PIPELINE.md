# AvisTrack Data Pipeline — from raw chamber video to trained model

This document is the map of how a wave of chamber recordings becomes a
YOLO train/val/test dataset and a selected champion model, plus the
provenance of the tooling itself. Read this first when you come back to the
project after a break.

> **Scope note.** Written 2026-07 during the plus-chamber Wave3→Wave4 work.
> It covers the *workspace-mode* (multi-chamber) pipeline, which is the
> current path. Legacy single-drive `--config` tooling still exists but is
> not described here.

---

## 1. TL;DR — the six stages

```
01a sample_clips ─┐
                  ├─▶ 02 extract_frames ─▶ 03 review_triage ─▶ [CVAT] ─▶ 04 import_annotations ─▶ 05 build_dataset ─▶ 06 train ─▶ 07 eval
01b select_frames ┘        (candidate frames + triage)   (human keep/drop)  (draw boxes)    (batch → annotations/)   (recipe → dataset)  (bakeoff)  (champion)
```

- **Selection** (stage 01) chooses *which frames* get annotated. Two
  interchangeable methods feed the identical downstream:
  - `01a` random clip sampling (blind, coverage).
  - `01b` model-signal frame selection (targeted: novelty / hard cases /
    false positives, computed from the previous model's per-frame output).
- **`01b` enters the flow at stage 03** — it produces triage-ready frames
  directly, so it skips `02`.
- Everything from **stage 03 onward is identical** regardless of how frames
  were chosen.

---

## 2. Two-layer storage

| Layer | What lives there | Example |
|---|---|---|
| **Workspace SSD** (a.k.a. the T7) | central artifacts: `clips/ frames/ annotations/ manifests/ datasets/ models/` | `F:\BAM\avistrack_workspace\plus\` (drive label `T7-Lab`) |
| **Chamber drive(s)** | raw videos + per-wave metadata | `E:\...\Wave4_PairedCapacityPlus\` (drive label `103-A`) |

The workspace is resolved from `workspace.yaml` (path templates) +
`sources.yaml` (which chamber drive holds which wave, matched by drive
UUID). The two layers can be on different physical drives, and a chamber
drive can be re-mounted under any letter on any machine — provenance keys
on the drive UUID, not the path.

> The plus workspace was originally built on a Linux box where the T7 was
> mounted at `/media/woodlab/T7-Lab/...`; the same T7 is now `F:` on this
> Windows machine. The data is identical; only drive letters differ.

---

## 3. The numbered pipeline (tools/)

Files that make up one wave's annotation→training loop are numbered by
stage. When a stage has interchangeable methods, they share the number and
differ by a letter suffix (`01a`, `01b`).

| File | Stage | What it does |
|---|---|---|
| `00_status.py` | status | Read-only dashboard: clips / annotations / datasets / models per chamber×wave. Writes nothing. |
| `01a_sample_clips.py` | select | Weighted-random 3 s clips from raw videos; perspective-warps to 640×640; appends `all_clips.csv`. |
| `01b_select_frames.py` | select | Model-signal frame selection (workspace-mode). Reads the previous model's tracking parquets, keeps only **valid-range** frames (mandatory; see `avistrack/valid_ranges.py`), buckets them (dup / lowconf / miss / coverage; plus an optional **novelty** bucket = embedding-distance vs a reference set, which decodes+embeds a shortlist and needs `--ref-emb` + `--yolo-weights`). **Phase 1 (read-only, except novelty):** applies your per-bucket quotas + temporal de-dup, writes a plan CSV. **Phase 2 (`--materialize`):** decode → warp 640×640 → register one virtual clip per source session in all_clips.csv → flat frames + triage manifest for 03_review_triage. |
| `02_extract_frames.py` | triage-prep | Even-spaced frames per clip + dHash dedup → candidate PNGs + `manifests/triage/{batch}.csv`. |
| `03_review_triage.py` | triage | Interactive keep/drop of candidate frames; rejects move to `_rejected/`. |
| `04_import_annotations.py` | import | CVAT project zip → flat `annotations/{batch_id}/` + `annotation_batches.csv`. |
| `05_build_dataset.py` | build | Recipe → immutable `datasets/{name}/` view (images/labels + manifest + frozen recipe). |
| `06_train.py` | train | *(wraps `train/run_pipeline.py`)* multi-model bakeoff + lineage snapshot. |
| `07_eval.py` | eval | *(wraps `eval/run_eval.py`)* evaluate on test split, select champion. |

> **Naming constraint.** Python module names cannot start with a digit, so
> `from tools.04_import_annotations import parse_frame_name` is a syntax
> error. Numbered files are therefore **CLI entry points only**; any helper
> that another tool imports lives in the `avistrack/` package (unnumbered).
> e.g. `parse_frame_name` is imported by `05_build_dataset.py` and must be
> importable — it is exposed via the package, not the numbered script.

> **Not renumbered:** one-off setup (`init_chamber_workspace`,
> `register_chamber_source`, `pick_rois`, `calibrate_time`,
> `edit_valid_ranges`) and pure queries (`list_clips`, `list_experiments`)
> are not per-wave steps and keep their names.

---

## 4. Key data structures & provenance

Every stage leaves an auditable trail. Nothing about a frame's fate is
implicit.

- **`manifests/all_clips.csv`** — the sampling ledger. One row per clip:
  `clip_path, chamber_id, wave_id, source_video, source_drive_uuid, layout,
  start_sec, duration_sec, fps, sampled_at`. There is **no train/val/test
  here** — splitting happens only at build time.
- **`frames/{chamber}/{wave}/*.png`** — the actual images (flat; a
  `_rejected/` subdir holds triage rejects). Frame stem =
  `{chamber}_{wave}_{clip_stem}_f{idx:06d}`.
- **`annotations/{batch_id}/*.txt`** — the **flat-batch annotation
  journal** (see §5). One `.txt` per frame (empty = negative sample), plus
  `_meta.json` + `obj.names`.
- **`manifests/annotation_batches.csv`** — one row per import batch:
  `batch_id, chamber_type, cvat_project_id/name, n_frames, n_positive,
  n_negative, exported_at, imported_at, source_zip`.
- **`datasets/{name}/`** — an immutable built view: `recipe.yaml` (frozen),
  `manifest.csv` (per-frame: split, chamber, wave, clip_stem, frame_stem,
  **batch_id**, image/label link + src), `data.yaml`, and
  `images|labels/{train,val,test}/`. On Windows these are real copies
  (`link_mode=copy`); on Linux they are symlinks.
- **`models/{experiment}/`** — `meta.json` (recipe_hash + git_sha +
  git_dirty + timestamps), `snapshots/` (frozen experiment/recipe/data.yaml
  **+ `uncommitted.diff`**), `phase1/{model}/` bakeoff runs +
  `test_eval.csv`, `final/{best.pt, best.onnx, champion.meta.json}`.

---

## 5. The flat-batch annotation journal

Annotations are a **journal of batches**, not a per-clip tree.

- One CVAT project export = one **batch** = `annotations/{batch_id}/`
  (`{chamber_type}_{YYYY-MM-DD}_batchNN`).
- The same frame may be annotated in more than one batch (re-annotation).
  A recipe's `annotations.resolution` decides which wins:
  `latest` (highest batch_id), `first`, or `error`.
- A recipe selects batches with `annotations.batches` / `exclude_batches`.

This is exactly what makes multi-wave, iterative annotation clean: **each
wave / each annotation round is a new batch**, and a recipe combines the
batches it wants.

---

## 6. Provenance of the tooling itself (important — do not lose)

The code that **built the pre-w4 plus dataset was never committed.** It
lived only as uncommitted working-tree state on the lab machine, captured
by `lineage.py` at experiment time in:

```
models/pre-w4_plus_2026-05-02/snapshots/uncommitted.diff
```

Consequences discovered 2026-07:

- The **committed** `build_dataset.py` read a *nested per-clip* annotation
  layout (`annotations/{chamber}/{wave}/{clip_stem}/*.txt`). The **on-disk
  data is flat-batch** (`annotations/{batch_id}/*.txt`). Running the
  committed tool on the real workspace found **0 frames**.
- The commit that built pre-w4 (`git_sha 8213399…`) is **not in this repo**
  — it was on the lab machine's own line and never pushed.
- The lost code was **recovered from that snapshot diff** and restored
  (commit "Restore batch-aware build_dataset + import_annotations"). The
  diff's four base blobs are byte-identical to the pre-restore files, so it
  reconstructs exactly this tree. The diff was truncated at EOF inside
  `import_annotations.main()`; the three complete files were `git apply`-ed
  and `import_annotations.py` was rebuilt by hand (legacy-mode tail restored
  from the prior version).
- **Verification:** rebuilding the pre-w4 recipe over the existing workspace
  yields **633 frames split 507/62/64 — identical to the pre-w4 manifest**.

Lesson: `06_train`/lineage snapshots (`uncommitted.diff`) are a real safety
net, but the fix is to **commit dataset-assembly code**, which this work
does.

---

## 7. Design decisions

- **`01b` selection is frame-first, but keeps clip provenance (option A).**
  Model-signal selection picks individual frames from the previous model's
  per-frame output, not from sampled clips. To keep `05_build_dataset`
  unchanged (it filters frames by eligible `clip_stem` from
  `all_clips.csv`), each selected frame is registered under a **virtual
  clip** entry in `all_clips.csv`, and frame names keep the
  `{chamber}_{wave}_{clip}_fNNNNNN` convention. Provenance (which source
  video + frame) is preserved; no build_dataset change needed.
- **Everything entering the dataset is perspective-transformed to
  640×640.** Non-negotiable.
- **Numbered CLI + config files, no TUI.** The pipeline runs a few times a
  year and must stay reproducible/scriptable; the genuinely interactive
  steps (`03_review_triage`, `pick_rois`) already have GUIs. `00_status.py`
  covers "where am I" without a TUI.

---

## 8. Combining waves — recipe example

To build a wave3 + wave4 dataset where wave4 also lands in val/test, once
wave4 has been imported as its own batch:

```yaml
name:         plus_w3w4_2026-07
chamber_type: plus
include:
  chambers: ["*"]
  waves:    ["*"]          # or [wave3, wave4]
annotations:
  batches:  ["*"]          # or [plus_2026-05-02_batch01, plus_2026-07-..._batch01]
  resolution: latest
require_annotations: true
split:
  ratios: {train: 0.80, val: 0.10, test: 0.10}
  stratify: wave           # every wave appears in every split
  seed: 42
classes: ["chick"]
```

---

## 9. Reproducibility notes

- A build is deterministic given `(recipe, workspace state)`: same
  `split.seed` + same frames → same split. Re-running a build makes a new
  named dataset; it never re-annotates.
- Adding new frames (e.g. wave4) changes group sizes, so a rebuild does
  **not** guarantee an old frame keeps its previous split. This is a
  test-set *comparability* consideration, not a data risk — the source
  frames/annotations never move. If a fixed held-out test set matters for
  comparing models across dataset versions, pin it deliberately.
- Reproducing an *old* dataset requires the *code version* that built it
  (see §6). Always commit dataset-assembly changes.
