# VR Workspace Bootstrap — Session log 2026-05-01

End-to-end first-time setup of a VR chamber-type workspace under the
multi-chamber storage architecture (improve-plan §1). This log doubles as
the onboarding recipe for the next chamber type / new wave.

---

## State at start

- `improve-plan.md` Stage A-G largely implemented (loader/workspace/
  init/register/sample_clips already on the new yaml schema).
- `extract_frames.py` + `review_triage.py` still on legacy `--config` flow.
- E:\ drive: 39 raw .mkv files spread across `Wave2/IndividualDanger/`,
  `Wave3/{IndividualDanger empty, VRChamber/{RGB,IR}_Videos}`, `Wave3_1/RGB/`.
- No camera_rois / time_calibration / clip / dataset / model artifacts yet.

## State at end

- Central workspace bootstrapped at `F:\BAM\avistrack_workspace\vr\`.
- E:\ raw videos restructured into the standard
  `<wave>/00_raw_videos/{RGB,IR}/` + `<wave>/02_Global_Metadata/` layout,
  uniformly across the 3 waves.
- chamber `vr_105A` registered (volume serial `FAD0-443D`); 3 waves
  (`wave2`, `wave3`, `wave3_1`) declared in `sources.yaml`.
- ROI corners picked for 37/39 videos. Two Day11 .mkv files (RGB+IR) had
  corrupt EBML headers (filled with 0x00); renamed to `*.mkv.broken` and
  documented inline in `sources.yaml`.
- `extract_frames.py` and `review_triage.py` patched to support workspace
  mode (`--workspace-yaml --chamber-id --wave-id`) without breaking the
  legacy `--config` flow.
- Time calibration (`calibrate_time.py`) intentionally skipped — VR chamber
  workflow does not need wall-clock-anchored sampling.

## Where we left off

About to run the first sample → extract → triage cycle on `wave3_1` (RGB,
2 source videos) to validate the patched tooling. Commands prepared but
not yet executed.

---

## Key decisions (with rationale)

### Workspace location
**`F:\BAM\avistrack_workspace`** — independent of `F:\BAM\Track\` to keep
the new architecture cleanly separate from existing artifacts.

### Source-drive layout policy
Decision: retrofit the 3 existing waves into the user's standard layout
*without* moving 50 GB of corrupt-or-otherwise data; just rename folders.
- `00_raw_videos/{RGB,IR}/*.mkv` (subdir per modality)
- `02_Global_Metadata/` (metadata folder name uses **Global**, not the
  improve-plan template's `Chamber_Metadata`, to match the user's history)

After retrofit, every wave entry in `sources.yaml` is an identical template
(same `raw_videos_subpath` / `metadata_subpath` placeholders), so adding
wave4… is a copy-paste.

### Modality strategy
RGB and IR are recorded by the same camera at the same time. ROI corners
auto-carry from previous video → in practice user only picks ~4 unique
quadrilaterals across 39 videos. Both modalities are registered under the
same wave entry; `sample_clips --modality {rgb|ir}` picks at sample time
via filename keyword filtering.

### Day11 corrupt files
Two files (~50 GB total) had 0x00 EBML headers, opencv `opened=False`.
Excluded from pipeline by renaming `*.mkv` → `*.mkv.broken` (extension
filter in `workspace.list_videos`). Files preserved on disk; recovery via
`ffmpeg -err_detect ignore_err` documented as a future option in the
NOTE comment under wave2 in `sources.yaml`.

### Skip time calibration
VR single-subject workflow doesn't need wall-clock anchoring. We don't run
`calibrate_time.py` and therefore can't run `edit_valid_ranges.py` either
(it depends on `time_calibration.json`).

### extract_frames + review_triage workspace migration
These two tools were the last legacy `--config` holdouts. Patched in this
session to accept `--workspace-yaml --chamber-id --wave-id`. Output paths:
- frames → `{workspace}/frames/{chamber_id}/{wave_id}/{clip_stem}/f{idx:06d}.png`
- triage manifest → `{workspace}/manifests/triage/{batch_id}.csv`
- rejected → moved to `{frames}/{clip_stem}/_rejected/`

Legacy `--config` mode preserved for waves that may still use the old
single-drive layout (e.g. Wave2 IR future video-mode annotation).

---

## Files created / modified

| Path | What |
|---|---|
| `F:\BAM\avistrack_workspace\vr\workspace.yaml` | new — per `init_chamber_workspace.py` |
| `F:\BAM\avistrack_workspace\vr\sources.yaml`   | new — `vr_105A` + 3 waves; inline NOTE re Day11 |
| `F:\BAM\avistrack_workspace\vr\{clips,frames,annotations,manifests,datasets,models}/` | empty subdirs |
| `E:\_avistrack_source.yaml` | new — chamber marker (uuid `FAD0-443D`) |
| `E:\Wave2\02_Global_Metadata\camera_rois.json` | 20 entries (Day11 excluded) |
| `E:\Wave3\02_Global_Metadata\camera_rois.json` | 15 entries (7 RGB + 8 IR) |
| `E:\Wave3_1\02_Global_Metadata\camera_rois.json` | 2 entries |
| `E:\Wave2\00_raw_videos\Day11_*.mkv.broken` | quarantined corrupt files |
| `tools/extract_frames.py` | + workspace mode (`_resolve_paths_workspace`) |
| `tools/review_triage.py` | + workspace mode + per-clip subdir support |

No `avistrack/` package code changed — both tool patches are self-contained.

---

## Standard operation procedure (replicable)

For each new chamber type / drive / wave:

```powershell
# Once per chamber type (workspace SSD must be plugged in):
python tools/init_chamber_workspace.py `
    --workspace-root F:/BAM/avistrack_workspace --chamber-type <type>

# Once per chamber drive (drive must be mounted):
python tools/register_chamber_source.py `
    --workspace-root F:/BAM/avistrack_workspace `
    --chamber-type <type> --chamber-id <type>_<id> `
    --mount <drive_letter>:/

# Per wave: edit sources.yaml waves: list to add structured entry
#   (or scan_legacy_wave.py for old dumps without 00_raw_videos layout)

# Per wave: pick ROIs (interactive GUI; corners carry forward across videos)
python tools/pick_rois.py `
    --workspace-yaml F:/BAM/avistrack_workspace/<type>/workspace.yaml `
    --chamber-id <id> --wave-id <wave> --modality all

# Per wave: validate ROI coverage
python tools/pick_rois.py validate `
    --workspace-yaml ... --chamber-id ... --wave-id ... --modality rgb
python tools/pick_rois.py validate `
    --workspace-yaml ... --chamber-id ... --wave-id ... --modality ir

# (Optional) calibrate_time.py + edit_valid_ranges.py — only when wall-clock
# sampling matters (skipped for VR).

# ── For image-based annotation flow (frames in CVAT) ──
python tools/sample_clips.py `
    --workspace-yaml ... --chamber-id ... --wave-id ... `
    --modality rgb --n 20 --duration 3 --min-gap 5

python tools/extract_frames.py `
    --workspace-yaml ... --chamber-id ... --wave-id ... `
    --frames-per-clip 5 --hash-threshold 5
# Prints batch_id like vr_105A_wave3_1_2026-05-01_batch01

python tools/review_triage.py `
    --workspace-yaml ... --chamber-id ... --wave-id ... `
    --batch <batch_id>
# Browser opens at http://localhost:5000
# Keys: a/→ approve, x/← reject, z undo, space next-pending, Ctrl-S save

# ── Then: upload approved frames to CVAT, annotate, export YOLO txt ──
python tools/import_annotations.py `
    --workspace-yaml ... --cvat-export ./export.zip
```

---

## Rationale: why we didn't skip clips and go straight to frames

User's reasonable question. Decision: **keep the clip intermediate layer**.

- `manifests/all_clips.csv` is the lineage anchor (chamber_id + wave_id +
  source_video + drive_uuid + sampled_at). Skip clips → skip lineage.
- `build_dataset.py --recipe` filters at clip level (`include/exclude`).
- `import_annotations.py` mirrors clip subdirs into annotations/.
- Clips are cheap (~few MB each, 100 MB for 20 × 3-sec).
- Frames-only path would require a parallel lineage system + edits to ~80%
  of downstream tools — 4-6h work for zero user-facing benefit (annotation
  unit in CVAT is the frame either way).

The clip layer is a **provenance unit, not an annotation unit**. CVAT still
gets PNG frames.

---

## Known gaps / future work

- `extract_frames.py validate` subcommand doesn't exist (we use direct
  `pick_rois.py validate` for ROIs only). Might add a manifest validator
  later if needed.
- `pick_rois.py validate --modality all` not supported (only rgb/ir).
  Run twice to cover both. Minor tool wart.
- `pick_rois.py validate` crashes when target wave has 0 videos in chosen
  modality (e.g. wave3_1 IR). Cosmetic — exit non-zero with empty input.
- Day11 .mkv.broken recovery via `ffmpeg -err_detect ignore_err` not
  attempted; ~50 GB of potentially recoverable footage.
- Wave2 IR not yet sampled. User opted to defer until needed.
