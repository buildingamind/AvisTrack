# Chamber Workspace Bootstrap SOP

Standard operation procedure for setting up a new chamber type / drive / wave
under the multi-chamber storage architecture. Extracted from the
2026-05-01 VR-chamber bootstrap session.

---

## Architecture context

Two-layer storage:

- **Workspace SSD** — central artifacts (clips, frames, annotations, manifests,
  datasets, models). One workspace per *chamber type*, e.g.
  `F:\BAM\avistrack_workspace\vr\` for VR chambers.
- **Chamber drive(s)** — raw videos + per-wave metadata. One marker file
  `<drive>:\_avistrack_source.yaml` carries the chamber identity (UUID) so
  the workspace can resolve it on any machine.

Tools are workspace-mode (`--workspace-yaml --chamber-id --wave-id`) and
expect both layers to be mounted. Legacy `--config` mode still works for
single-drive layouts (e.g. some Wave2 IR future video annotations).

---

## Per-wave standard layout (on the chamber drive)

```
<drive>:/<wave>/
    00_raw_videos/{RGB,IR}/*.mkv          # raw recordings, modality subdirs
    02_Global_Metadata/                   # camera_rois.json, time_calibration.json, ...
```

`02_Global_Metadata` (not `Chamber_Metadata`) is the canonical name. After
retrofit, every `waves:` entry in `sources.yaml` becomes an identical
template — adding a new wave is copy-paste.

---

## Standard procedure (replicable)

For each new chamber type / drive / wave:

```powershell
# 1. Once per chamber type (workspace SSD plugged in):
python tools/init_chamber_workspace.py `
    --workspace-root F:/BAM/avistrack_workspace --chamber-type <type>

# 2. Once per chamber drive (drive mounted; writes _avistrack_source.yaml):
python tools/register_chamber_source.py `
    --workspace-root F:/BAM/avistrack_workspace `
    --chamber-type <type> --chamber-id <type>_<id> `
    --mount <drive_letter>:/

# 3. Per wave: edit sources.yaml `waves:` list to add a structured entry.
#    For old dumps without 00_raw_videos layout, use scan_legacy_wave.py.

# 4. Per wave: pick ROIs (interactive GUI; corners auto-carry across videos)
python tools/pick_rois.py `
    --workspace-yaml F:/BAM/avistrack_workspace/<type>/workspace.yaml `
    --chamber-id <id> --wave-id <wave> --modality all

# 5. Validate ROI coverage (rgb / ir separately; --modality all not supported)
python tools/pick_rois.py validate `
    --workspace-yaml ... --chamber-id ... --wave-id ... --modality rgb
python tools/pick_rois.py validate `
    --workspace-yaml ... --chamber-id ... --wave-id ... --modality ir

# 6. (Optional) Time calibration — only when wall-clock sampling matters.
#    VR single-subject workflow skips this entirely.
#    edit_valid_ranges.py depends on time_calibration.json, so it's also
#    skipped when calibrate_time.py is skipped.

# ── Image-based annotation flow (frames → CVAT) ──

# 7. Sample clips from the wave (provenance unit)
python tools/01a_sample_clips.py `
    --workspace-yaml ... --chamber-id ... --wave-id ... `
    --modality rgb --n 20 --duration 3 --min-gap 5

# 8. Extract PNG frames per clip
python tools/02_extract_frames.py `
    --workspace-yaml ... --chamber-id ... --wave-id ... `
    --frames-per-clip 5 --hash-threshold 5
# Prints batch_id, e.g. vr_105A_wave3_1_2026-05-01_batch01

# 9. Triage (browser GUI)
python tools/03_review_triage.py `
    --workspace-yaml ... --chamber-id ... --wave-id ... `
    --batch <batch_id>
# http://localhost:5000
# Keys: a/→ approve, x/← reject, z undo, space next-pending, Ctrl-S save

# 10. Upload approved frames to CVAT, annotate, export YOLO txt, then:
python tools/04_import_annotations.py `
    --workspace-yaml ... --cvat-export ./export.zip
```

---

## Conventions & known constraints

### Modality strategy
RGB and IR are recorded by the same camera concurrently. Both modalities
register under the same wave entry; `sample_clips --modality {rgb|ir}`
picks at sample time via filename keyword filtering. Corners auto-carry
between consecutive videos, so picking ROIs for ~40 videos in practice
means ~4 unique quadrilaterals.

### Clip layer is provenance, not annotation unit
We keep clips as the lineage anchor (`manifests/all_clips.csv` ties
`chamber_id + wave_id + source_video + drive_uuid + sampled_at`).
`build_dataset.py --recipe` filters at clip level. CVAT still gets PNG
frames — the clip layer is invisible to annotators. Skipping clips would
require a parallel lineage system + edits across most downstream tools.

### Corrupt source files
If `cv2.VideoCapture(path).isOpened() == False` (e.g. EBML headers filled
with `0x00`), rename `*.mkv` → `*.mkv.broken`. The extension filter in
`workspace.list_videos` then excludes them. Files stay on disk for
possible future recovery via `ffmpeg -err_detect ignore_err`.

### Skip time calibration when wall-clock isn't needed
VR single-subject workflows don't need time calibration. Document the
decision inline in `sources.yaml` (per-wave NOTE comment) so future
operators know it was deliberate.

### Tool warts (known)
- `pick_rois.py validate --modality all` is not supported — run twice.
- `pick_rois.py validate` exits non-zero on a wave with 0 videos in the
  chosen modality (cosmetic; empty input).
- `extract_frames.py validate` subcommand does not exist.

---

## File outputs (reference)

| Path | What |
|---|---|
| `<workspace>/<type>/workspace.yaml` | created by `init_chamber_workspace.py` |
| `<workspace>/<type>/sources.yaml` | edited by `register_chamber_source.py` + manual `waves:` entries |
| `<workspace>/<type>/{clips,frames,annotations,manifests,datasets,models}/` | created on first use |
| `<drive>:/_avistrack_source.yaml` | drive marker (UUID) |
| `<drive>:/<wave>/02_Global_Metadata/camera_rois.json` | per-wave ROI corners |
| frames | `{workspace}/frames/{chamber_id}/{wave_id}/{clip_stem}/f{idx:06d}.png` |
| triage manifest | `{workspace}/manifests/triage/{batch_id}.csv` |
| rejected frames | moved to `{frames}/{clip_stem}/_rejected/` |
