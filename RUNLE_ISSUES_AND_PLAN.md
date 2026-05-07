# Runle Issues & Fix Plan (AvisTrack)

> **Purpose**: AvisTrack (analysis pipeline) side TODO list. Mirrors the format of `ChamberBroadcaster/RUNLE_ISSUES_AND_PLAN.md`; cross-repo decisions reference each other.
>
> **Last updated**: 2026-04-19

---

## Table of Contents

| Section | Count |
|---|---|
| [Fixed](#fixed) | — |
| [Unfixed](#unfixed) | A1 |
| [Batch Implementation Plan](#batch-implementation-plan) | Batch A — Video Input Validation |

---

## Fixed

*(none yet)*

---

## Unfixed

### A1 — Analysis entry points must reject un-remuxed recording segments

- **Upstream context**: ChamberBroadcaster recordings retain fragmented MP4 as a crash-resistance measure (see `ChamberBroadcaster/RUNLE_ISSUES_AND_PLAN.md` Q1 / Q21). After a recording session ends, `scripts/remux_after.py` on the ChamberBroadcaster side must be run to complete the re-mux. Un-remuxed segments have the following issues:
  - Windows built-in player cannot seek
  - Clip tools have slow random access
  - Inference experience is degraded / edge cases may read half-frames
- **Symptom**: If AvisTrack silently accepts un-remuxed files → tracking results may contain spurious samples from frame-read anomalies, which are expensive to trace after the fact.
- **Requirement**: All video-read entry points (dataset loader / sample clip tool / evaluation scripts / CLI / any `cv2.VideoCapture` / `decord` / `ffmpeg-python` call sites) must pass through a unified validator before opening a file, performing **3 hard-fail checks**. Any one triggered → `raise`, **no warn / no skip / no silent downgrade**:

  1. **Filename starts with `FRAGMENTED_`** → this segment has not yet been processed by `scripts/remux_after.py`.
  2. **The file's directory contains `!!_PENDING_REMUX_README.txt`** → at least one segment in the whole directory is unprocessed; even if the current segment appears to lack the `FRAGMENTED_` prefix (e.g. manually renamed by user), reject the entire directory to prevent inconsistency.
  3. **The `timestamp_calibration.jsonl` in the same directory shows the corresponding `video_file` with the latest status `remux_status=pending` and no subsequent `reason=remux_complete` row** → accounting-layer incomplete.

- **Why all three (OR logic)**:
  - Filename only: user can bypass by manually renaming
  - txt only: a single file copied in isolation won't carry the txt
  - jsonl only: old recordings / non-standard paths may lack a jsonl
  - The three checks are complementary; any one triggered is sufficient to reject

- **Error message template** (English, for cross-member copy-paste debugging):
  ```
  Refusing to read {path}: recording segment has not been remuxed.
  Trigger: {which of the 3 checks fired}
  Fix:
    conda activate chamber_broadcaster
    cd <ChamberBroadcaster repo>
    python scripts/remux_after.py "{parent_dir}"
  Background: AvisTrack/RUNLE_ISSUES_AND_PLAN.md#a1
             ChamberBroadcaster/RUNLE_ISSUES_AND_PLAN.md (Q1 / Q21)
  ```

- **No downgrade option** (e.g. an `ALLOW_FRAGMENTED=1` environment variable): the value of the contract comes from its rigidity. Any "temporary bypass" switch will become on-by-default → spurious samples creep back in. Anyone who needs to debug with fragmented files can manually remux a copy.

- **Files** (TBD, enumerate on first implementation): unified helper `validate_recording_path(path)`, called unconditionally by all video-read entry points.

---

## Batch Implementation Plan

### Batch A — Video Input Validation (A1)

1. **Enumerate video-read entry points**: grep this repo for `cv2.VideoCapture` / `decord` / `ffmpeg-python` / `imageio` / any custom video reader usage; list dataset loaders, sample clip tools, evaluation scripts, CLI entry points.
2. **Create helper** `validate_recording_path(path: Path) -> None`:
   - Implement the 3 hard-fail checks
   - On failure raise `RecordingNotRemuxedError(path, trigger, fix_cmd)`; exception class inherits `RuntimeError`, `__str__` uses the error message template above
3. **Enforce the helper at all entry points**, removing duplicate / incomplete checks.
4. **Integration tests**:
   - Construct `FRAGMENTED_dummy.mp4` → rejected (trigger: filename)
   - Construct `dummy.mp4` + `!!_PENDING_REMUX_README.txt` in same dir → rejected (trigger: pending txt)
   - Construct `dummy.mp4` + `timestamp_calibration.jsonl` containing `{"video_file": "dummy.mp4", "remux_status": "pending"}` with no subsequent `remux_complete` → rejected (trigger: jsonl pending)
   - Same as above but jsonl has a subsequent `{"video_file": "dummy.mp4", "reason": "remux_complete"}` row → passes
   - Valid `dummy.mp4` with no txt / clean jsonl → passes

### Decision Log

- **A1** (2026-04-19): The 04-19 decision in ChamberBroadcaster Q21 chose "broadcaster drops marker + standalone remux script" (see upstream md). The analysis side guards against this with 3 hard-fail checks: filename / txt presence / jsonl status. Three OR conditions; any one triggered rejects. Error message points to upstream `scripts/remux_after.py`. Refused to add a "skip validation" downgrade switch — contract value comes from rigidity.

---

### Key File Index

| Issue | Core Files |
|---|---|
| A1 | All video-read entry points in this repo (enumerate on Batch A first implementation); new `validate_recording_path(path)` helper; new exception `RecordingNotRemuxedError` |
