#!/usr/bin/env python3
"""
tools/01b_select_frames.py
──────────────────────────
Model-signal frame selection — the targeted alternative to 01a_sample_clips.

Instead of sampling random clips, this reads the *previous model's per-frame
tracking output* (04_Tracking_RGB/raw/*.parquet) for a (chamber, wave),
keeps only frames inside the experiment's **valid ranges**, and selects the
frames most worth annotating, grouped into interpretable buckets:

  dup       n_det > 1            — false positives (e.g. food dish read as a
                                   2nd chick, or an edge chick split in two)
  lowconf   0 < conf < thresh    — the model is unsure
  miss      n_det == 0           — the model found nothing
  coverage  everything else      — normal frames, for plain coverage

Valid-range filtering is MANDATORY (see avistrack.valid_ranges): without it
the pool is dominated by pre-experiment empty-cage footage.

You set the per-bucket QUOTA. The tool is a mechanism, not a policy — the
quotas are yours.

PHASE 1 (this entry point) is read-only planning: it computes the candidate
pools, applies your quotas with temporal de-duplication, and writes a
selection plan CSV. It never writes to the workspace. PHASE 2
(materialisation: decode → perspective-transform 640×640 → register virtual
clips in all_clips.csv → flat frames + triage manifest) follows once quotas
are set.

Usage
-----
    # True (valid-filtered) candidate pool:
    python tools/01b_select_frames.py \\
        --workspace-yaml F:/BAM/avistrack_workspace/plus/workspace.yaml \\
        --sources-yaml   sources_wave4_temp.yaml \\
        --chamber-id plus_103A --wave-id wave4 --report-only

    # Select with per-bucket quotas → plan CSV:
    python tools/01b_select_frames.py --workspace-yaml ... --sources-yaml ... \\
        --chamber-id plus_103A --wave-id wave4 \\
        --n-dup 280 --n-lowconf 40 --n-miss 0 --n-coverage 80 --out plan_103A.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from avistrack import valid_ranges as vrmod  # noqa: E402
from avistrack.core.time_lookup import load_segment_starts  # noqa: E402
from avistrack.workspace import load_context  # noqa: E402

BUCKETS = ("dup", "lowconf", "miss", "coverage")
SESSION_RE = re.compile(r"(Day\d+_\d{6}_\d{4})")   # tracking parquet stem key


# ── Signal → bucket ───────────────────────────────────────────────────────

def label_buckets(df: pd.DataFrame, lowconf_thresh: float) -> pd.Series:
    """Assign each per-frame row to exactly one bucket (priority: dup >
    miss > lowconf > coverage)."""
    n_det = df["n_det"].to_numpy()
    conf  = df["conf"].to_numpy()
    bucket = pd.Series("coverage", index=df.index, dtype=object)
    bucket[(n_det >= 1) & (conf > 0) & (conf < lowconf_thresh)] = "lowconf"
    bucket[n_det == 0] = "miss"
    bucket[n_det > 1] = "dup"
    return bucket


def _bucket_score(df: pd.DataFrame, bucket: str) -> pd.Series:
    """Higher score = more worth annotating, per bucket."""
    if bucket == "dup":       # more detections / lower conf = more interesting
        return df["n_det"].astype(float) * 100 - df["conf"].astype(float)
    if bucket == "lowconf":   # lowest confidence first
        return -df["conf"].astype(float)
    if bucket == "miss":      # all equal; nudge by frame-diff if present
        return df["fdiff"].astype(float) if "fdiff" in df else pd.Series(0.0, index=df.index)
    return pd.Series(0.0, index=df.index)  # coverage: no preference


# ── Temporal de-duplication ───────────────────────────────────────────────

def temporal_dedup(sub: pd.DataFrame, min_gap: int) -> pd.DataFrame:
    """Greedily drop frames within ``min_gap`` frames of an already-kept
    frame from the same source_video (keeps the highest-scoring first)."""
    if min_gap <= 0 or sub.empty:
        return sub
    keep_idx = []
    for _video, g in sub.groupby("source_video", sort=False):
        g = g.sort_values("score", ascending=False)
        kept_frames: list[int] = []
        for idx, frame in zip(g.index, g["frame"].to_numpy()):
            if all(abs(int(frame) - kf) >= min_gap for kf in kept_frames):
                kept_frames.append(int(frame))
                keep_idx.append(idx)
    return sub.loc[keep_idx]


# ── Selection ─────────────────────────────────────────────────────────────

def select_candidates(
    df: pd.DataFrame,
    quotas: dict[str, int],
    lowconf_thresh: float,
    min_frame_gap: int,
    seed: int,
) -> pd.DataFrame:
    """Return the selected rows with a 'bucket' column. Pure function over a
    per-frame dataframe with columns [source_video, frame, conf, n_det, ...]."""
    df = df.copy()
    df["bucket"] = label_buckets(df, lowconf_thresh)
    picked = []
    for bucket in BUCKETS:
        quota = quotas.get(bucket, 0)
        if quota <= 0:
            continue
        sub = df[df["bucket"] == bucket].copy()
        if sub.empty:
            continue
        sub["score"] = _bucket_score(sub, bucket)
        # Shortlist before the O(n*k) temporal dedup so it stays cheap on
        # multi-hundred-thousand-row buckets.
        cap = max(quota * 30, quota)
        if bucket == "coverage":
            shortlist = sub.sample(n=min(len(sub), cap), random_state=seed)
            deduped = temporal_dedup(shortlist, min_frame_gap)
            take = deduped.sample(n=min(quota, len(deduped)), random_state=seed)
        else:
            shortlist = sub.sort_values("score", ascending=False).head(cap)
            deduped = temporal_dedup(shortlist, min_frame_gap)
            take = deduped.sort_values("score", ascending=False).head(quota)
        picked.append(take)
    if not picked:
        return df.iloc[0:0].assign(bucket=[])
    return pd.concat(picked).sort_values(["source_video", "frame"])


# ── Load valid-range-filtered tracking for one (chamber, wave) ─────────────

def load_valid_tracking(ctx, modality: str = "rgb") -> tuple[pd.DataFrame, dict]:
    """Concatenate every session's tracking parquet, keeping only frames
    inside valid_ranges. Returns (df, stats). df has a 'source_video'
    (session) column. valid_ranges filtering is mandatory."""
    vr = vrmod.load_valid_ranges(ctx.valid_ranges_file)
    if not vr:
        raise SystemExit(
            f"valid_ranges.json missing/empty at {ctx.valid_ranges_file}. "
            f"Valid-range filtering is mandatory for selection.")
    starts = load_segment_starts(ctx.timestamp_calibration_file)
    fps = float(getattr(ctx.workspace.chamber, "fps", None) or 30.0)
    trk = ctx.wave_root / "04_Tracking_RGB" / "raw"
    if not trk.is_dir():
        raise SystemExit(f"tracking dir not found: {trk}")

    parts, n_vids, n_used = [], 0, 0
    for vp in ctx.list_videos(modality=modality):
        n_vids += 1
        V = vp.name
        if V not in vr or V not in starts:
            continue
        m = SESSION_RE.search(V)
        if not m:
            continue
        tp = trk / f"{m.group(1)}.parquet"
        if not tp.exists():
            continue
        df = pd.read_parquet(tp).copy()
        n = int(df["frame"].max()) + 1
        wins = vrmod.frame_windows(vr[V], starts[V], fps, n)
        if not wins:
            continue
        d = df[vrmod.valid_mask(df["frame"].to_numpy(), wins)].copy()
        d["source_video"] = m.group(1)
        parts.append(d)
        n_used += 1

    if not parts:
        raise SystemExit(
            "no valid-range frames found — check valid_ranges.json, "
            "timestamp_calibration.jsonl (needs new_segment events), and the "
            "04_Tracking_RGB/raw parquets.")
    df = pd.concat(parts, ignore_index=True)
    return df, {"videos": n_vids, "sessions_used": n_used, "fps": fps}


def report_pools(df: pd.DataFrame, lowconf_thresh: float) -> dict[str, int]:
    b = label_buckets(df, lowconf_thresh)
    return {k: int((b == k).sum()) for k in BUCKETS}


# ── Phase 2: materialise the selection ────────────────────────────────────

ALL_CLIPS_FIELDS = [
    "clip_path", "chamber_id", "wave_id", "source_video", "source_drive_uuid",
    "layout", "start_sec", "duration_sec", "fps", "sampled_at",
]
TRIAGE_FIELDS = [
    "Frame_Filename", "Source_Clip", "Original_Video_Path", "Frame_Idx",
    "Timestamp", "Bucket", "Triage_Status",
]


def _append_all_clips(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ALL_CLIPS_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def _derive_triage_batch_id(manifests_dir: Path, chamber: str, wave: str) -> str:
    prefix = f"{chamber}_{wave}_{date.today().isoformat()}_batch"
    n = 1
    if manifests_dir.exists():
        nums = []
        for p in manifests_dir.iterdir():
            if p.is_file() and p.suffix == ".csv" and p.stem.startswith(prefix):
                try:
                    nums.append(int(p.stem[len(prefix):]))
                except ValueError:
                    pass
        if nums:
            n = max(nums) + 1
    return f"{prefix}{n:02d}"


def materialise_selection(ctx, sel: pd.DataFrame, wave_id: str, fps: float) -> dict:
    """PHASE 2: decode each selected frame from its remuxed video, perspective-
    transform to the chamber target size, write a flat PNG, register one
    virtual clip per source session in all_clips.csv, and emit a triage
    manifest for 03_review_triage. Everything written is perspective-corrected."""
    import cv2  # noqa: E402
    from avistrack.core.transformer import PerspectiveTransformer  # noqa: E402
    from avistrack.core import rois as roi_utils  # noqa: E402

    chamber = ctx.chamber.chamber_id
    ts = ctx.workspace.chamber.target_size
    target_size = tuple(ts) if ts else (640, 640)

    # session -> remuxed video path
    vids = {}
    for vp in ctx.list_videos(modality="rgb"):
        m = SESSION_RE.search(vp.name)
        if m:
            vids[m.group(1)] = vp

    frame_dir = ctx.frame_dir
    frame_dir.mkdir(parents=True, exist_ok=True)

    all_clips_rows, triage_rows = [], []
    n_written = n_fail = 0
    for session, g in sel.groupby("source_video", sort=True):
        vp = vids.get(session)
        if vp is None:
            print(f"  ! no video for session {session}; skipping {len(g)} frame(s)")
            n_fail += len(g); continue
        corners = roi_utils.resolve_corners(ctx.metadata_dir, vp.name)
        if not corners:
            print(f"  ! no corners for {vp.name}; skipping {len(g)} frame(s)")
            n_fail += len(g); continue
        tf = PerspectiveTransformer(corners, target_size)
        virtual_stem = f"{chamber}_{wave_id}_{session}_sel_transformed"
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"  ! cannot open {vp}; skipping {len(g)} frame(s)")
            n_fail += len(g); cap.release(); continue
        for _, r in g.sort_values("frame").iterrows():
            fi = int(r["frame"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok or frame is None:
                n_fail += 1; continue
            warped = tf.transform(frame)
            fname = f"{virtual_stem}_f{fi:06d}.png"
            cv2.imwrite(str(frame_dir / fname), warped)
            triage_rows.append({
                "Frame_Filename": fname,
                "Source_Clip": f"{virtual_stem}.mp4",
                "Original_Video_Path": str(vp),
                "Frame_Idx": str(fi),
                "Timestamp": f"{fi / fps:.3f}",
                "Bucket": r["bucket"],
                "Triage_Status": "pending",
            })
            n_written += 1
        cap.release()
        fr = g["frame"].astype(int)
        all_clips_rows.append({
            "clip_path": f"clips/{chamber}/{wave_id}/{virtual_stem}.mp4",
            "chamber_id": chamber, "wave_id": wave_id,
            "source_video": vp.name, "source_drive_uuid": ctx.chamber.drive_uuid,
            "layout": ctx.wave.layout,
            "start_sec": f"{fr.min() / fps:.2f}",
            "duration_sec": f"{(fr.max() - fr.min()) / fps:.2f}",
            "fps": f"{fps:.3f}",
            "sampled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

    _append_all_clips(ctx.all_clips_csv, all_clips_rows)
    manifests_dir = ctx.manifests_root / "triage"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    batch_id = _derive_triage_batch_id(manifests_dir, chamber, wave_id)
    manifest_path = manifests_dir / f"{batch_id}.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRIAGE_FIELDS)
        w.writeheader()
        w.writerows(triage_rows)

    return {"written": n_written, "failed": n_fail,
            "virtual_clips": len(all_clips_rows),
            "batch_id": batch_id, "manifest": manifest_path, "frames_dir": frame_dir}


# ── Novelty bucket (embedding distance vs a reference set) ─────────────────
#
# Unlike dup/lowconf/miss/coverage (free from the parquet), novelty needs
# pixels: it decodes + perspective-warps a shortlist of coverage frames,
# embeds them with the same model that produced the reference embeddings,
# and keeps the frames least similar to anything already in the dataset.

def _decode_warp_frames(ctx, sub: pd.DataFrame):
    """Yield (row_index, warped_bgr_640) for each row in ``sub`` (needs
    source_video + frame). Same decode+warp path as materialise."""
    import cv2  # noqa: E402
    from avistrack.core.transformer import PerspectiveTransformer  # noqa: E402
    from avistrack.core import rois as roi_utils  # noqa: E402

    ts = ctx.workspace.chamber.target_size
    target_size = tuple(ts) if ts else (640, 640)
    vids = {}
    for vp in ctx.list_videos(modality="rgb"):
        m = SESSION_RE.search(vp.name)
        if m:
            vids[m.group(1)] = vp
    for session, g in sub.groupby("source_video", sort=True):
        vp = vids.get(session)
        if vp is None:
            continue
        corners = roi_utils.resolve_corners(ctx.metadata_dir, vp.name)
        if not corners:
            continue
        tf = PerspectiveTransformer(corners, target_size)
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            cap.release(); continue
        for idx, r in g.sort_values("frame").iterrows():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame"]))
            ok, frame = cap.read()
            if ok and frame is not None:
                yield idx, tf.transform(frame)
        cap.release()


def select_novelty(ctx, df: pd.DataFrame, quota: int, ref_emb_path: Path,
                   yolo_weights: Path, lowconf_thresh: float, min_frame_gap: int,
                   seed: int, shortlist_factor: int = 10,
                   shortlist_cap: int = 2000) -> pd.DataFrame:
    """Pick the ``quota`` coverage frames most novel vs the reference
    embeddings (novelty = 1 - max cosine similarity). Decodes + warps +
    embeds a bounded shortlist so cost stays proportional to the quota."""
    import numpy as np

    ref = np.load(str(ref_emb_path)).astype("float32")
    ref = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8)

    d = df.copy()
    d["bucket"] = label_buckets(d, lowconf_thresh)
    pool = d[d["bucket"] == "coverage"].copy()
    if pool.empty:
        return pool.assign(novelty=[])
    pool["score"] = 0.0
    n_short = min(len(pool), max(quota * shortlist_factor, quota), shortlist_cap)
    shortlist = temporal_dedup(pool.sample(n=n_short, random_state=seed), min_frame_gap)

    imgs, idxs = [], []
    for idx, warped in _decode_warp_frames(ctx, shortlist):
        imgs.append(warped)
        idxs.append(idx)
    if not imgs:
        return pool.iloc[0:0].assign(novelty=[])

    from ultralytics import YOLO
    model = YOLO(str(yolo_weights))
    embs = []
    B = 64
    for i in range(0, len(imgs), B):
        for t in model.embed(imgs[i:i + B], verbose=False):
            embs.append(t.detach().cpu().numpy().astype("float32").ravel())
    E = np.stack(embs)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    novelty = 1.0 - (E @ ref.T).max(axis=1)

    res = df.loc[idxs].copy()
    res["bucket"] = "novelty"
    res["novelty"] = novelty
    return res.sort_values("novelty", ascending=False).head(quota)


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace-yaml", required=True, type=Path)
    ap.add_argument("--sources-yaml", type=Path, default=None,
                    help="sources.yaml (default: sibling of workspace.yaml)")
    ap.add_argument("--chamber-id", required=True)
    ap.add_argument("--wave-id", required=True)
    ap.add_argument("--modality", default="rgb", choices=["rgb", "ir"])
    ap.add_argument("--lowconf-thresh", type=float, default=0.4)
    ap.add_argument("--min-frame-gap", type=int, default=15,
                    help="Min frames between selected frames of the same session")
    ap.add_argument("--n-dup",      type=int, default=0)
    ap.add_argument("--n-lowconf",  type=int, default=0)
    ap.add_argument("--n-miss",     type=int, default=0)
    ap.add_argument("--n-coverage", type=int, default=0)
    ap.add_argument("--n-novelty",  type=int, default=0,
                    help="Novelty frames (embedding distance vs --ref-emb); needs "
                         "--ref-emb + --yolo-weights (decodes + embeds a shortlist)")
    ap.add_argument("--ref-emb", type=Path, default=None,
                    help="Reference embeddings .npy (L2-normalized rows) for novelty")
    ap.add_argument("--yolo-weights", type=Path, default=None,
                    help="Model .pt to embed candidate frames (must match --ref-emb)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-only", action="store_true",
                    help="Only print the candidate pool sizes; no selection")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write the selection plan CSV here")
    ap.add_argument("--materialize", action="store_true",
                    help="PHASE 2: decode + perspective-warp + write the selected "
                         "frames into the workspace (flat frames + all_clips.csv "
                         "virtual clips + triage manifest)")
    args = ap.parse_args()

    sources_yaml = args.sources_yaml or args.workspace_yaml.with_name("sources.yaml")
    ctx = load_context(
        workspace_yaml=args.workspace_yaml, sources_yaml=sources_yaml,
        chamber_id=args.chamber_id, wave_id=args.wave_id, require_drive=True)

    df, stats = load_valid_tracking(ctx, modality=args.modality)
    pools = report_pools(df, args.lowconf_thresh)

    print(f"{args.chamber_id}/{args.wave_id}: {len(df):,} valid-range frames "
          f"from {stats['sessions_used']}/{stats['videos']} session(s), fps={stats['fps']:g}")
    print(f"Candidate pools (lowconf<{args.lowconf_thresh}):")
    for k in BUCKETS:
        print(f"  {k:9s}: {pools[k]:>10,}")

    if args.report_only:
        print("\n(report-only — set --n-* quotas to select)")
        return

    quotas = {"dup": args.n_dup, "lowconf": args.n_lowconf,
              "miss": args.n_miss, "coverage": args.n_coverage}
    sel = select_candidates(df, quotas, args.lowconf_thresh,
                            args.min_frame_gap, args.seed)

    if args.n_novelty > 0:
        if not args.ref_emb or not args.yolo_weights:
            raise SystemExit("--n-novelty requires --ref-emb and --yolo-weights")
        print(f"\nComputing novelty bucket ({args.n_novelty}) — decode + embed shortlist ...")
        nov = select_novelty(ctx, df, args.n_novelty, args.ref_emb,
                             args.yolo_weights, args.lowconf_thresh,
                             args.min_frame_gap, args.seed)
        sel = pd.concat([sel, nov])
        # novelty draws from the coverage pool; drop any frame picked twice
        sel = sel.drop_duplicates(subset=["source_video", "frame"], keep="first")
    quotas["novelty"] = args.n_novelty

    print(f"\nSelected {len(sel)} frame(s):")
    for k in (*BUCKETS, "novelty"):
        n = int((sel["bucket"] == k).sum())
        if quotas.get(k, 0) or n:
            print(f"  {k:9s}: {n:>6} / quota {quotas.get(k, 0)}")

    if args.out:
        cols = [c for c in ("source_video", "frame", "bucket", "conf", "n_det",
                            "cx", "cy", "w", "h", "lum", "fdiff", "novelty") if c in sel.columns]
        sel[cols].to_csv(args.out, index=False)
        print(f"\nPlan written -> {args.out}")

    if args.materialize:
        print("\nMaterialising (decode -> perspective-warp 640x640 -> write) ...")
        m = materialise_selection(ctx, sel, args.wave_id, stats["fps"])
        print(f"  wrote {m['written']} frame(s); {m['failed']} failed; "
              f"{m['virtual_clips']} virtual clip(s) registered")
        print(f"  frames       : {m['frames_dir']}")
        print(f"  triage batch : {m['batch_id']}  ->  {m['manifest']}")
        print(f"\nNext: python tools/03_review_triage.py --workspace-yaml <ws> "
              f"--sources-yaml <src> --chamber-id {args.chamber_id} "
              f"--wave-id {args.wave_id} --batch {m['batch_id']}")
    else:
        print("\nNext (PHASE 2, writes to workspace): re-run with --materialize to "
              "decode + warp 640x640 + write frames + all_clips.csv + triage manifest.")


if __name__ == "__main__":
    main()
