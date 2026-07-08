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
import re
import sys
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-only", action="store_true",
                    help="Only print the candidate pool sizes; no selection")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write the selection plan CSV here")
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

    print(f"\nSelected {len(sel)} frame(s):")
    for k in BUCKETS:
        n = int((sel["bucket"] == k).sum())
        if quotas.get(k, 0) or n:
            print(f"  {k:9s}: {n:>6} / quota {quotas.get(k, 0)}")

    if args.out:
        cols = [c for c in ("source_video", "frame", "bucket", "conf", "n_det",
                            "cx", "cy", "w", "h", "lum", "fdiff") if c in sel.columns]
        sel[cols].to_csv(args.out, index=False)
        print(f"\nPlan written -> {args.out}")
    print("\nNext (PHASE 2, writes to workspace): materialise the plan — decode "
          "each frame, perspective-transform to 640x640, register virtual clips "
          "in all_clips.csv, write flat frames + triage manifest for 03_review_triage.")


if __name__ == "__main__":
    main()
