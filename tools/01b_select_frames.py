#!/usr/bin/env python3
"""
tools/01b_select_frames.py
──────────────────────────
Model-signal frame selection — the targeted alternative to 01a_sample_clips.

Instead of sampling random clips, this reads the *previous model's per-frame
tracking output* (the 04_Tracking_RGB parquets) and selects the frames that
are most worth annotating, grouped into interpretable buckets:

  dup       n_det > 1            — false positives (e.g. food dish read as a
                                   2nd chick, or an edge chick split in two)
  lowconf   0 < conf < thresh    — the model is unsure
  miss      n_det == 0           — the model found nothing
  coverage  everything else      — normal frames, for plain coverage

You set the per-bucket QUOTA (how many frames from each). The tool does not
decide the sampling policy — it is a mechanism; the quotas are yours.

This entry point is PHASE 1 (planning): it is read-only. It computes the
candidate pools, applies your quotas with temporal de-duplication, and writes
a selection plan CSV. It never touches the workspace. PHASE 2
(materialisation: decode → perspective-transform 640×640 → register virtual
clips in all_clips.csv → write flat frames + triage manifest) runs only after
you have reviewed the pool and set final quotas.

Usage
-----
    # Report the candidate pool (no quotas needed) + write full plan:
    python tools/01b_select_frames.py \\
        --tracking-dir E:/Wave4_PairedCapacityPlus/04_Tracking_RGB/raw \\
        --report-only

    # Select with explicit per-bucket quotas:
    python tools/01b_select_frames.py \\
        --tracking-dir E:/Wave4_PairedCapacityPlus/04_Tracking_RGB/raw \\
        --n-dup 500 --n-lowconf 150 --n-miss 50 --n-coverage 100 \\
        --lowconf-thresh 0.4 --min-frame-gap 15 --seed 42 \\
        --out plan.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BUCKETS = ("dup", "lowconf", "miss", "coverage")


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


# ── Parquet loading ───────────────────────────────────────────────────────

def load_tracking(tracking_dir: Path) -> pd.DataFrame:
    """Concatenate every per-session tracking parquet, tagging source_video."""
    parquets = sorted(tracking_dir.glob("*.parquet"))
    if not parquets:
        raise SystemExit(f"no .parquet files under {tracking_dir}")
    frames = []
    for p in parquets:
        d = pd.read_parquet(p, columns=None)
        d = d.copy()
        d["source_video"] = p.stem
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    for col in ("frame", "conf", "n_det"):
        if col not in df.columns:
            raise SystemExit(f"tracking parquet missing required column {col!r} "
                             f"(have {list(df.columns)})")
    return df


def report_pools(df: pd.DataFrame, lowconf_thresh: float) -> dict[str, int]:
    b = label_buckets(df, lowconf_thresh)
    return {k: int((b == k).sum()) for k in BUCKETS}


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracking-dir", required=True, type=Path,
                    help="Directory of per-session tracking parquets (04_Tracking_RGB/raw)")
    ap.add_argument("--lowconf-thresh", type=float, default=0.4)
    ap.add_argument("--min-frame-gap", type=int, default=15,
                    help="Min frames between selected frames of the same video")
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

    df = load_tracking(args.tracking_dir)
    pools = report_pools(df, args.lowconf_thresh)

    print(f"Loaded {len(df):,} frame-rows from {args.tracking_dir}")
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
        print(f"\nPlan written → {args.out}")
    print("\nNext (PHASE 2, writes to workspace): materialise the plan — decode "
          "each frame, perspective-transform to 640x640, register virtual clips "
          "in all_clips.csv, write flat frames + triage manifest for 03_review_triage.")


if __name__ == "__main__":
    main()
