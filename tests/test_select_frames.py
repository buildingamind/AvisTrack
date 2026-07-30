"""
tests/test_select_frames.py – pure selection logic in tools/01b_select_frames.py.

Loads the numbered CLI tool by file path (its module name starts with a
digit, so it can only be loaded this way, not imported).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_select_frames():
    path = REPO_ROOT / "tools" / "01b_select_frames.py"
    spec = importlib.util.spec_from_file_location("select_frames_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sf = _load_select_frames()


def _df(rows):
    return pd.DataFrame(rows)


def test_label_buckets_priority():
    df = _df([
        {"frame": 0, "conf": 0.9, "n_det": 2},   # dup   (n_det>1 wins)
        {"frame": 1, "conf": 0.9, "n_det": 0},   # miss
        {"frame": 2, "conf": 0.3, "n_det": 1},   # lowconf
        {"frame": 3, "conf": 0.9, "n_det": 1},   # coverage
    ])
    assert list(sf.label_buckets(df, lowconf_thresh=0.4)) == \
        ["dup", "miss", "lowconf", "coverage"]


def test_select_respects_quota_and_bucket():
    rows = [{"source_video": "v", "frame": i * 100, "conf": 0.9, "n_det": 2}
            for i in range(100)]                        # dup, well-spaced
    rows += [{"source_video": "v", "frame": 50000 + i * 100, "conf": 0.9, "n_det": 1}
             for i in range(100)]                       # coverage
    sel = sf.select_candidates(_df(rows), {"dup": 10, "coverage": 5},
                               lowconf_thresh=0.4, min_frame_gap=15, seed=1)
    assert (sel["bucket"] == "dup").sum() == 10
    assert (sel["bucket"] == "coverage").sum() == 5
    assert len(sel) == 15


def test_temporal_dedup_enforces_gap():
    # 5 frames all within 12 of each other; min_gap 15 keeps exactly one.
    df = _df([{"source_video": "v", "frame": f, "conf": 0.9, "n_det": 2, "score": 1.0}
              for f in (0, 3, 6, 9, 12)])
    assert len(sf.temporal_dedup(df, min_gap=15)) == 1


def test_zero_quota_selects_nothing():
    df = _df([{"source_video": "v", "frame": 0, "conf": 0.9, "n_det": 2}])
    assert len(sf.select_candidates(df, {"dup": 0}, 0.4, 15, 1)) == 0


def test_cross_bucket_dedup_drops_near_neighbours():
    """min_frame_gap must hold ACROSS buckets, not only inside each one."""
    df = _df([
        {"source_video": "v", "frame": 100, "bucket": "lowconf"},
        {"source_video": "v", "frame": 102, "bucket": "miss"},     # 2 frames later
        {"source_video": "v", "frame": 900, "bucket": "dup"},      # far away, kept
    ])
    out = sf.cross_bucket_dedup(df, min_gap=60)
    assert len(out) == 2
    # the rarer bucket wins the collision
    assert set(out["bucket"]) == {"miss", "dup"}


def test_cross_bucket_dedup_is_per_video():
    df = _df([
        {"source_video": "a", "frame": 100, "bucket": "dup"},
        {"source_video": "b", "frame": 101, "bucket": "dup"},   # different video
    ])
    assert len(sf.cross_bucket_dedup(df, min_gap=60)) == 2


def test_cross_bucket_dedup_noop_when_gap_zero():
    df = _df([{"source_video": "v", "frame": 1, "bucket": "dup"},
              {"source_video": "v", "frame": 2, "bucket": "dup"}])
    assert len(sf.cross_bucket_dedup(df, min_gap=0)) == 2


def test_shrink_keeps_unix_time_in_double_precision():
    """unix_time is a ~1.78e9 epoch; float32 would quantise it to ~100 s."""
    import numpy as np
    t = 1_780_105_357.0
    df = _df([{"unix_time": t, "conf": 0.5, "cx": 1.0, "cy": 2.0,
               "w": 3.0, "h": 4.0, "lum": 5.0, "fdiff": 6.0, "n_det": 1}])
    out = sf._shrink(df)
    assert out["unix_time"].dtype == "float64"
    assert float(out["unix_time"].iloc[0]) == t
    for c in ("conf", "cx", "cy", "w", "h", "lum", "fdiff"):
        assert out[c].dtype == "float32"
    # float32 really would have destroyed it -- guard the premise of this test
    assert float(np.float32(t)) != t


def test_shrink_tolerates_missing_columns():
    df = _df([{"conf": 0.5}])
    assert sf._shrink(df)["conf"].dtype == "float32"
