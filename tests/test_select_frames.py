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
