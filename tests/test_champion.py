"""tests/test_champion.py — champion selection policy."""
from __future__ import annotations

from avistrack.champion import select_champion


def _row(name, mAP, fps):
    return {"run_name": name, "mAP50-95": mAP, "gpu_fps": fps}


def test_empty():
    assert select_champion([]) == (None, False)


def test_clear_winner_no_tiebreak():
    # Top mAP is also (comfortably) ahead → argmax wins, no tiebreak.
    rows = [_row("a", 0.77, 140), _row("b", 0.74, 160), _row("c", 0.72, 120)]
    champ, applied = select_champion(rows, tiebreak_pts=0.01)
    assert champ["run_name"] == "a"
    assert applied is False


def test_prew4_shape_argmax_is_also_fastest_enough():
    # Mirrors pre-w4: yolo8s (0.7725, 143fps) beats yolo8m (0.7710, 75fps).
    # They are within 0.01, but the argmax is ALSO the faster one → no tiebreak.
    rows = [_row("yolo8s", 0.7725, 143.6), _row("yolo8m", 0.7710, 75.8)]
    champ, applied = select_champion(rows, tiebreak_pts=0.01)
    assert champ["run_name"] == "yolo8s"
    assert applied is False


def test_speed_tiebreak_changes_winner():
    # A slightly-lower-mAP but much faster model within the 0.01 band wins.
    rows = [_row("slow_top", 0.7725, 70), _row("fast_near", 0.7700, 150)]
    champ, applied = select_champion(rows, tiebreak_pts=0.01)
    assert champ["run_name"] == "fast_near"
    assert applied is True


def test_band_excludes_outside_margin():
    # fast_far is faster but 0.02 below top → outside the 0.01 band, ignored.
    rows = [_row("top", 0.7725, 80), _row("fast_far", 0.7500, 200)]
    champ, applied = select_champion(rows, tiebreak_pts=0.01)
    assert champ["run_name"] == "top"
    assert applied is False
