"""avistrack/champion.py — champion selection from a phase bakeoff.

Pure, dependency-light selection logic (no ultralytics / torch import) so it is
unit-testable. The heavy evaluation that *produces* the rows lives in the CLI
entry point ``eval/select_champion.py``.

Selection policy (reproduces the pre-w4 convention)
---------------------------------------------------
* Primary criterion: highest ``metric_key`` (default ``mAP50-95``) on the test
  split.
* Speed tiebreak: if one or more *other* runs land within ``tiebreak_pts`` of
  the top score, prefer the **fastest** (highest ``speed_key``) among that
  near-tie band. A model that is statistically indistinguishable in accuracy
  but faster is the better deployment champion.
* ``tiebreak_applied`` is True only when the tiebreak actually changed the
  winner (i.e. the chosen run is not the raw metric-argmax).
"""
from __future__ import annotations

from typing import Optional


def _f(row: dict, key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def select_champion(
    rows: list[dict],
    metric_key: str = "mAP50-95",
    tiebreak_pts: float = 0.01,
    speed_key: str = "gpu_fps",
) -> tuple[Optional[dict], bool]:
    """Return ``(champion_row, tiebreak_applied)``.

    ``champion_row`` is ``None`` when ``rows`` is empty. Rows are dicts that
    must carry ``metric_key`` and (for the tiebreak) ``speed_key`` and
    ``run_name``.
    """
    if not rows:
        return None, False

    top = max(rows, key=lambda r: _f(r, metric_key))
    top_score = _f(top, metric_key)

    band = [r for r in rows if _f(r, metric_key) >= top_score - tiebreak_pts]
    champ = max(band, key=lambda r: _f(r, speed_key))

    tiebreak_applied = champ.get("run_name") != top.get("run_name")
    return champ, tiebreak_applied
