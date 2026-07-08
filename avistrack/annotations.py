"""
avistrack.annotations
──────────────────────
Shared annotation helpers that must be importable by the numbered CLI tools
in ``tools/`` (whose filenames start with a digit and therefore cannot be
imported as modules). Keep importable logic here, not in the numbered
scripts.
"""

from __future__ import annotations

import re

from avistrack.config.schema import SourcesConfig

# CVAT YOLO export uses _fNNNNNN as the per-clip frame index suffix.
FRAME_SUFFIX_RE = re.compile(r"^(.*)_f\d{6}$")


def parse_frame_name(stem: str, sources: SourcesConfig) -> tuple[str, str, str]:
    """
    Parse a CVAT frame stem like
    ``vr_105A_wave3_1_Wave3_VRChamber_..._s11645_transformed_f000044`` into
    ``(chamber_id, wave_id, clip_stem)``.

    Strategy: every (chamber_id, wave_id) registered in sources.yaml is
    tested as a prefix on ``stem``. The longest matching prefix wins —
    necessary because wave_ids contain underscores ("wave3", "wave3_1",
    "wave3_negsample") and a naive split would mis-attribute.
    """
    m = FRAME_SUFFIX_RE.match(stem)
    if not m:
        raise ValueError(
            f"frame {stem!r}: expected '..._fNNNNNN' suffix from extract_frames.py"
        )
    clip_with_chamber = m.group(1)

    candidates: list[tuple[str, str, int]] = []
    for ch in sources.chambers:
        for wv in ch.waves:
            prefix = f"{ch.chamber_id}_{wv.wave_id}_"
            if clip_with_chamber.startswith(prefix):
                candidates.append((ch.chamber_id, wv.wave_id, len(prefix)))
    if not candidates:
        raise ValueError(
            f"frame {stem!r}: no (chamber_id, wave_id) pair from sources.yaml "
            f"matches as a prefix"
        )
    candidates.sort(key=lambda x: -x[2])  # longest prefix wins
    chamber_id, wave_id, _ = candidates[0]
    return chamber_id, wave_id, clip_with_chamber
