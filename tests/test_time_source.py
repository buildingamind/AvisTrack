"""
tests/test_time_source.py — frame<->time from ChamberBroadcaster's capture log.

Covers avistrack/core/time_lookup.py: load_segment_starts (parse
timestamp_calibration.jsonl) and TimeLookup.from_segment_start (linear
frame->unix from a segment's wall-clock start, fps-cancelling).
"""

from __future__ import annotations

import json

from avistrack.core.time_lookup import TimeLookup, load_segment_starts


def test_load_segment_starts_strips_fragmented_and_skips_junk(tmp_path):
    p = tmp_path / "timestamp_calibration.jsonl"
    p.write_text("\n".join([
        json.dumps({"reason": "periodic_5min", "unix_s": 1000.0}),           # ignored
        json.dumps({"reason": "new_segment", "unix_s": 2000.0, "stream": "rgb",
                    "video_file": "FRAGMENTED_A_Cam1_RGB.mp4"}),
        json.dumps({"reason": "new_segment", "unix_s": 2000.5, "stream": "ir",
                    "video_file": "FRAGMENTED_A_Cam1_IR.mp4"}),
        json.dumps({"reason": "segment_close", "unix_s": 5000.0,
                    "video_file": "FRAGMENTED_A_Cam1_RGB.mp4"}),             # ignored
        json.dumps({"reason": "new_segment", "unix_s": 6000.0,
                    "video_file": "B_Cam1_RGB.mp4"}),                        # no prefix
        "not json", "",                                                      # tolerated
    ]), encoding="utf-8")
    assert load_segment_starts(p) == {
        "A_Cam1_RGB.mp4": 2000.0,
        "A_Cam1_IR.mp4": 2000.5,
        "B_Cam1_RGB.mp4": 6000.0,
    }


def test_load_segment_starts_first_wins(tmp_path):
    p = tmp_path / "ts.jsonl"
    p.write_text("\n".join([
        json.dumps({"reason": "new_segment", "unix_s": 100.0, "video_file": "V.mp4"}),
        json.dumps({"reason": "new_segment", "unix_s": 999.0, "video_file": "V.mp4"}),
    ]), encoding="utf-8")
    assert load_segment_starts(p) == {"V.mp4": 100.0}


def test_from_segment_start_maps_start_and_offset():
    fps, n = 30.0, 108000
    tl = TimeLookup.from_segment_start(1000.0, fps, n)
    assert abs(tl.frame_to_unix(0) - 1000.0) < 1e-6
    assert abs(tl.frame_to_unix(int(60 * fps)) - (1000.0 + 60)) < 1e-6


def test_from_segment_start_fps_cancels():
    # For any fps, frame = sec*fps then frame_to_unix must give start + sec.
    for fps in (29.962, 30.0, 31.941):
        tl = TimeLookup.from_segment_start(500.0, fps, 100000)
        frame = int(120 * fps)
        assert abs(tl.frame_to_unix(frame) - (500.0 + frame / fps)) < 1e-6
