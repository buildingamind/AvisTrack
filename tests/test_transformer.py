"""
tests/test_transformer.py – smoke tests for PerspectiveTransformer.
Run with: pytest tests/
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from avistrack.core.transformer import PerspectiveTransformer


SAMPLE_CORNERS = [[315, 61], [889, 63], [870, 629], [327, 622]]


def test_transform_output_size_fixed():
    tf    = PerspectiveTransformer(SAMPLE_CORNERS, target_size=(640, 640))
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    out   = tf.transform(dummy)
    assert out.shape == (640, 640, 3), f"Expected (640,640,3), got {out.shape}"


def test_transform_output_size_auto():
    tf    = PerspectiveTransformer(SAMPLE_CORNERS)
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    out   = tf.transform(dummy)
    assert out.ndim == 3
    assert out.shape[2] == 3


def test_from_roi_file():
    rois = {"test_video.mkv": SAMPLE_CORNERS}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(rois, f)
        tmp_path = f.name

    tf = PerspectiveTransformer.from_roi_file(tmp_path, "test_video.mkv", target_size=(640, 640))
    assert tf.output_size == (640, 640)
    Path(tmp_path).unlink()


def test_from_roi_file_missing_key():
    rois = {"other_video.mkv": SAMPLE_CORNERS}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(rois, f)
        tmp_path = f.name

    with pytest.raises(KeyError):
        PerspectiveTransformer.from_roi_file(tmp_path, "missing.mkv")
    Path(tmp_path).unlink()
