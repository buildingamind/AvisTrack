"""
tests/test_config.py – smoke tests for config loading.
"""

import tempfile
from pathlib import Path

import pytest

from avistrack.config.loader import load_config


MINIMAL_YAML = """
experiment: "test"
model:
  backend: "yolo"
  weights: "/fake/path/best.pt"
chamber:
  n_subjects: 9
"""


def test_load_minimal_config():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(MINIMAL_YAML)
        tmp = f.name
    cfg = load_config(tmp)
    assert cfg.experiment == "test"
    assert cfg.model.backend == "yolo"
    assert cfg.chamber.n_subjects == 9
    Path(tmp).unlink()


def test_root_placeholder_resolution():
    yaml_content = """
experiment: "test"
drive:
  root: "/media/drive/Wave3"
  roi_file: "{root}/02_Global_Metadata/camera_rois.json"
model:
  backend: "yolo"
  weights: "{root}/03_Model_Training/best.pt"
chamber:
  n_subjects: 9
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp = f.name
    cfg = load_config(tmp)
    assert cfg.drive.roi_file == "/media/drive/Wave3/02_Global_Metadata/camera_rois.json"
    assert cfg.model.weights  == "/media/drive/Wave3/03_Model_Training/best.pt"
    Path(tmp).unlink()
