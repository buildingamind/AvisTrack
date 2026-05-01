"""
tests/test_workspace_bootstrap.py – Step B: workspace skeleton + drive registration.

Covers tools/init_chamber_workspace.py and tools/register_chamber_source.py.
The tools are imported as modules so we can exercise their pure functions
without spawning subprocesses (and without touching real removable drives).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_tool(filename: str):
    """Import tools/<filename> as a module (the dir isn't a package)."""
    path = REPO_ROOT / "tools" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


# ── init_chamber_workspace ──────────────────────────────────────────────

def test_init_creates_tree_and_valid_yamls(tmp_path: Path):
    init_tool = _load_tool("init_chamber_workspace.py")
    workspace_root = tmp_path / "wkspc"

    chamber_dir = init_tool.init_workspace(
        workspace_root=workspace_root,
        chamber_type="collective",
        n_subjects=9,
        fps=30,
        target_size=[640, 640],
        timezone="America/New_York",
        force=False,
    )

    assert chamber_dir == workspace_root / "collective"
    for sub in init_tool.WORKSPACE_SUBDIRS:
        assert (chamber_dir / sub).is_dir(), f"missing {sub}/"
    assert (chamber_dir / "workspace.yaml").is_file()
    assert (chamber_dir / "sources.yaml").is_file()

    # workspace.yaml must round-trip through load_workspace and reflect overrides.
    from avistrack.config import load_workspace, load_sources
    wcfg = load_workspace(chamber_dir / "workspace.yaml")
    assert wcfg.chamber_type == "collective"
    assert wcfg.chamber.n_subjects == 9
    assert wcfg.chamber.fps == 30
    expected_root = (workspace_root / "collective").as_posix()
    assert Path(wcfg.workspace.root).as_posix() == expected_root
    assert Path(wcfg.workspace.clips).as_posix() == f"{expected_root}/clips"

    # sources.yaml is empty but well-formed.
    scfg = load_sources(chamber_dir / "sources.yaml", probe=False)
    assert scfg.chamber_type == "collective"
    assert scfg.chambers == []


def test_init_refuses_double_init_without_force(tmp_path: Path):
    init_tool = _load_tool("init_chamber_workspace.py")
    workspace_root = tmp_path / "wkspc"

    init_tool.init_workspace(
        workspace_root=workspace_root, chamber_type="collective",
        n_subjects=9, fps=30, target_size=[640, 640],
        timezone="America/New_York", force=False,
    )
    with pytest.raises(SystemExit, match="already initialised"):
        init_tool.init_workspace(
            workspace_root=workspace_root, chamber_type="collective",
            n_subjects=9, fps=30, target_size=[640, 640],
            timezone="America/New_York", force=False,
        )


def test_init_refuses_inside_git_repo(tmp_path: Path):
    init_tool = _load_tool("init_chamber_workspace.py")
    fake_repo = tmp_path / "repo"
    (fake_repo / ".git").mkdir(parents=True)
    with pytest.raises(SystemExit, match="git repo"):
        init_tool.init_workspace(
            workspace_root=fake_repo / "wkspc",
            chamber_type="collective",
            n_subjects=9, fps=30, target_size=[640, 640],
            timezone="America/New_York", force=False,
        )


# ── register_chamber_source ─────────────────────────────────────────────

def _bootstrap_workspace(tmp_path: Path, chamber_type: str = "collective") -> Path:
    init_tool = _load_tool("init_chamber_workspace.py")
    workspace_root = tmp_path / "wkspc"
    init_tool.init_workspace(
        workspace_root=workspace_root, chamber_type=chamber_type,
        n_subjects=9, fps=30, target_size=[640, 640],
        timezone="America/New_York", force=False,
    )
    return workspace_root


def test_register_writes_marker_and_appends_chamber(tmp_path: Path, monkeypatch):
    workspace_root = _bootstrap_workspace(tmp_path)
    register = _load_tool("register_chamber_source.py")

    fake_mount = tmp_path / "drive-A"
    fake_mount.mkdir()

    # No probing needed – we pass mount + uuid explicitly.
    marker, sources_yaml, created = register.register(
        workspace_root=workspace_root,
        chamber_type="collective",
        chamber_id="collective_104A",
        mount=fake_mount,
        drive_uuid="abcd-1234",
        drive_label="104-A",
        force=False,
    )

    assert created is True
    # Marker on the drive root.
    assert marker == fake_mount / register.SOURCE_MARKER_FILENAME
    payload = yaml.safe_load(marker.read_text())
    assert payload["chamber_id"]   == "collective_104A"
    assert payload["chamber_type"] == "collective"
    assert payload["drive_uuid"]   == "ABCD-1234"  # normalized
    assert payload["drive_label"]  == "104-A"
    assert "registered_at" in payload

    # Workspace sources.yaml updated.
    raw = yaml.safe_load(sources_yaml.read_text())
    assert raw["chamber_type"] == "collective"
    assert len(raw["chambers"]) == 1
    entry = raw["chambers"][0]
    assert entry["chamber_id"]  == "collective_104A"
    assert entry["drive_uuid"]  == "ABCD-1234"
    assert entry["drive_label"] == "104-A"
    assert entry["waves"]       == []


def test_register_is_idempotent_and_preserves_waves(tmp_path: Path):
    workspace_root = _bootstrap_workspace(tmp_path)
    register = _load_tool("register_chamber_source.py")

    fake_mount = tmp_path / "drive-A"
    fake_mount.mkdir()
    register.register(
        workspace_root=workspace_root, chamber_type="collective",
        chamber_id="collective_104A", mount=fake_mount,
        drive_uuid="abcd-1234", drive_label="104-A", force=False,
    )

    # Manually add a wave so we can prove re-registration keeps it.
    sources_yaml = workspace_root / "collective" / "sources.yaml"
    raw = yaml.safe_load(sources_yaml.read_text())
    raw["chambers"][0]["waves"] = [{
        "wave_id": "wave2", "layout": "structured",
        "wave_subpath": "Wave2",
        "raw_videos_subpath": "{wave_subpath}/00_raw_videos",
        "metadata_subpath":   "{wave_subpath}/02_Chamber_Metadata",
    }]
    sources_yaml.write_text(yaml.safe_dump(raw, sort_keys=False))

    # Re-register the same chamber with a new label.
    _, _, created = register.register(
        workspace_root=workspace_root, chamber_type="collective",
        chamber_id="collective_104A", mount=fake_mount,
        drive_uuid="abcd-1234", drive_label="104-A v2",
        force=False,
    )
    assert created is False

    raw2 = yaml.safe_load(sources_yaml.read_text())
    assert len(raw2["chambers"]) == 1
    entry = raw2["chambers"][0]
    assert entry["drive_label"] == "104-A v2"
    assert len(entry["waves"]) == 1
    assert entry["waves"][0]["wave_id"] == "wave2"


def test_register_refuses_marker_pinned_to_other_chamber(tmp_path: Path):
    workspace_root = _bootstrap_workspace(tmp_path)
    register = _load_tool("register_chamber_source.py")

    fake_mount = tmp_path / "drive-A"
    fake_mount.mkdir()
    # Pre-existing marker pinned to a different chamber_id.
    (fake_mount / register.SOURCE_MARKER_FILENAME).write_text(yaml.safe_dump({
        "chamber_id": "collective_999X", "chamber_type": "collective",
        "drive_uuid": "ABCD-1234",
    }))

    with pytest.raises(SystemExit, match="already pinned"):
        register.register(
            workspace_root=workspace_root, chamber_type="collective",
            chamber_id="collective_104A", mount=fake_mount,
            drive_uuid="abcd-1234", drive_label="104-A", force=False,
        )


def test_register_rejects_chamber_type_mismatch(tmp_path: Path):
    """If sources.yaml has been hand-edited so its chamber_type field
    disagrees with the directory we register into, refuse rather than
    silently overwriting it."""
    workspace_root = _bootstrap_workspace(tmp_path, chamber_type="vr")
    register = _load_tool("register_chamber_source.py")

    # Poison the file to simulate a manual edit / wrong chamber_type.
    sources_yaml = workspace_root / "vr" / "sources.yaml"
    sources_yaml.write_text(yaml.safe_dump(
        {"chamber_type": "collective", "chambers": []}, sort_keys=False,
    ))

    fake_mount = tmp_path / "drive-X"
    fake_mount.mkdir()
    with pytest.raises(SystemExit, match="refusing to register"):
        register.register(
            workspace_root=workspace_root, chamber_type="vr",
            chamber_id="vr_201", mount=fake_mount,
            drive_uuid="0000-1111", drive_label="VR-201", force=False,
        )


def test_register_requires_initialized_workspace(tmp_path: Path):
    register = _load_tool("register_chamber_source.py")
    fake_mount = tmp_path / "drive-A"
    fake_mount.mkdir()
    with pytest.raises(SystemExit, match="not found"):
        register.register(
            workspace_root=tmp_path / "uninit",
            chamber_type="collective", chamber_id="collective_104A",
            mount=fake_mount, drive_uuid="abcd-1234",
            drive_label="104-A", force=False,
        )
