#!/usr/bin/env python3
"""
tools/register_chamber_source.py
────────────────────────────────
Bind one chamber's external drive to a workspace.

What the tool does:

1. Find the candidate drive – either ``--mount /path/to/drive`` is given,
   or all currently-mounted volumes are listed via
   :func:`avistrack.config.drive_probe.list_mounted_drives` and the user
   picks one.
2. Read (or compute) the volume UUID for that mount.
3. Write ``_avistrack_source.yaml`` to the drive root, anchoring it to
   the chamber_id / chamber_type pair.
4. Append (or update in-place) the chamber entry in
   ``{workspace_root}/{chamber_type}/sources.yaml``.

Re-running with the same ``--chamber-id`` is idempotent: the existing
chamber entry is rewritten with the latest drive UUID/label, and any
``waves:`` already registered are preserved.

Usage
-----
    # interactive – pick from mounted volumes
    python tools/register_chamber_source.py \
        --workspace-root /media/ssd/avistrack \
        --chamber-type collective \
        --chamber-id collective_104A

    # explicit mount + UUID
    python tools/register_chamber_source.py \
        --workspace-root /media/ssd/avistrack \
        --chamber-type collective \
        --chamber-id collective_104A \
        --mount /media/woodlab/104-A \
        --drive-uuid ABCD-1234 \
        --drive-label 104-A
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from avistrack.config import drive_probe, load_sources  # noqa: E402

SOURCE_MARKER_FILENAME = "_avistrack_source.yaml"


# ── Drive selection ─────────────────────────────────────────────────────

def _resolve_uuid_for_mount(mount: Path) -> Optional[str]:
    """Look up UUID of a known mount in the platform listing."""
    target = str(mount.resolve()).rstrip("/\\").lower()
    for vol in drive_probe.list_mounted_drives():
        m = str(Path(vol["mount"]).resolve()).rstrip("/\\").lower()
        if m == target:
            return drive_probe.normalize_uuid(vol["uuid"])
    return None


def _pick_mount_interactive() -> dict:
    """Show a numbered list of mounted volumes and let the user pick."""
    drives = drive_probe.list_mounted_drives()
    if not drives:
        raise SystemExit(
            "No mounted volumes detected. Pass --mount and --drive-uuid "
            "explicitly, or check that the chamber drive is plugged in."
        )

    print("Mounted volumes:")
    for i, vol in enumerate(drives, 1):
        label = vol.get("label") or "<no label>"
        print(f"  [{i}] {vol['mount']}  uuid={vol['uuid']}  label={label}")

    while True:
        choice = input(f"Pick a drive [1-{len(drives)}] (or q to quit): ").strip()
        if choice.lower() in {"q", "quit", ""}:
            raise SystemExit("aborted by user.")
        if choice.isdigit() and 1 <= int(choice) <= len(drives):
            return drives[int(choice) - 1]
        print("invalid selection.")


# ── On-drive marker ─────────────────────────────────────────────────────

def _write_drive_marker(
    mount: Path,
    chamber_id: str,
    chamber_type: str,
    drive_uuid: str,
    drive_label: Optional[str],
    force: bool,
) -> Path:
    """Write _avistrack_source.yaml to the drive root.

    If a marker already exists with the same chamber_id we overwrite it
    (timestamps and label may be refreshed). If the existing marker
    points at a *different* chamber_id, refuse unless --force.
    """
    marker_path = mount / SOURCE_MARKER_FILENAME
    if marker_path.exists():
        existing = yaml.safe_load(marker_path.read_text()) or {}
        existing_id = existing.get("chamber_id")
        if existing_id and existing_id != chamber_id and not force:
            raise SystemExit(
                f"{marker_path} already pinned to chamber_id={existing_id!r}; "
                f"refusing to retag as {chamber_id!r}. Re-run with --force "
                f"if this is really the right drive."
            )

    payload = {
        "chamber_id":   chamber_id,
        "chamber_type": chamber_type,
        "drive_uuid":   drive_uuid,
        "drive_label":  drive_label or "",
        "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    marker_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return marker_path


# ── sources.yaml update ─────────────────────────────────────────────────

def _upsert_chamber_entry(
    sources_yaml: Path,
    chamber_type: str,
    chamber_id: str,
    drive_uuid: str,
    drive_label: Optional[str],
) -> bool:
    """Add or update one chamber block. Returns True if a new entry was created."""
    raw = yaml.safe_load(sources_yaml.read_text()) or {}
    if raw.get("chamber_type") and raw["chamber_type"] != chamber_type:
        raise SystemExit(
            f"{sources_yaml} is for chamber_type={raw['chamber_type']!r}; "
            f"refusing to register a {chamber_type!r} chamber here."
        )
    raw["chamber_type"] = chamber_type
    chambers = raw.get("chambers") or []

    created = True
    for entry in chambers:
        if entry.get("chamber_id") == chamber_id:
            entry["drive_uuid"]  = drive_uuid
            entry["drive_label"] = drive_label or entry.get("drive_label", "")
            entry.setdefault("waves", [])
            created = False
            break
    else:
        chambers.append({
            "chamber_id":  chamber_id,
            "drive_uuid":  drive_uuid,
            "drive_label": drive_label or "",
            "waves":       [],
        })

    raw["chambers"] = chambers
    sources_yaml.write_text(yaml.safe_dump(raw, sort_keys=False))
    return created


# ── Entry point ─────────────────────────────────────────────────────────

def register(
    workspace_root: Path,
    chamber_type: str,
    chamber_id: str,
    mount: Optional[Path],
    drive_uuid: Optional[str],
    drive_label: Optional[str],
    force: bool,
) -> tuple[Path, Path, bool]:
    """Run the registration flow. Returns (marker_path, sources_yaml, created)."""
    sources_yaml = workspace_root / chamber_type / "sources.yaml"
    if not sources_yaml.exists():
        raise SystemExit(
            f"{sources_yaml} not found. Run init_chamber_workspace.py first."
        )

    if mount is None:
        picked = _pick_mount_interactive()
        mount = Path(picked["mount"])
        if drive_uuid is None:
            drive_uuid = drive_probe.normalize_uuid(picked["uuid"])
        if drive_label is None:
            drive_label = picked.get("label") or ""

    if drive_uuid is None:
        drive_uuid = _resolve_uuid_for_mount(mount)
    if drive_uuid is None:
        raise SystemExit(
            f"Could not determine a UUID for {mount}. Pass --drive-uuid explicitly."
        )
    drive_uuid = drive_probe.normalize_uuid(drive_uuid)

    if not mount.exists():
        raise SystemExit(f"mount path does not exist: {mount}")

    marker_path = _write_drive_marker(
        mount=mount,
        chamber_id=chamber_id,
        chamber_type=chamber_type,
        drive_uuid=drive_uuid,
        drive_label=drive_label,
        force=force,
    )
    created = _upsert_chamber_entry(
        sources_yaml=sources_yaml,
        chamber_type=chamber_type,
        chamber_id=chamber_id,
        drive_uuid=drive_uuid,
        drive_label=drive_label,
    )

    # Re-load (no probe needed) to validate schema after we rewrote the file.
    load_sources(sources_yaml, workspace_root=workspace_root, probe=False)

    return marker_path, sources_yaml, created


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace-root", required=True, type=Path)
    p.add_argument("--chamber-type",   required=True)
    p.add_argument("--chamber-id",     required=True,
                   help="Stable chamber identifier, e.g. collective_104A")
    p.add_argument("--mount", type=Path,
                   help="Drive mount point. If omitted, pick interactively.")
    p.add_argument("--drive-uuid",
                   help="Override volume UUID (e.g. when probe can't see it).")
    p.add_argument("--drive-label",
                   help="Friendly label written to the source marker.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing marker pinned to a different chamber_id.")
    args = p.parse_args()

    marker, sources_yaml, created = register(
        workspace_root=args.workspace_root,
        chamber_type=args.chamber_type,
        chamber_id=args.chamber_id,
        mount=args.mount,
        drive_uuid=args.drive_uuid,
        drive_label=args.drive_label,
        force=args.force,
    )

    verb = "Registered" if created else "Updated"
    print(f"{verb} chamber '{args.chamber_id}' for chamber_type='{args.chamber_type}'.")
    print(f"  drive marker  : {marker}")
    print(f"  workspace yaml: {sources_yaml}")
    print()
    print("Next: declare waves on this chamber by editing the 'waves:' list in")
    print(f"  {sources_yaml}")
    print("or via tools/scan_legacy_wave.py for old dumps without metadata.")


if __name__ == "__main__":
    main()
