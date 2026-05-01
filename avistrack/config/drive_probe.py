"""
Cross-platform helper for locating a chamber drive by its UUID.

Each chamber writes to a dedicated external drive identified in
``sources.yaml`` by ``drive_uuid``. The UUID is the filesystem volume
identifier:

* Windows – ``VolumeSerialNumber`` formatted ``XXXX-XXXX``
  (e.g. ``ABCD-1234``), as reported by ``Get-Volume`` /
  ``Win32_LogicalDisk``.
* Linux – the filesystem UUID under ``/dev/disk/by-uuid/`` (the same
  string ``blkid`` prints for ``UUID=``).

Comparison is case-insensitive and tolerates surrounding whitespace.
"""

from __future__ import annotations

import platform
import re
import subprocess
from pathlib import Path
from typing import Optional


def normalize_uuid(uuid: str) -> str:
    """Canonicalise a UUID for equality checks."""
    return uuid.strip().upper()


def probe_drive_mount(drive_uuid: str) -> Optional[Path]:
    """
    Return the current mount point of the volume with the given UUID, or
    ``None`` if the drive is not mounted.

    No exception is raised when the underlying probe command fails; the
    caller decides whether a missing chamber drive is fatal.
    """
    if not drive_uuid:
        return None

    target = normalize_uuid(drive_uuid)
    system = platform.system()

    try:
        if system == "Windows":
            return _probe_windows(target)
        if system == "Linux":
            return _probe_linux(target)
        if system == "Darwin":
            return _probe_macos(target)
    except (OSError, subprocess.SubprocessError):
        return None

    return None


def list_mounted_drives() -> list[dict]:
    """
    Enumerate currently-mounted volumes as ``{uuid, mount, label}`` dicts.

    Used by ``tools/register_chamber_source.py`` to let the user pick a
    drive interactively. Returns ``[]`` if the platform probe fails.
    """
    system = platform.system()
    try:
        if system == "Windows":
            return _list_windows()
        if system == "Linux":
            return _list_linux()
        if system == "Darwin":
            return _list_macos()
    except (OSError, subprocess.SubprocessError):
        return []
    return []


# ── Windows ───────────────────────────────────────────────────────────────

_PS_LIST_VOLUMES = (
    "Get-CimInstance Win32_LogicalDisk | "
    "Where-Object { $_.VolumeSerialNumber } | "
    "ForEach-Object { '{0}|{1}|{2}' -f $_.DeviceID, $_.VolumeSerialNumber, $_.VolumeName }"
)


def _run_powershell(script: str) -> str:
    out = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, timeout=15,
    )
    return out.stdout


def _format_windows_serial(raw: str) -> str:
    """Win32 reports the serial as 8 hex chars (no dash). Format ABCD-1234."""
    s = raw.strip().upper()
    if len(s) == 8 and re.fullmatch(r"[0-9A-F]{8}", s):
        return f"{s[:4]}-{s[4:]}"
    return s


def _list_windows() -> list[dict]:
    out = _run_powershell(_PS_LIST_VOLUMES)
    result = []
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 2:
            continue
        device, serial = parts[0], parts[1]
        label = parts[2] if len(parts) > 2 else ""
        if not device or not serial:
            continue
        result.append({
            "uuid":  _format_windows_serial(serial),
            "mount": device + "\\",     # "E:\"
            "label": label,
        })
    return result


def _probe_windows(target_uuid: str) -> Optional[Path]:
    for vol in _list_windows():
        if normalize_uuid(vol["uuid"]) == target_uuid:
            return Path(vol["mount"])
    return None


# ── Linux ─────────────────────────────────────────────────────────────────

def _list_linux() -> list[dict]:
    """Use ``lsblk`` to enumerate volumes with UUID + mount + label."""
    out = subprocess.run(
        ["lsblk", "-o", "UUID,MOUNTPOINT,LABEL", "-n", "-P"],
        capture_output=True, text=True, timeout=10,
    )
    result = []
    pattern = re.compile(r'(\w+)="([^"]*)"')
    for line in out.stdout.splitlines():
        fields = dict(pattern.findall(line))
        uuid = fields.get("UUID", "")
        mount = fields.get("MOUNTPOINT", "")
        if uuid and mount:
            result.append({
                "uuid":  uuid,
                "mount": mount,
                "label": fields.get("LABEL", ""),
            })
    return result


def _probe_linux(target_uuid: str) -> Optional[Path]:
    # Fast path: /dev/disk/by-uuid/<uuid> is a symlink to the device node.
    by_uuid = Path("/dev/disk/by-uuid")
    if by_uuid.exists():
        for entry in by_uuid.iterdir():
            if normalize_uuid(entry.name) == target_uuid:
                # Got the device node; find its mountpoint via lsblk.
                break

    for vol in _list_linux():
        if normalize_uuid(vol["uuid"]) == target_uuid:
            return Path(vol["mount"])
    return None


# ── macOS (best-effort, for dev) ──────────────────────────────────────────

def _list_macos() -> list[dict]:
    out = subprocess.run(
        ["diskutil", "info", "-all"],
        capture_output=True, text=True, timeout=15,
    )
    blocks = out.stdout.split("**********\n")
    result = []
    for block in blocks:
        uuid = mount = label = ""
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("Volume UUID:"):
                uuid = line.split(":", 1)[1].strip()
            elif line.startswith("Mount Point:"):
                mount = line.split(":", 1)[1].strip()
            elif line.startswith("Volume Name:"):
                label = line.split(":", 1)[1].strip()
        if uuid and mount:
            result.append({"uuid": uuid, "mount": mount, "label": label})
    return result


def _probe_macos(target_uuid: str) -> Optional[Path]:
    for vol in _list_macos():
        if normalize_uuid(vol["uuid"]) == target_uuid:
            return Path(vol["mount"])
    return None
