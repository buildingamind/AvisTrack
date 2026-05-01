#!/usr/bin/env python3
"""
tools/edit_valid_ranges.py
──────────────────────────
Interactive terminal tool for recording valid wall-clock time ranges per video.

Requires ``time_calibration.json`` to display each video's span.
Stores results in ``valid_ranges.json``.  ``sample_clips.py`` reads this
file and **only** samples from within the defined ranges.

Subcommands
-----------
    add      Interactively add valid time ranges (default)
    list     Show all recorded valid ranges
    remove   Interactively remove valid ranges

Usage
-----
    # Add ranges (default subcommand):
    python tools/edit_valid_ranges.py --config configs/wave2_collective.yaml

    # Explicit subcommand:
    python tools/edit_valid_ranges.py add --config configs/wave2_collective.yaml

    # Show existing ranges:
    python tools/edit_valid_ranges.py list --config configs/wave2_collective.yaml

    # Remove ranges:
    python tools/edit_valid_ranges.py remove --config configs/wave2_collective.yaml
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))
from avistrack.config.loader import load_config
from avistrack.core.time_lookup import TimeLookup

DT_FMT = "%Y-%m-%d %H:%M:%S"
DT_SHORT = "%Y-%m-%d %H:%M"


def _resolve_paths(args) -> tuple[Path, Path, str]:
    """
    Resolve ``(time_calibration_path, valid_ranges_path, timezone)``.

    Two input modes:

    * **workspace mode** – ``--workspace-yaml --chamber-id --wave-id``
      uses :func:`avistrack.workspace.load_context`. Legacy waves get
      their valid_ranges.json under ``_avistrack_added/{wave_id}/``;
      structured waves under ``02_Chamber_Metadata/``. The timezone
      comes from ``workspace.yaml``'s ``time.timezone``.
    * **legacy --config** – pulls the same fields out of the old
      single-file YAML.

    Workspace mode is selected when any of its three flags is present;
    all three are then required.
    """
    workspace_yaml = getattr(args, "workspace_yaml", None)
    sources_yaml   = getattr(args, "sources_yaml",   None)
    chamber_id     = getattr(args, "chamber_id",     None)
    wave_id        = getattr(args, "wave_id",        None)
    config         = getattr(args, "config",         None)

    workspace_mode = bool(workspace_yaml or chamber_id or wave_id)
    if workspace_mode:
        if config:
            print("❌ --config cannot be combined with workspace-mode flags "
                  "(--workspace-yaml / --chamber-id / --wave-id)")
            sys.exit(1)
        missing = [
            name for name, val in (
                ("--workspace-yaml", workspace_yaml),
                ("--chamber-id",     chamber_id),
                ("--wave-id",        wave_id),
            ) if not val
        ]
        if missing:
            print(f"❌ workspace mode requires {', '.join(missing)}")
            sys.exit(1)

        from avistrack.workspace import load_context
        workspace_yaml = Path(workspace_yaml)
        sources_yaml = Path(sources_yaml) if sources_yaml else \
            workspace_yaml.with_name("sources.yaml")
        if not sources_yaml.exists():
            print(f"❌ sources.yaml not found at {sources_yaml}")
            sys.exit(1)
        ctx = load_context(
            workspace_yaml=workspace_yaml,
            sources_yaml=sources_yaml,
            chamber_id=chamber_id, wave_id=wave_id,
            require_drive=True,
        )
        return (ctx.time_calibration_file,
                ctx.valid_ranges_file,
                ctx.workspace.time.timezone)

    if not config:
        print("❌ --config or workspace-mode flags are required")
        sys.exit(1)
    cfg = load_config(config)
    return Path(cfg.drive.time_calibration), Path(cfg.drive.valid_ranges), cfg.time.timezone

# Formats accepted for user input (tried in order)
_INPUT_FMTS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %I:%M %p",
    "%Y-%m-%d %I:%M %p",
    "%Y-%m-%d %I:%M:%S %p",
]


# ── Helpers ───────────────────────────────────────────────────────────────

def parse_user_time(text: str, tz: ZoneInfo) -> datetime | None:
    """Try multiple strptime formats to parse user input → tz-aware dt."""
    text = text.strip()
    for fmt in _INPUT_FMTS:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=tz)
        except ValueError:
            continue
    return None


def load_valid_ranges(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_valid_ranges(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n✅ Saved → {path}")


def load_calibration(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def get_video_spans(calibration: dict, tz_str: str) -> dict:
    """Return {video_name: (start_dt, end_dt, hours)}."""
    spans = {}
    for vname, vdata in calibration.items():
        if vname == "_meta":
            continue
        samples = vdata.get("samples", [])
        if len(samples) < 2:
            continue
        lookup = TimeLookup.from_calibration(calibration, vname, tz_str)
        start_dt = lookup.frame_to_datetime(0)
        end_dt = lookup.frame_to_datetime(vdata["total_frames"] - 1)
        hours = (end_dt - start_dt).total_seconds() / 3600
        spans[vname] = (start_dt, end_dt, hours)
    return spans


# ── Subcommand: add ──────────────────────────────────────────────────────

def cmd_add(args):
    cal_path, vr_path, tz_str = _resolve_paths(args)
    tz = ZoneInfo(tz_str)

    if not cal_path.exists():
        print(f"❌ time_calibration.json not found: {cal_path}")
        print("   Run `calibrate_time.py calibrate` first.")
        sys.exit(1)

    calibration = load_calibration(cal_path)
    spans = get_video_spans(calibration, tz_str)
    if not spans:
        print("❌ No calibrated videos with ≥ 2 samples.")
        sys.exit(1)

    ranges = load_valid_ranges(vr_path)
    video_list = sorted(spans.keys())

    while True:
        print()
        print("📂 Calibrated videos:")
        for i, vname in enumerate(video_list, 1):
            start_dt, end_dt, hours = spans[vname]
            n_existing = len(ranges.get(vname, []))
            tag = f"  ({n_existing} range{'s' if n_existing != 1 else ''})" if n_existing else ""
            print(f"  {i}. {vname}   "
                  f"{start_dt.strftime(DT_FMT)} → "
                  f"{end_dt.strftime(DT_FMT)}  ({hours:.1f}h){tag}")

        sel = input(f"\nSelect video [1-{len(video_list)}], or 'done': ").strip()
        if sel.lower() in ("done", "d", "q", ""):
            break
        try:
            idx = int(sel) - 1
            vname = video_list[idx]
        except (ValueError, IndexError):
            print("  Invalid selection.")
            continue

        start_dt, end_dt, _ = spans[vname]
        print(f"\n  {vname}")
        print(f"  Span: {start_dt.strftime(DT_FMT)} → {end_dt.strftime(DT_FMT)}")

        # Show existing ranges
        if vname in ranges and ranges[vname]:
            print("  Existing ranges:")
            for r in ranges[vname]:
                print(f"    • {r['start']} → {r['end']}")

        # Add new ranges
        while True:
            print()
            s_input = input(
                f"  Start (e.g. {start_dt.strftime(DT_SHORT)}), "
                f"or 'done': "
            ).strip()
            if s_input.lower() in ("done", "d", "q", ""):
                break

            s_dt = parse_user_time(s_input, tz)
            if not s_dt:
                print(f"  ❌ Cannot parse '{s_input}'.  "
                      f"Format: YYYY-MM-DD HH:MM")
                continue

            e_input = input(
                f"  End   (e.g. {end_dt.strftime(DT_SHORT)}): "
            ).strip()
            e_dt = parse_user_time(e_input, tz)
            if not e_dt:
                print(f"  ❌ Cannot parse '{e_input}'.  "
                      f"Format: YYYY-MM-DD HH:MM")
                continue

            if e_dt <= s_dt:
                print("  ❌ End must be after start.")
                continue

            hours = (e_dt - s_dt).total_seconds() / 3600
            ranges.setdefault(vname, []).append({
                "start": s_dt.strftime(DT_FMT),
                "end":   e_dt.strftime(DT_FMT),
            })
            print(f"  ✅ Added: {s_dt.strftime(DT_FMT)} → "
                  f"{e_dt.strftime(DT_FMT)}  ({hours:.1f}h)")

    save_valid_ranges(vr_path, ranges)


# ── Subcommand: list ─────────────────────────────────────────────────────

def cmd_list(args):
    _, vr_path, _ = _resolve_paths(args)

    if not vr_path.exists():
        print(f"ℹ️  No valid_ranges.json found at {vr_path}")
        print("   Run `add` to create one.")
        return

    ranges = load_valid_ranges(vr_path)
    if not ranges:
        print("ℹ️  valid_ranges.json is empty.")
        return

    print(f"\n📋 Valid ranges ({vr_path}):\n")
    for vname, vranges in sorted(ranges.items()):
        print(f"  {vname}:")
        if not vranges:
            print("    (none)")
        for i, r in enumerate(vranges, 1):
            start = datetime.fromisoformat(r["start"])
            end   = datetime.fromisoformat(r["end"])
            hours = (end - start).total_seconds() / 3600
            print(f"    {i}. {r['start']} → {r['end']}  ({hours:.1f}h)")

    total = sum(len(v) for v in ranges.values())
    print(f"\n  Total: {total} range(s) across {len(ranges)} video(s)")


# ── Subcommand: remove ───────────────────────────────────────────────────

def cmd_remove(args):
    _, vr_path, _ = _resolve_paths(args)

    if not vr_path.exists():
        print("ℹ️  No valid_ranges.json found.")
        return

    ranges = load_valid_ranges(vr_path)
    if not ranges:
        print("ℹ️  No ranges to remove.")
        return

    video_list = sorted(ranges.keys())
    print("\n📋 Videos with valid ranges:")
    for i, vname in enumerate(video_list, 1):
        n = len(ranges[vname])
        print(f"  {i}. {vname}  ({n} range{'s' if n != 1 else ''})")

    sel = input(f"\nSelect video [1-{len(video_list)}]: ").strip()
    try:
        idx = int(sel) - 1
        vname = video_list[idx]
    except (ValueError, IndexError):
        print("  Invalid selection.")
        return

    vranges = ranges[vname]
    print(f"\n  {vname}:")
    for i, r in enumerate(vranges, 1):
        start = datetime.fromisoformat(r["start"])
        end   = datetime.fromisoformat(r["end"])
        hours = (end - start).total_seconds() / 3600
        print(f"    {i}. {r['start']} → {r['end']}  ({hours:.1f}h)")

    sel = input(
        f"\n  Remove which range [1-{len(vranges)}], or 'all': "
    ).strip()
    if sel.lower() == "all":
        del ranges[vname]
        print(f"  ✅ Removed all ranges for {vname}")
    else:
        try:
            ri = int(sel) - 1
            removed = vranges.pop(ri)
            print(f"  ✅ Removed: {removed['start']} → {removed['end']}")
            if not vranges:
                del ranges[vname]
        except (ValueError, IndexError):
            print("  Invalid selection.")
            return

    save_valid_ranges(vr_path, ranges)


# ── Entry point ──────────────────────────────────────────────────────────

def _add_resolve_args(p, *, config_required: bool) -> None:
    """Wire up the path-resolution flags shared by every subcommand."""
    # config is mutually exclusive with workspace-mode flags but argparse
    # doesn't model that cleanly across subcommands; _resolve_paths
    # validates the combination at runtime instead.
    p.add_argument("--config", default=None,
                   required=config_required,
                   help="Legacy AvisTrack YAML config")
    p.add_argument("--workspace-yaml", default=None,
                   help="Path to {workspace_root}/{chamber_type}/workspace.yaml "
                        "(workspace mode — pair with --chamber-id and --wave-id).")
    p.add_argument("--sources-yaml",   default=None,
                   help="Path to sources.yaml (default: sibling of workspace.yaml).")
    p.add_argument("--chamber-id",     default=None,
                   help="Chamber id from sources.yaml (workspace mode).")
    p.add_argument("--wave-id",        default=None,
                   help="Wave id from sources.yaml (workspace mode).")


def main():
    parser = argparse.ArgumentParser(
        description="Record valid time ranges per video.")
    sub = parser.add_subparsers(dest="cmd")

    for name, helptext in [
        ("add",    "Add valid time ranges (default)"),
        ("list",   "Show all recorded ranges"),
        ("remove", "Remove valid ranges"),
    ]:
        p = sub.add_parser(name, help=helptext)
        _add_resolve_args(p, config_required=False)

    # Top-level mirrors so `python edit_valid_ranges.py --config X` (or the
    # workspace-mode equivalent) defaults to the 'add' subcommand.
    _add_resolve_args(parser, config_required=False)

    args = parser.parse_args()

    if not args.cmd:
        has_workspace = bool(args.workspace_yaml or args.chamber_id or args.wave_id)
        if args.config or has_workspace:
            args.cmd = "add"
        else:
            parser.print_help()
            sys.exit(1)

    {"add": cmd_add, "list": cmd_list, "remove": cmd_remove}[args.cmd](args)


if __name__ == "__main__":
    main()
