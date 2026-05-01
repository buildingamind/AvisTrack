#!/usr/bin/env python3
"""
tools/show_lineage.py
─────────────────────
Reverse-lookup any artifact back to recipe → dataset → experiment → eval.

Usage
-----
    python tools/show_lineage.py {workspace}/models/{exp}/final/best.pt
    python tools/show_lineage.py {workspace}/datasets/{ds}/manifest.csv
    python tools/show_lineage.py /path/to/_meta.json
    python tools/show_lineage.py /path/to/eval_config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from avistrack import lineage as L


def _print_kv(prefix: str, key: str, value) -> None:
    if isinstance(value, dict):
        print(f"{prefix}{key}:")
        for k, v in value.items():
            _print_kv(prefix + "  ", k, v)
    elif isinstance(value, list):
        if not value:
            print(f"{prefix}{key}: []")
        else:
            print(f"{prefix}{key}:")
            for i, v in enumerate(value):
                _print_kv(prefix + "  ", f"[{i}]", v)
    else:
        print(f"{prefix}{key}: {value}")


def render(out: dict) -> None:
    kind = out.get("kind", "unknown")
    print(f"# {kind}")
    print(f"artifact: {out.get('artifact')}")

    if kind == "experiment_artifact":
        meta = out.get("experiment_meta", {})
        print(f"meta_path: {out.get('meta_path')}")
        print()
        for k in ["experiment_name", "chamber_type", "dataset_name",
                  "recipe_hash", "git_sha", "git_dirty",
                  "started_at", "ended_at", "final_weights",
                  "workspace_root"]:
            if k in meta:
                _print_kv("  ", k, meta[k])
    elif kind == "dataset":
        print(f"dataset_dir: {out.get('dataset_dir')}")
        print(f"recipe_hash: {out.get('recipe_hash', '')}")
        print()
        if "recipe" in out:
            _print_kv("", "recipe", out["recipe"])
        _print_kv("", "chambers", out.get("chambers", []))
        _print_kv("", "waves",    out.get("waves", []))
    elif kind == "eval":
        for k in ["experiment_name", "dataset_name", "split", "weights",
                  "git_sha", "started_at", "data_yaml"]:
            if k in out:
                _print_kv("  ", k, out[k])
        if "experiment_meta" in out:
            print()
            print("# experiment")
            for k in ["experiment_name", "chamber_type", "dataset_name",
                      "recipe_hash", "git_sha", "started_at", "ended_at"]:
                if k in out["experiment_meta"]:
                    _print_kv("  ", k, out["experiment_meta"][k])
    elif kind == "batch_output":
        for k, v in out.items():
            if k in ("kind", "artifact"):
                continue
            _print_kv("  ", k, v)
    else:
        print("(no lineage information could be resolved for this path)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("artifact", type=Path,
                   help="Path to a weights file, dataset, eval, or batch output.")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of pretty text.")
    args = p.parse_args()

    out = L.trace_lineage(args.artifact)
    if args.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        render(out)


if __name__ == "__main__":
    main()
