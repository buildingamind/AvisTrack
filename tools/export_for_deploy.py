#!/usr/bin/env python3
"""
tools/export_for_deploy.py
──────────────────────────
Export a YOLO ``.pt`` to a TensorRT FP16 ``.engine`` and tag the filename with
the build GPU's identity.  Must be run on the **production PC** that will
actually load the engine — TensorRT engines are not portable across CUDA
compute capabilities.

GPU tag format
--------------
``<short_name>_sm<major><minor>``  e.g.  ``rtx3070_sm86``  ``rtx2080_sm75``

If the same prod PC ever hosts multiple GPUs of different generations,
multiple ``.engine`` files can co-exist next to ``best.pt``;
``YoloKalmanProcessor`` will pick the one matching the runtime GPU.

Usage
-----
    python tools/export_for_deploy.py \\
        --weights /mnt/hdd/Wave3_test/03_Model_Training/champion/best.pt \\
        --imgsz   640
"""

import argparse
import re
import shutil
import sys
from pathlib import Path


def detect_gpu_tag() -> str:
    try:
        import torch
    except ImportError:
        print("❌ torch not installed — cannot detect GPU.")
        sys.exit(1)
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This script must run on a machine with an NVIDIA GPU.")
        sys.exit(1)
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    short = re.sub(r"[^a-z0-9]+", "", name.lower().replace("nvidia", "").replace("geforce", ""))
    return f"{short}_sm{major}{minor}"


def main():
    ap = argparse.ArgumentParser(description="Export YOLO .pt → TensorRT FP16 .engine, GPU-tagged.")
    ap.add_argument("--weights", required=True, type=Path,
                    help="Path to champion best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--no-half", action="store_true",
                    help="Disable FP16 (default: FP16 enabled)")
    args = ap.parse_args()

    if not args.weights.exists():
        print(f"❌ Weights not found: {args.weights}")
        sys.exit(1)

    gpu_tag = detect_gpu_tag()
    print(f"🖥️  GPU tag: {gpu_tag}")

    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    print(f"📦 Exporting {args.weights.name} → engine "
          f"(imgsz={args.imgsz}, half={not args.no_half}, device={args.device})")
    exported_path = model.export(
        format="engine",
        half=not args.no_half,
        imgsz=args.imgsz,
        device=args.device,
    )

    # Ultralytics returns the .engine path as a string; rename to embed the GPU tag.
    src = Path(exported_path)
    if not src.exists():
        # Some Ultralytics versions return None — fall back to convention next to .pt
        src = args.weights.with_suffix(".engine")
    if not src.exists():
        print(f"❌ Could not locate exported engine (expected near {args.weights})")
        sys.exit(1)

    tagged = args.weights.parent / f"{args.weights.stem}_{gpu_tag}.engine"
    if src.resolve() != tagged.resolve():
        shutil.move(str(src), str(tagged))
    print(f"\n✅ Engine written: {tagged}")
    print(f"   Point ChamberBroadcaster YoloKalmanProcessor.model_path here, or to {args.weights} "
          f"(it'll auto-prefer the matching engine in the same dir).")


if __name__ == "__main__":
    main()
