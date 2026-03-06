"""
AvisTrack – model-agnostic tracking engine for avian behavioral experiments.

Quick start
-----------
    import avistrack
    tracker = avistrack.load_tracker("configs/wave3_collective.yaml")
    result  = tracker.update(frame)          # works for both realtime & offline
"""

from avistrack.config.loader import load_config
from avistrack.backends.base import TrackerBackend


def load_tracker(config_path: str) -> "TrackerBackend":
    """
    Factory function.  Reads the YAML config and returns the appropriate
    backend instance, ready to call .update(frame).

    Parameters
    ----------
    config_path : str | Path
        Path to a YAML config file (see configs/template.yaml for reference).
    """
    cfg = load_config(config_path)
    backend_name = cfg.model.backend.lower()

    if backend_name == "yolo":
        mode = cfg.model.get("mode", "offline").lower()
        if mode == "realtime":
            from avistrack.backends.yolo.realtime import YoloRealtimeTracker
            return YoloRealtimeTracker(cfg)
        else:
            from avistrack.backends.yolo.offline import YoloOfflineTracker
            return YoloOfflineTracker(cfg)

    elif backend_name == "dlc":
        from avistrack.backends.dlc import DLCTracker
        return DLCTracker(cfg)

    elif backend_name == "vit":
        from avistrack.backends.vit import ViTTracker
        return ViTTracker(cfg)

    else:
        raise ValueError(
            f"Unknown backend '{backend_name}'. "
            "Supported values: 'yolo', 'dlc', 'vit'."
        )


__all__ = ["load_tracker", "load_config"]
