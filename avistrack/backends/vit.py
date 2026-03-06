"""
ViT (DorsalVentralNet) backend.

Ported from ChamberBroadcaster/chamber_broadcaster/processors/joshu_tracking.py
Adapted to return standard Detection objects.

The DorsalVentralNet model class itself is NOT vendored here – it lives in
ChamberBroadcaster.  Install that package (pip install -e /path/to/ChamberBroadcaster)
or point PYTHONPATH at it before using this backend.
"""

import collections
import queue
import threading
import logging
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from avistrack.backends.base import TrackerBackend, Detection

logger = logging.getLogger(__name__)


class ViTTracker(TrackerBackend):
    """
    Non-blocking keypoint tracker using the DorsalVentralNet (Vision Transformer).

    Behaviour mirrors JoshuTrackingProcessor from ChamberBroadcaster:
    each update() call is non-blocking and returns the most recent prediction.

    Config keys expected under cfg.model:
        weights               : {"bg": "/path/bg.pt", "kp": "/path/kp.pt"}
        model_type            : swin backbone variant (default "swin_s3_small_224")
        target_model_size     : input resolution (default 256)
        sequence_length       : temporal context frames (default 4)
        num_instances         : number of animals (default 1)
        keypoint_names        : list of keypoint label strings
        frame_channels        : 3 (RGB) or 4 (RGB+IR, default 3)
        mask_temperature      : float (default 1.0)
        head_dropout_rate     : float (default 0.1)
        backbone_drop_rate    : float (default 0.1)
        backbone_drop_path_rate: float (default 0.1)
    """

    def __init__(self, cfg):
        # ── Lazy import ──────────────────────────────────────────────
        try:
            from chamber_broadcaster.models.dorsal_ventral_net import DorsalVentralNet
        except ImportError:
            raise ImportError(
                "DorsalVentralNet not found. "
                "Install ChamberBroadcaster: pip install -e /path/to/ChamberBroadcaster"
            )

        model_cfg = cfg.model
        self._num_instances    = int(model_cfg.get("num_instances", 1))
        self._model_type       = model_cfg.get("model_type", "swin_s3_small_224")
        self._target_size      = int(model_cfg.get("target_model_size", 256))
        self._output_map_size  = int(model_cfg.get("output_map_size", self._target_size))
        self._seq_len          = int(model_cfg.get("sequence_length", 4))
        self._frame_channels   = int(model_cfg.get("frame_channels", 3))
        self._mask_temp        = float(model_cfg.get("mask_temperature", 1.0))
        self._head_drop        = float(model_cfg.get("head_dropout_rate", 0.1))
        self._bb_drop          = float(model_cfg.get("backbone_drop_rate", 0.1))
        self._bb_drop_path     = float(model_cfg.get("backbone_drop_path_rate", 0.1))

        # Keypoint names
        kp_names = model_cfg.get("keypoint_names")
        n_kp     = model_cfg.get("num_keypoints")
        if kp_names:
            self._kp_names = kp_names
            self._num_kp   = len(kp_names)
        elif n_kp:
            self._num_kp   = int(n_kp)
            self._kp_names = [f"kp_{i}" for i in range(self._num_kp)]
        else:
            raise ValueError("cfg.model must provide 'keypoint_names' or 'num_keypoints'")

        # ── Device ───────────────────────────────────────────────────
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"[ViT] Using device: {self._device}")

        # ── Build model ──────────────────────────────────────────────
        total_channels = self._frame_channels * self._seq_len
        self._model = DorsalVentralNet(
            num_keypoints       = self._num_kp,
            mask_output_size    = self._target_size,
            heatmap_output_size = self._output_map_size,
            in_chans            = total_channels,
            model_cfg = {
                "model_type":              self._model_type,
                "target_model_size":       self._target_size,
                "backbone_drop_rate":      self._bb_drop,
                "backbone_drop_path_rate": self._bb_drop_path,
                "head_dropout_rate":       self._head_drop,
                "mask_temperature":        self._mask_temp,
            },
        ).to(self._device)

        weights = model_cfg.weights
        self._load_weights(weights.get("bg"), weights.get("kp"))
        self._model.eval()

        # ── State ────────────────────────────────────────────────────
        self._frame_buffer: collections.deque = collections.deque(maxlen=self._seq_len)
        self._last_detections: list[Detection] = []
        self._frame_idx = 0

        # ── Background thread ────────────────────────────────────────
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("[ViT] Background inference thread started.")

    # ── Public API ───────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> list[Detection]:
        """Non-blocking update. Returns last known keypoints immediately."""
        self._frame_idx += 1
        resized = cv2.resize(frame, (self._target_size, self._target_size))
        src_h, src_w = frame.shape[:2]
        item = (resized, self._frame_idx, (src_h, src_w))

        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            pass

        return self._last_detections

    def release(self) -> None:
        logger.info("[ViT] Releasing resources...")
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()

    # ── Internal ─────────────────────────────────────────────────────────

    def _load_weights(self, bg_path: str, kp_path: str):
        for path, subnet, name in [
            (bg_path, self._model.background_net, "background"),
            (kp_path, self._model.keypoint_net,   "keypoint"),
        ]:
            if not path or not __import__("pathlib").Path(path).exists():
                raise FileNotFoundError(f"[ViT] {name} checkpoint not found: {path}")
            state = torch.load(path, map_location=self._device, weights_only=True)
            subnet.load_state_dict(state, strict=False)
            logger.info(f"[ViT] Loaded {name} weights from {path}")

    def _loop(self):
        while True:
            try:
                item = self._queue.get()
                if item is None:
                    break

                resized, frame_idx, (src_h, src_w) = item
                self._frame_buffer.append(resized)

                seq = list(self._frame_buffer)
                if not seq:
                    continue
                # Pad to sequence length if necessary
                if len(seq) < self._seq_len:
                    seq = [seq[0]] * (self._seq_len - len(seq)) + seq

                tensor = self._preprocess(seq).to(self._device)

                with torch.no_grad():
                    _, heatmaps = self._model(tensor)

                N, C, H, W = heatmaps.shape
                probs = F.softmax(heatmaps.view(N, C, -1), dim=2).view(N, C, H, W)
                preds = self._extract_keypoints(probs)

                animals: list[list[dict]] = [[] for _ in range(self._num_instances)]
                for kp_idx, instances in enumerate(preds):
                    for inst_idx, (x, y, lk) in enumerate(instances):
                        if inst_idx < self._num_instances:
                            animals[inst_idx].append({
                                "label":      self._kp_names[kp_idx],
                                "x":          x * src_w / self._output_map_size,
                                "y":          y * src_h / self._output_map_size,
                                "likelihood": lk,
                            })

                self._last_detections = [
                    Detection(
                        track_id  = i + 1,
                        x=0.0, y=0.0, w=0.0, h=0.0,
                        keypoints = kps,
                    )
                    for i, kps in enumerate(animals)
                ]

            except Exception as e:
                logger.error(f"[ViT] Loop error: {e}", exc_info=True)

    def _preprocess(self, frames: list[np.ndarray]) -> torch.Tensor:
        seq = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).float() / 255.0
        seq = F.interpolate(seq, size=(self._target_size, self._target_size),
                            mode="bilinear", align_corners=False)
        if self._frame_channels == 4:
            gray = (seq[:, 0:1] * 0.114 + seq[:, 1:2] * 0.587 + seq[:, 2:3] * 0.299)
            seq  = torch.cat([seq, gray], dim=1)
        return seq.unsqueeze(0)   # (1, T, C, H, W)

    def _extract_keypoints(self, heatmaps: torch.Tensor):
        heatmaps = heatmaps.squeeze(0)
        C, H, W  = heatmaps.shape
        dilated  = F.max_pool2d(heatmaps.unsqueeze(0), kernel_size=3,
                                stride=1, padding=1).squeeze(0)
        max_vals = torch.amax(heatmaps.view(C, -1), dim=1).view(C, 1, 1)
        peaks    = (heatmaps == dilated) & (heatmaps > max_vals * 0.2)

        all_kps = []
        for i in range(C):
            if not peaks[i].any():
                all_kps.append([])
                continue
            coords = peaks[i].nonzero(as_tuple=False)          # (yx)
            vals   = heatmaps[i, coords[:, 0], coords[:, 1]]
            top_n  = min(self._num_instances, len(vals))
            idx    = torch.argsort(vals, descending=True)[:top_n]
            all_kps.append([
                (float(coords[j, 1]), float(coords[j, 0]), float(vals[j]))
                for j in idx.cpu().numpy()
            ])
        return all_kps
