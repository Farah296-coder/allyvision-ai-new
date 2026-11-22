"""Metric outdoor monocular depth estimation with explicit model validation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from pathlib import Path


DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
LOCAL_DEPTH_MODEL = Path(__file__).resolve().parent.parent / "models" / "depth_anything_v2_metric_outdoor_small"


def _depth_model_source():
    checkpoint = LOCAL_DEPTH_MODEL / "model.safetensors"
    if (LOCAL_DEPTH_MODEL / "config.json").is_file() and checkpoint.is_file():
        lfs_marker = b"version https://git-lfs.github.com/spec/v1"
        with checkpoint.open("rb") as weights:
            prefix = weights.read(len(lfs_marker))
        if len(prefix) >= 8 and prefix != lfs_marker:
            return str(LOCAL_DEPTH_MODEL)
    return DEPTH_MODEL_ID


class MetricOutdoorDepth:
    """Depth Anything V2 Outdoor Small; output units are meters."""

    def __init__(self, device: str, input_size: int = 336) -> None:
        self.device = device
        if input_size < 224 or input_size % 14 != 0:
            raise ValueError("depth input size must be at least 224 and divisible by the model's 14-pixel patch size")
        model_source = _depth_model_source()
        self.processor = AutoImageProcessor.from_pretrained(model_source)
        self.processor.size = {"height": input_size, "width": input_size}
        self.input_size = input_size
        self.model = AutoModelForDepthEstimation.from_pretrained(model_source).to(device).eval()
        config = self.model.config
        if getattr(config, "depth_estimation_type", None) != "metric":
            raise RuntimeError(f"Refusing non-metric depth checkpoint: {getattr(config, 'depth_estimation_type', None)!r}")
        self.max_depth_m = float(getattr(config, "max_depth", 0.0))
        if not np.isfinite(self.max_depth_m) or self.max_depth_m <= 0:
            raise RuntimeError("Metric depth checkpoint does not declare a valid max_depth in meters")

    @torch.inference_mode()
    def estimate(self, rgb_image) -> np.ndarray:
        inputs = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        prediction = self.model(**inputs).predicted_depth
        prediction = F.interpolate(
            prediction.unsqueeze(1), size=(rgb_image.height, rgb_image.width),
            mode="bicubic", align_corners=False,
        )[0, 0]
        # The selected checkpoint is configured for metric output and an 80 m outdoor range.
        return prediction.clamp_(0.0, self.max_depth_m).cpu().numpy().astype(np.float32)
