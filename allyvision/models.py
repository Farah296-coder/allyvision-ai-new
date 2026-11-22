"""Lazy loading for the pretrained segmentation and detection models."""

from pathlib import Path

import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from ultralytics import YOLO
from .depth import MetricOutdoorDepth


SEGMENTATION_MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
DETECTION_WEIGHTS = "yolo11n.pt"
NAVIGATION_OBSTACLE_LABELS = {
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
    "fire hydrant", "stop sign", "parking meter", "bench", "chair",
    "couch", "dining table", "potted plant", "suitcase", "traffic light",
}


class PerceptionModels:
    def __init__(self, device: str = "auto", *, segmentation_size: int = 256, cpu_threads: int = 1,
                 enable_depth: bool = True, walkable_labels: tuple[str, ...] = ("sidewalk",),
                 road_labels: tuple[str, ...] = ("road",), depth_size: int = 336) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch environment")
        self.device = device
        if segmentation_size < 128:
            raise ValueError("segmentation_size must be at least 128 pixels")
        self.segmentation_size = segmentation_size
        if device == "cpu":
            torch.set_num_threads(max(1, min(cpu_threads, torch.get_num_threads())))
        self.processor = AutoImageProcessor.from_pretrained(SEGMENTATION_MODEL_ID)
        # Smaller inputs make the CPU profile much faster; inference output is
        # still resized to the source frame before corridor analysis.
        self.processor.size = {"height": segmentation_size, "width": segmentation_size}
        self.segmenter = SegformerForSemanticSegmentation.from_pretrained(SEGMENTATION_MODEL_ID).to(device).eval()
        local_weights = Path(__file__).resolve().parent.parent / "models" / DETECTION_WEIGHTS
        self.detector = YOLO(str(local_weights if local_weights.exists() else DETECTION_WEIGHTS))
        self.id2label = {int(k): str(v) for k, v in self.segmenter.config.id2label.items()}
        self.walkable_labels = {name.strip().lower() for name in walkable_labels if name.strip()}
        self.road_labels = {name.strip().lower() for name in road_labels if name.strip()}
        self.walkable_ids = {
            idx for idx, label in self.id2label.items()
            if label.strip().lower() in self.walkable_labels
        }
        self.road_ids = {
            idx for idx, label in self.id2label.items()
            if label.strip().lower() in self.road_labels
        }
        if not self.walkable_ids:
            raise RuntimeError(f"No configured walkable labels {sorted(self.walkable_labels)} matched: {self.id2label}")
        self.depth_model = MetricOutdoorDepth(device, input_size=depth_size) if enable_depth else None

    def segment(self, rgb_image):
        import torch.nn.functional as F

        inputs = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = self.segmenter(**inputs).logits
            logits = F.interpolate(logits, size=(rgb_image.height, rgb_image.width), mode="bilinear", align_corners=False)
            probs = logits.softmax(dim=1)[0]
            confidence, class_map = probs.max(dim=0)
        return class_map.cpu().numpy().astype("uint8"), confidence.cpu().numpy().astype("float32")

    def detect(self, bgr_image, *, confidence: float = 0.25, imgsz: int = 640):
        result = self.detector.predict(
            source=bgr_image, conf=confidence, imgsz=imgsz,
            device=self.device, verbose=False,
        )[0]
        # Return every class YOLO11n recognizes. Navigation separately marks
        # the subset that should count as a possible path obstruction.
        objects = []
        if result.boxes is None:
            return objects
        for box in result.boxes:
            cls_id = int(box.cls.item())
            label = str(result.names[cls_id])
            xyxy = [int(round(v)) for v in box.xyxy[0].tolist()]
            objects.append({
                "label": label, "confidence": float(box.conf.item()), "box": xyxy, "distance_m": None,
                "navigation_obstacle": label in NAVIGATION_OBSTACLE_LABELS,
            })
        return objects

    def estimate_depth(self, rgb_image):
        if self.depth_model is None:
            return None
        return self.depth_model.estimate(rgb_image)

