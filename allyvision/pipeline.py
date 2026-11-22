"""Per-frame fusion, route choice, and color-coded navigation visualization."""

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image

from .navigation import TemporalStatusSmoother, analyze_directions, choose_recommendation
from .spatial_matrix import build_spatial_matrix, draw_spatial_matrix
from .unified_visualizer import UnifiedVisualizationConfig, reliable_object_distance, render_unified


@dataclass
class FrameResult:
    directions: list[dict]
    obstacles: list[dict]
    detected_objects: list[dict]
    spatial_matrix: list[dict]
    object_message: str
    device: str
    recommended_direction: str | None
    route_message: str


PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32],
], dtype=np.uint8)
STATUS_COLORS = {"FREE": (45, 190, 45), "BLOCKED": (35, 35, 225), "UNCERTAIN": (0, 175, 245)}
WINDOW_NAME = "AllyVision: open-path guidance"


def _draw_navigation_overlay(image: np.ndarray, decisions, recommendation: str | None) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    y_top = int(h * 0.52)
    tint = out.copy()
    for i, decision in enumerate(decisions):
        x1, x2 = i * w // 3, (i + 1) * w // 3
        color = STATUS_COLORS[decision.status]
        cv2.rectangle(tint, (x1, y_top), (x2 - 1, h - 1), color, thickness=-1)
    cv2.addWeighted(tint, 0.24, out, 0.76, 0, out)

    # Keep the colors and labels legible against both the scene and segmentation.
    for i, decision in enumerate(decisions):
        x1, x2 = i * w // 3, (i + 1) * w // 3
        color = STATUS_COLORS[decision.status]
        cv2.rectangle(out, (x1, y_top), (x2 - 1, h - 1), color, 3)
        label = f"{decision.direction}: {decision.status}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        label_x = max(x1 + 5, min(x2 - tw - 8, x1 + 8))
        label_y = max(th + 8, y_top - 10)
        cv2.rectangle(out, (label_x - 4, label_y - th - 6), (label_x + tw + 4, label_y + 5), (20, 20, 20), -1)
        cv2.putText(out, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

    if recommendation:
        headline = f"OPEN SIDEWALK: {recommendation}"
        target = next(i for i, d in enumerate(decisions) if d.direction == recommendation)
        x1, x2 = target * w // 3, (target + 1) * w // 3
        cx = (x1 + x2) // 2
        cv2.arrowedLine(out, (cx, int(h * 0.76)), (cx, int(h * 0.93)), STATUS_COLORS["FREE"], 8, tipLength=0.3)
        banner_color = STATUS_COLORS["FREE"]
    else:
        headline = "NO CLEAR SIDEWALK - WAIT / REASSESS"
        banner_color = STATUS_COLORS["UNCERTAIN"]
    cv2.rectangle(out, (0, 0), (w, min(48, h - 1)), (22, 22, 22), -1)
    cv2.putText(out, headline, (12, min(34, h - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, banner_color, 2, cv2.LINE_AA)
    return out


def _panel(image: np.ndarray, title: str, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ih, iw = image.shape[:2]
    scale = min(w / max(1, iw), h / max(1, ih))
    rw, rh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_AREA)
    fitted = np.zeros((h, w, 3), dtype=np.uint8)
    x, y = (w - rw) // 2, (h - rh) // 2
    fitted[y:y + rh, x:x + rw] = resized
    cv2.rectangle(fitted, (0, 0), (w, 34), (18, 18, 18), -1)
    cv2.putText(fitted, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
    return fitted


def _dashboard(original, segmentation, detections, depth_m, sectors, decisions,
               object_notice="NO OBJECT DETECTED", size=(640, 360)):
    w, h = size
    if depth_m is None:
        depth_vis = np.zeros_like(original)
        cv2.putText(depth_vis, "Metric depth unavailable", (30, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        depth_valid = np.isfinite(depth_m) & (depth_m > 0)
        depth_scaled = np.clip(depth_m / 80.0 * 255, 0, 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_TURBO)
        depth_vis[~depth_valid] = 0
        cv2.putText(depth_vis, "0 m near  |  80 m far", (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

    status_panel = np.full((h, w, 3), (34, 34, 34), dtype=np.uint8)
    cv2.putText(status_panel, "CONSERVATIVE ROUTE STATUS", (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (250, 250, 250), 2)
    cv2.putText(status_panel, object_notice[:52], (18, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
    y = 130
    for item in decisions:
        color = STATUS_COLORS[item.status]
        text = f"{item.direction}: {item.status}  {item.confidence:.0%}"
        cv2.putText(status_panel, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        y += 62
        distance = item.nearest_obstacle_m
        reliable = (distance is not None and item.nearest_obstacle_spread is not None
                    and 0.3 <= distance <= 12.0 and item.nearest_obstacle_spread <= 0.35)
        distance_text = f"~{distance:.1f}m" if reliable else "--"
        detail = f"sidewalk {item.walkable_fraction:.0%} | obstacle {distance_text}"
        cv2.putText(status_panel, detail, (40, y - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
    advice = "No clear route: wait / reassess" if not choose_recommendation(decisions) else f"Suggested: {choose_recommendation(decisions)}"
    cv2.putText(status_panel, advice, (18, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)

    panels = [
        _panel(original, "1  ORIGINAL FRAME", size),
        _panel(segmentation, "2  SEGMENTATION OVERLAY", size),
        _panel(detections, "3  OBJECT DETECTIONS", size),
        _panel(depth_vis, "4  METRIC DEPTH", size),
        _panel(sectors, "5  LEFT / CENTER / RIGHT", size),
        _panel(status_panel, "6  NAVIGATION DECISION", size),
    ]
    return np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))


def process_frame(bgr: np.ndarray, models, *, detector_confidence: float = 0.25, imgsz: int = 320,
                  depth_map_m: np.ndarray | None = None, run_depth: bool = True,
                  smoother: TemporalStatusSmoother | None = None,
                  visualization: str = "unified",
                  visualization_config: UnifiedVisualizationConfig | None = None):
    if bgr is None or bgr.size == 0:
        raise ValueError("Empty frame")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    class_map, conf_map = models.segment(Image.fromarray(rgb))
    detected_objects = models.detect(bgr, confidence=detector_confidence, imgsz=imgsz)
    obstacles = [obj for obj in detected_objects if obj.get("navigation_obstacle", True)]
    if depth_map_m is None and run_depth and hasattr(models, "estimate_depth"):
        depth_map_m = models.estimate_depth(Image.fromarray(rgb))
    if depth_map_m is not None and depth_map_m.shape != class_map.shape:
        depth_map_m = cv2.resize(depth_map_m, (class_map.shape[1], class_map.shape[0]), interpolation=cv2.INTER_LINEAR)
    decisions = analyze_directions(
        class_map, conf_map, models.walkable_ids, models.road_ids, detected_objects, depth_map_m=depth_map_m,
    )
    if smoother is not None:
        smoother.apply(decisions)
    spatial_matrix = build_spatial_matrix(
        bgr.shape, class_map, conf_map, models.walkable_ids, models.road_ids, detected_objects,
    )
    recommendation = choose_recommendation(decisions)
    route_message = f"OPEN SIDEWALK: {recommendation}" if recommendation else "NO CLEAR SIDEWALK - WAIT / REASSESS"

    labels = np.clip(class_map.astype(np.int32), 0, len(PALETTE) - 1)
    color_seg = PALETTE[labels][:, :, ::-1].copy()
    seg_vis = cv2.addWeighted(bgr, 0.42, color_seg, 0.58, 0)
    annotated = bgr.copy()
    detection_view = bgr.copy()
    for obj in detected_objects:
        x1, y1, x2, y2 = obj["box"]
        visible_distance = reliable_object_distance(obj, depth_map_m is not None)
        dist = f" ~{visible_distance:.1f}m" if visible_distance is not None else ""
        caption = f"{obj['label']} {obj['confidence']:.2f}{dist}"
        for canvas in (annotated, seg_vis, detection_view):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.putText(canvas, caption, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2, cv2.LINE_AA)
    detection_view = draw_spatial_matrix(detection_view, spatial_matrix)
    sectors = _draw_navigation_overlay(bgr, decisions, recommendation)
    if detected_objects:
        labels_found = list(dict.fromkeys(obj["label"] for obj in detected_objects))
        summary = ", ".join(labels_found[:5])
        if len(labels_found) > 5:
            summary += f", +{len(labels_found) - 5} types"
        object_message = f"OBJECTS DETECTED ({len(detected_objects)}): {summary}"
    else:
        object_message = "NO OBJECT DETECTED"
    if visualization == "unified":
        combined = render_unified(
            bgr, class_map, conf_map, models.walkable_ids, models.road_ids,
            detected_objects, decisions, recommendation, object_message,
            id2label=getattr(models, "id2label", None),
            depth_available=depth_map_m is not None,
            spatial_matrix=spatial_matrix,
            config=visualization_config,
        )
    elif visualization == "debug":
        combined = _dashboard(bgr, seg_vis, detection_view, depth_map_m, sectors, decisions, object_notice=object_message)
    else:
        raise ValueError("visualization must be 'unified' or 'debug'")
    payload = FrameResult(
        directions=[d.to_dict() for d in decisions],
        obstacles=obstacles,
        detected_objects=detected_objects,
        spatial_matrix=spatial_matrix,
        object_message=object_message,
        device=models.device,
        recommended_direction=recommendation,
        route_message=route_message,
    )
    return asdict(payload), combined, depth_map_m
