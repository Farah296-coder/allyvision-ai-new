"""Single-frame overlay for segmentation, objects, free space, depth, and routes."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .spatial_matrix import draw_spatial_matrix


@dataclass(frozen=True)
class UnifiedVisualizationConfig:
    walkable_color: tuple[int, int, int] = (45, 210, 45)  # BGR green
    road_color: tuple[int, int, int] = (35, 35, 225)      # BGR red
    obstacle_color: tuple[int, int, int] = (30, 30, 255)
    object_color: tuple[int, int, int] = (255, 190, 0)
    uncertain_color: tuple[int, int, int] = (0, 175, 245)
    semantic_opacity: float = 0.28
    free_space_opacity: float = 0.34
    corridor_opacity: float = 0.10
    show_labels: bool = True
    max_display_depth_m: float = 12.0
    max_depth_spread: float = 0.35


STATUS_COLORS = {
    "FREE": (45, 210, 45),
    "BLOCKED": (35, 35, 235),
    "UNCERTAIN": (0, 175, 245),
}


def reliable_object_distance(obj: dict, depth_available: bool,
                             config: UnifiedVisualizationConfig | None = None) -> float | None:
    """Return a conservative display value; unstable or far estimates stay hidden."""
    config = config or UnifiedVisualizationConfig()
    dist, spread = obj.get("distance_m"), obj.get("distance_spread")
    if not depth_available or dist is None or spread is None:
        return None
    if not 0.3 <= float(dist) <= config.max_display_depth_m or float(spread) > config.max_depth_spread:
        return None
    return float(dist)


def _blend_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if mask.any():
        pixels = image[mask].astype(np.float32)
        image[mask] = np.clip(pixels * (1.0 - alpha) + np.asarray(color, dtype=np.float32) * alpha, 0, 255).astype(np.uint8)


def _text_box(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int], scale=0.48):
    x, baseline = origin
    thickness = 1
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = min(max(2, x), max(2, image.shape[1] - tw - 8))
    baseline = min(max(th + 5, baseline), image.shape[0] - base - 2)
    cv2.rectangle(image, (x - 3, baseline - th - 4), (x + tw + 4, baseline + base + 2), (20, 20, 20), -1)
    cv2.putText(image, text, (x, baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return (x - 3, baseline - th - 4, x + tw + 4, baseline + base + 2)


def _overlaps(rect: tuple[int, int, int, int], others: list[tuple[int, int, int, int]]) -> bool:
    x1, y1, x2, y2 = rect
    return any(x1 < ox2 and x2 > ox1 and y1 < oy2 and y2 > oy1 for ox1, oy1, ox2, oy2 in others)


def render_unified(
    frame: np.ndarray,
    class_map: np.ndarray,
    confidence_map: np.ndarray,
    walkable_ids: set[int],
    road_ids: set[int],
    detected_objects: list[dict],
    decisions: list,
    recommendation: str | None,
    object_message: str,
    id2label: dict[int, str] | None = None,
    *,
    depth_available: bool,
    spatial_matrix: list[dict] | None = None,
    config: UnifiedVisualizationConfig | None = None,
) -> np.ndarray:
    """Render all available outputs aligned to the source frame's coordinates."""
    config = config or UnifiedVisualizationConfig()
    if frame is None or frame.ndim != 3:
        raise ValueError("frame must be a BGR image")
    h, w = frame.shape[:2]
    if class_map.shape != (h, w) or confidence_map.shape != (h, w):
        raise ValueError("segmentation and confidence maps must match source frame size")

    out = frame.copy()
    walkable = np.isin(class_map, list(walkable_ids))
    road = np.isin(class_map, list(road_ids))
    confident_walkable = walkable & (confidence_map >= 0.48)
    uncertain_walkable = walkable & ~confident_walkable

    # Tint navigation sectors, then show semantic surfaces and usable free sidewalk.
    roi_top = int(h * 0.52)
    for idx, decision in enumerate(decisions):
        x1, x2 = idx * w // 3, (idx + 1) * w // 3
        mask = np.zeros((h, w), dtype=bool)
        mask[roi_top:, x1:x2] = True
        _blend_mask(out, mask, STATUS_COLORS.get(decision.status, config.uncertain_color), config.corridor_opacity)

    _blend_mask(out, road, config.road_color, config.semantic_opacity)
    _blend_mask(out, uncertain_walkable, config.uncertain_color, config.semantic_opacity)

    # SegFormer also supplies pixel masks for common object and environmental
    # classes. This is semantic (class-level), not per-instance segmentation.
    if id2label:
        structure_labels = {"building", "wall", "fence", "pole", "vegetation", "terrain", "guard rail", "bridge", "tunnel"}
        person_labels = {"person", "rider"}
        vehicle_labels = {"car", "truck", "bus", "train", "motorcycle", "bicycle"}
        for class_id, raw_label in id2label.items():
            label = raw_label.strip().lower()
            if class_id in walkable_ids or class_id in road_ids:
                continue
            mask = (class_map == int(class_id)) & (confidence_map >= 0.50)
            if label in structure_labels:
                _blend_mask(out, mask, config.road_color, config.semantic_opacity * 0.48)
            elif label in person_labels:
                _blend_mask(out, mask, config.object_color, config.semantic_opacity)
            elif label in vehicle_labels:
                _blend_mask(out, mask, (230, 115, 30), config.semantic_opacity)
            elif label in {"traffic light", "traffic sign"}:
                _blend_mask(out, mask, config.uncertain_color, config.semantic_opacity)

    occupied = np.zeros((h, w), dtype=bool)
    for obj in detected_objects:
        if not obj.get("navigation_obstacle", True):
            continue
        x1, y1, x2, y2 = (int(v) for v in obj["box"])
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        if x2 > x1 and y2 > y1:
            occupied[y1:y2, x1:x2] = True
    free_space = confident_walkable & ~occupied
    blocked_walkable = confident_walkable & occupied
    _blend_mask(out, free_space, config.walkable_color, config.free_space_opacity)
    _blend_mask(out, blocked_walkable, config.obstacle_color, config.free_space_opacity)

    # Draw every supported detector output. Use semantic rectangles as context,
    # but keep incidental objects distinct from navigation hazards.
    label_rects: list[tuple[int, int, int, int]] = []
    ordered = sorted(detected_objects, key=lambda o: float(o.get("confidence", 0.0)), reverse=True)
    for obj in ordered:
        x1, y1, x2, y2 = (int(v) for v in obj["box"])
        x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
        y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        color = config.obstacle_color if obj.get("navigation_obstacle", True) else config.object_color
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        if not config.show_labels:
            continue
        caption = f"{obj['label']} {float(obj.get('confidence', 0.0)):.2f}"
        dist = reliable_object_distance(obj, depth_available, config)
        if dist is not None:
            caption += f" ~{dist:.1f}m"
        (tw, th), base = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        candidates = [
            (x1, y1 - 5),
            (x1, y2 + th + 7),
            (x2 - tw, y1 - 5),
            (x2 - tw, y2 + th + 7),
        ]
        for lx, baseline in candidates:
            rect = (lx - 3, baseline - th - 4, lx + tw + 4, baseline + base + 2)
            if (0 <= rect[0] and rect[2] < w and rect[1] >= 0 and rect[3] < h
                    and not _overlaps(rect, label_rects)):
                label_rects.append(_text_box(out, caption, (lx, baseline), color, scale=0.46))
                break

    # Corridor boundaries and status tags stay in source-image coordinates.
    cv2.line(out, (0, roi_top), (w - 1, roi_top), (245, 245, 245), 1, cv2.LINE_AA)
    placed_rects = label_rects.copy()
    for idx, decision in enumerate(decisions):
        x1, x2 = idx * w // 3, (idx + 1) * w // 3
        color = STATUS_COLORS.get(decision.status, config.uncertain_color)
        if idx:
            cv2.line(out, (x1, roi_top), (x1, h - 1), color, 2, cv2.LINE_AA)
        title = f"{decision.direction}: {decision.status} {decision.confidence:.0%}"
        lane_width = x2 - x1
        label_scale = min(0.48, max(0.26, (lane_width - 12) / 350))
        (text_width, text_height), base = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, label_scale, 1)
        label_x = x1 + max(4, (lane_width - text_width) // 2)
        for baseline in (h - 10, h - 32, roi_top + 48, roi_top + 22):
            baseline = min(h - 6, baseline)
            rect = (label_x - 3, baseline - text_height - 4, label_x + text_width + 4, baseline + base + 2)
            if rect[1] >= 0 and rect[3] < h and not _overlaps(rect, placed_rects):
                placed_rects.append(_text_box(out, title, (label_x, baseline), color, scale=label_scale))
                break

    if recommendation:
        idx = next((i for i, d in enumerate(decisions) if d.direction == recommendation), 1)
        cx = (idx * w // 3 + (idx + 1) * w // 3) // 2
        cv2.arrowedLine(out, (cx, int(h * 0.78)), (cx, int(h * 0.95)), config.walkable_color, 5, tipLength=0.3)

    # Compact decision and object-status HUD, overlaid on the same camera frame.
    panel_w = min(430, max(40, w - 16))
    panel_h = min(162, max(40, h - 16))
    x0, y0 = 8, 8
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.76, out, 0.24, 0, out)
    cv2.putText(out, "ALLYVISION NAVIGATION", (x0 + 10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    y = y0 + 46
    for decision in decisions:
        nearest = decision.nearest_obstacle_m
        dist_reliable = (nearest is not None and decision.nearest_obstacle_spread is not None
                         and 0.3 <= nearest <= config.max_display_depth_m
                         and decision.nearest_obstacle_spread <= config.max_depth_spread)
        dist_text = f" | nearest ~{nearest:.1f}m" if dist_reliable else ""
        line = f"{decision.direction}: {decision.status} {decision.confidence:.0%}{dist_text}"
        cv2.putText(out, line, (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, STATUS_COLORS.get(decision.status, config.uncertain_color), 1, cv2.LINE_AA)
        y += 21
    object_line = object_message if len(object_message) < 59 else object_message[:56] + "..."
    cv2.putText(out, object_line, (x0 + 10, min(y0 + panel_h - 25, y + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, config.object_color, 1, cv2.LINE_AA)
    legend = "GREEN walkable/free | RED road/blocked | CYAN other objects"
    cv2.putText(out, legend, (x0 + 10, y0 + panel_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (230, 230, 230), 1, cv2.LINE_AA)
    if spatial_matrix is not None:
        out = draw_spatial_matrix(out, spatial_matrix)
    return out
