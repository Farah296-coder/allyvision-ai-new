"""Per-frame 3x3 spatial localization and visualization helpers."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


REGION_NAMES = (
    ("TOP_LEFT", "Top Left"), ("TOP_CENTER", "Top Center"), ("TOP_RIGHT", "Top Right"),
    ("MIDDLE_LEFT", "Middle Left"), ("MIDDLE_CENTER", "Middle Center"), ("MIDDLE_RIGHT", "Middle Right"),
    ("BOTTOM_LEFT", "Bottom Left"), ("BOTTOM_CENTER", "Bottom Center"), ("BOTTOM_RIGHT", "Bottom Right"),
)
STATUS_COLORS = {"FREE": (45, 190, 45), "BLOCKED": (35, 35, 225), "UNCERTAIN": (0, 175, 245)}


def build_spatial_matrix(
    frame_shape: tuple[int, ...],
    class_map: np.ndarray,
    confidence_map: np.ndarray,
    walkable_ids: set[int],
    road_ids: set[int],
    detected_objects: list[dict[str, Any]],
    *,
    reliable_distance_max_m: float = 12.0,
    reliable_distance_spread: float = 0.35,
) -> list[dict[str, Any]]:
    """Build nine frame-aligned regions; object extent is clipped into every overlapped cell."""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    if h < 3 or w < 3:
        raise ValueError("Frame must be at least 3x3 for spatial matrix")
    if class_map.shape != (h, w) or confidence_map.shape != (h, w):
        raise ValueError("class and confidence maps must match frame dimensions")

    walkable = np.isin(class_map, list(walkable_ids))
    road = np.isin(class_map, list(road_ids))
    matrix: list[dict[str, Any]] = []
    for index, (region_id, label) in enumerate(REGION_NAMES):
        row, col = divmod(index, 3)
        x1, x2 = col * w // 3, (col + 1) * w // 3
        y1, y2 = row * h // 3, (row + 1) * h // 3
        region = (x1, y1, x2, y2)
        region_walk = walkable[y1:y2, x1:x2]
        region_road = road[y1:y2, x1:x2]
        region_conf = confidence_map[y1:y2, x1:x2]
        walk_fraction = float(region_walk.mean()) if region_walk.size else 0.0
        road_fraction = float(region_road.mean()) if region_road.size else 0.0
        mean_confidence = float(region_conf.mean()) if region_conf.size else 0.0
        walk_confidence = float(region_conf[region_walk].mean()) if region_walk.any() else 0.0

        objects: list[dict[str, Any]] = []
        obstacle_present = False
        for obj in detected_objects:
            bx1, by1, bx2, by2 = (int(v) for v in obj["box"])
            clipped = (max(0, min(w, bx1)), max(0, min(h, by1)), max(0, min(w, bx2)), max(0, min(h, by2)))
            ix1, iy1 = max(x1, clipped[0]), max(y1, clipped[1])
            ix2, iy2 = min(x2, clipped[2]), min(y2, clipped[3])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            box_area = max(1, (clipped[2] - clipped[0]) * (clipped[3] - clipped[1]))
            overlap_area = (ix2 - ix1) * (iy2 - iy1)
            distance = obj.get("distance_m")
            spread = obj.get("distance_spread")
            reliable_distance = (
                distance is not None and spread is not None
                and 0.3 <= float(distance) <= reliable_distance_max_m
                and float(spread) <= reliable_distance_spread
            )
            is_obstacle = bool(obj.get("navigation_obstacle", True))
            obstacle_present = obstacle_present or is_obstacle
            objects.append({
                "class": str(obj.get("label", "unknown")),
                "confidence": round(float(obj.get("confidence", 0.0)), 4),
                "bbox_xyxy": [int(v) for v in clipped],
                "center_xy": [round((clipped[0] + clipped[2]) / 2, 1), round((clipped[1] + clipped[3]) / 2, 1)],
                "overlap_bbox_xyxy": [ix1, iy1, ix2, iy2],
                "cell_overlap_fraction": round(overlap_area / box_area, 4),
                "obstacle": is_obstacle,
                "estimated_distance_m": round(float(distance), 2) if reliable_distance else None,
            })

        if obstacle_present:
            status, reason = "BLOCKED", "A detected navigation obstacle overlaps this region"
        elif mean_confidence < 0.48:
            status, reason = "UNCERTAIN", "Semantic confidence is low"
        elif walk_fraction >= 0.22 and walk_confidence >= 0.52:
            status, reason = "FREE", "Confident walkable surface is visible in this region"
        elif road_fraction >= 0.20:
            status, reason = "UNCERTAIN", "Road surface alone does not establish a safe walking region"
        elif walk_fraction <= 0.05 and mean_confidence >= 0.66:
            status, reason = "BLOCKED", "No walkable surface is visible in this region"
        else:
            status, reason = "UNCERTAIN", "Walkable surface evidence is mixed or insufficient"

        matrix.append({
            "id": region_id,
            "name": label,
            "row": row,
            "column": col,
            "bbox_xyxy": [x1, y1, x2, y2],
            "objects": objects,
            "object_count": len(objects),
            "detection_confidence": round(float(np.mean([o["confidence"] for o in objects])), 4) if objects else None,
            "walkable_coverage": round(walk_fraction, 4),
            "road_coverage": round(road_fraction, 4),
            "semantic_confidence": round(mean_confidence, 4),
            "obstacle_present": obstacle_present,
            "status": status,
            "reason": reason,
        })
    return matrix


def build_distance_matrix(frame_shape, rows, columns, objects):
    """Return nearest measured distance per cell, using box centers for assignment."""
    height, width = frame_shape[:2]
    matrix = [[-1.0] * columns for _ in range(rows)]
    for box, distance in objects:
        if distance is None or not np.isfinite(distance) or distance <= 0:
            continue
        x = (box[0] + box[2]) / 2
        y = (box[1] + box[3]) / 2
        column = min(columns - 1, max(0, int(x * columns / width)))
        row = min(rows - 1, max(0, int(y * rows / height)))
        previous = matrix[row][column]
        if previous < 0 or distance < previous:
            matrix[row][column] = round(float(distance), 2)
    return matrix


def render_distance_matrix(matrix):
    """Render the numeric values written to the hardware stream."""
    rows, columns = len(matrix), len(matrix[0])
    cell_width, cell_height = 70, 44
    panel = np.full((rows * cell_height, columns * cell_width, 3), 32, dtype=np.uint8)
    draw_grid_lines(panel, rows, columns)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for row, values in enumerate(matrix):
        for column, distance in enumerate(values):
            label = "-1" if distance < 0 else f"{distance:.2f}"
            (width, height), _ = cv2.getTextSize(label, font, 0.56, 1)
            x = column * cell_width + (cell_width - width) // 2
            y = row * cell_height + (cell_height + height) // 2
            color = (160, 160, 160) if distance < 0 else (180, 255, 180)
            cv2.putText(panel, label, (x, y), font, 0.56, color, 1, cv2.LINE_AA)
    return panel


def draw_grid_lines(frame: np.ndarray, rows: int, columns: int) -> None:
    """Draw cell boundaries in place on a video frame."""
    height, width = frame.shape[:2]
    for column in range(1, columns):
        x = column * width // columns
        cv2.line(frame, (x, 0), (x, height - 1), (220, 220, 220), 1, cv2.LINE_AA)
    for row in range(1, rows):
        y = row * height // rows
        cv2.line(frame, (0, y), (width - 1, y), (220, 220, 220), 1, cv2.LINE_AA)


def draw_spatial_matrix(frame: np.ndarray, matrix: list[dict[str, Any]]) -> np.ndarray:
    """Draw cell boundaries/status tags directly in source-frame coordinates."""
    out = frame.copy()
    h, w = out.shape[:2]
    if len(matrix) != 9:
        raise ValueError("spatial matrix must contain exactly nine cells")
    short_names = {"TOP_LEFT": "TL", "TOP_CENTER": "TC", "TOP_RIGHT": "TR",
                   "MIDDLE_LEFT": "ML", "MIDDLE_CENTER": "MC", "MIDDLE_RIGHT": "MR",
                   "BOTTOM_LEFT": "BL", "BOTTOM_CENTER": "BC", "BOTTOM_RIGHT": "BR"}
    short_status = {"FREE": "F", "BLOCKED": "B", "UNCERTAIN": "U"}
    draw_grid_lines(out, 3, 3)
    for cell in matrix:
        x1, y1, x2, y2 = cell["bbox_xyxy"]
        color = STATUS_COLORS[cell["status"]]
        label = f"{short_names[cell['id']]}:{short_status[cell['status']]}"
        if cell["object_count"]:
            label += f" O{cell['object_count']}"
        else:
            label += f" W{cell['walkable_coverage']:.0%}"
        cell_width = x2 - x1
        nominal_scale = min(0.48, max(0.18, w / 1500))
        (nominal_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, nominal_scale, 1)
        scale = min(nominal_scale, max(0.12, (cell_width - 8) / max(1, nominal_width) * nominal_scale))
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        tx = min(x1 + 5, max(2, x2 - tw - 8))
        # Keep top-row tags below the compact navigation HUD in the upper-left.
        ty = y2 - base - 7 if cell["row"] == 0 else y1 + th + 8
        ty = min(y2 - base - 3, max(y1 + th + 5, ty))
        cv2.rectangle(out, (tx - 3, ty - th - 4), (min(w - 1, tx + tw + 4), ty + base + 2), (24, 24, 24), -1)
        cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return out
