"""Conservative three-direction route availability and recommendation."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class DirectionResult:
    direction: str
    status: str
    confidence: float
    walkable_fraction: float
    bottom_walkable_fraction: float
    road_fraction: float
    obstacle_count: int
    reason: str
    nearest_obstacle_m: float | None = None
    nearest_obstacle_spread: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def estimate_box_distance(depth_map_m, box):
    """Return median depth in meters and relative interquartile spread."""
    h, w = depth_map_m.shape
    x1, y1, x2, y2 = (int(v) for v in box)
    # Sample the lower-middle of the box to reduce background contamination.
    sx1 = max(0, min(w - 1, int(x1 + (x2 - x1) * 0.25)))
    sx2 = max(sx1 + 1, min(w, int(x1 + (x2 - x1) * 0.75)))
    sy1 = max(0, min(h - 1, int(y1 + (y2 - y1) * 0.45)))
    sy2 = max(sy1 + 1, min(h, int(y1 + (y2 - y1) * 0.90)))
    values = depth_map_m[sy1:sy2, sx1:sx2]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 8:
        return None, None
    q25, median, q75 = np.percentile(values, (25, 50, 75))
    return round(float(median), 2), round(float((q75 - q25) / max(median, 1e-3)), 3)


def analyze_directions(
    class_map: np.ndarray,
    confidence_map: np.ndarray,
    walkable_ids: set[int],
    road_ids: set[int],
    obstacles: list[dict[str, Any]],
    *,
    depth_map_m: np.ndarray | None = None,
    near_obstacle_m: float = 2.0,
    roi_top_fraction: float = 0.52,
) -> list[DirectionResult]:
    """Classify sidewalk corridors; roadway pixels alone never mean FREE.

    A detection blocks a direction only when its box reaches the near walking
    region and meaningfully overlaps that direction's corridor. Objects high in
    the frame do not block a route.
    """
    if class_map.ndim != 2 or confidence_map.shape != class_map.shape:
        raise ValueError("class_map and confidence_map must be same-sized 2D arrays")
    h, w = class_map.shape
    if h < 8 or w < 3:
        raise ValueError("Input map is too small for three-way analysis")
    if depth_map_m is not None:
        if depth_map_m.shape != class_map.shape:
            raise ValueError("depth map must be aligned to the segmentation map")
        for obj in obstacles:
            obj["distance_m"], obj["distance_spread"] = estimate_box_distance(depth_map_m, obj["box"])
    else:
        for obj in obstacles:
            obj["distance_m"] = None
            obj["distance_spread"] = None
    y0 = max(0, min(h - 1, int(h * roi_top_fraction)))
    ymid = y0 + int((h - y0) * 0.72)
    names = ("LEFT", "CENTER", "RIGHT")
    results: list[DirectionResult] = []

    for i, name in enumerate(names):
        x1, x2 = i * w // 3, (i + 1) * w // 3
        lane = (x1, y0, x2, h)
        ids = class_map[y0:h, x1:x2]
        conf = confidence_map[y0:h, x1:x2]
        sidewalk = np.isin(ids, list(walkable_ids))
        road = np.isin(ids, list(road_ids))
        walk_fraction = float(sidewalk.mean()) if sidewalk.size else 0.0
        road_fraction = float(road.mean()) if road.size else 0.0
        bottom_sidewalk = sidewalk[max(0, ymid - y0):, :]
        bottom_fraction = float(bottom_sidewalk.mean()) if bottom_sidewalk.size else 0.0
        mean_walk_conf = float(conf[sidewalk].mean()) if sidewalk.any() else 0.0
        mean_scene_conf = float(conf.mean()) if conf.size else 0.0

        relevant: list[dict[str, Any]] = []
        uncertain_obstacles: list[dict[str, Any]] = []
        nearest_obstacle: float | None = None
        nearest_spread: float | None = None
        lane_area = max(1, (x2 - x1) * (h - y0))
        for obj in obstacles:
            if obj.get("navigation_obstacle", True) is False:
                continue
            box = tuple(int(v) for v in obj["box"])
            hit = _overlap(lane, box)
            if not hit:
                continue
            hx1, _, hx2, _ = hit
            hit_area = (hx2 - hx1) * (hit[3] - hit[1])
            reaches_near_field = box[3] >= int(h * 0.70)
            meaningful_overlap = hit_area / lane_area >= 0.035
            spans_lane = (hx2 - hx1) / max(1, x2 - x1) >= 0.16
            is_corridor_overlap = meaningful_overlap or spans_lane
            if depth_map_m is not None:
                obj_distance = obj.get("distance_m")
                if obj_distance is not None:
                    obj_distance = float(obj_distance)
                    if nearest_obstacle is None or obj_distance < nearest_obstacle:
                        nearest_obstacle = obj_distance
                        nearest_spread = obj.get("distance_spread")
                relevant_by_depth = obj_distance is not None and obj_distance <= near_obstacle_m
            else:
                obj["distance_m"] = None
                relevant_by_depth = False
            # Depth can promote a spatially intersecting object to an immediate blocker.
            if is_corridor_overlap:
                if depth_map_m is not None:
                    if relevant_by_depth:
                        relevant.append(obj)
                    elif reaches_near_field:
                        uncertain_obstacles.append(obj)
                elif reaches_near_field:
                    relevant.append(obj)

        if relevant:
            status, reason = "BLOCKED", "Obstacle overlaps the near walking corridor"
            decision_conf = float(np.mean([o["confidence"] for o in relevant]))
        elif uncertain_obstacles:
            status, reason = "UNCERTAIN", "Obstacle intersects the corridor but is beyond the near-distance threshold"
            decision_conf = float(np.mean([o["confidence"] for o in uncertain_obstacles]))
        elif mean_scene_conf < 0.48:
            status, reason = "UNCERTAIN", "Scene segmentation confidence is low"
            decision_conf = mean_scene_conf
        elif walk_fraction >= 0.22 and bottom_fraction >= 0.18 and mean_walk_conf >= 0.52:
            status, reason = "FREE", "Sidewalk surface is visible through the near field"
            decision_conf = min(mean_walk_conf, 0.5 + 0.5 * min(walk_fraction, bottom_fraction))
        elif road_fraction >= 0.20 and walk_fraction < 0.22:
            # A vehicle road can look visually open while still being unsafe to walk on.
            status, reason = "UNCERTAIN", "Roadway is visible; pedestrian safety is not established"
            decision_conf = float(np.mean(conf[road])) if road.any() else mean_scene_conf
        elif walk_fraction <= 0.05 and road_fraction <= 0.05 and mean_scene_conf >= 0.66:
            status, reason = "BLOCKED", "No sidewalk or roadway surface is visible in this corridor"
            decision_conf = mean_scene_conf
        else:
            status, reason = "UNCERTAIN", "Walkable sidewalk evidence is mixed or insufficient"
            decision_conf = min(mean_scene_conf, max(mean_walk_conf, 0.35))

        results.append(DirectionResult(
            direction=name,
            status=status,
            confidence=round(float(np.clip(decision_conf, 0, 1)), 3),
            walkable_fraction=round(walk_fraction, 3),
            bottom_walkable_fraction=round(bottom_fraction, 3),
            road_fraction=round(road_fraction, 3),
            obstacle_count=len(relevant),
            reason=reason,
            nearest_obstacle_m=round(nearest_obstacle, 2) if nearest_obstacle is not None else None,
            nearest_obstacle_spread=round(float(nearest_spread), 3) if nearest_spread is not None else None,
        ))
    return results


def choose_recommendation(directions: list[DirectionResult]) -> str | None:
    """Prefer a clear center sidewalk, otherwise choose the strongest free side."""
    by_name = {item.direction: item for item in directions}
    center = by_name.get("CENTER")
    if center and center.status == "FREE":
        return "CENTER"
    safe_sides = [by_name[name] for name in ("LEFT", "RIGHT") if name in by_name and by_name[name].status == "FREE"]
    if not safe_sides:
        return None
    return max(safe_sides, key=lambda item: (item.walkable_fraction * item.confidence, item.confidence)).direction


class TemporalStatusSmoother:
    """Conservative hysteresis: BLOCKED is immediate; FREE needs repeated evidence."""

    def __init__(self, free_confirm_frames: int = 2) -> None:
        self.free_confirm_frames = max(1, int(free_confirm_frames))
        self._free_streak = {name: 0 for name in ("LEFT", "CENTER", "RIGHT")}

    def apply(self, decisions: list[DirectionResult]) -> list[DirectionResult]:
        for item in decisions:
            if item.status == "BLOCKED":
                self._free_streak[item.direction] = 0
            elif item.status == "FREE":
                self._free_streak[item.direction] += 1
                if self._free_streak[item.direction] < self.free_confirm_frames:
                    item.status = "UNCERTAIN"
                    item.reason = "Sidewalk evidence needs confirmation across more frames"
            else:
                self._free_streak[item.direction] = 0
        return decisions
