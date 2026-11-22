import unittest

import numpy as np

from allyvision.pipeline import process_frame


class FakeModels:
    device = "cpu"
    walkable_ids = {1}
    road_ids = {0}

    def segment(self, _image):
        classes = np.full((90, 90), 2, dtype=np.uint8)
        classes[46:, 30:60] = 1
        confidence = np.full((90, 90), 0.9, dtype=np.float32)
        return classes, confidence

    def detect(self, _frame, **_kwargs):
        return []


class PipelineVisualizationTests(unittest.TestCase):
    def test_clear_center_sidewalk_is_recommended_and_colored_green(self):
        frame = np.full((90, 90, 3), 60, dtype=np.uint8)
        payload, preview, _ = process_frame(frame, FakeModels())
        self.assertEqual(payload["recommended_direction"], "CENTER")
        self.assertEqual([d["status"] for d in payload["directions"]], ["BLOCKED", "FREE", "BLOCKED"])
        self.assertEqual(preview.shape, frame.shape)
        free_pixel = preview[85, 55]
        self.assertGreater(int(free_pixel[1]), int(free_pixel[2]))  # visible green walkable overlay

    def test_six_panel_view_remains_available_for_debugging(self):
        frame = np.full((90, 90, 3), 60, dtype=np.uint8)
        _, preview, _ = process_frame(frame, FakeModels(), visualization="debug")
        self.assertEqual(preview.shape, (720, 1920, 3))

    def test_depth_map_is_reported_for_near_obstacles(self):
        frame = np.full((90, 90, 3), 60, dtype=np.uint8)
        models = FakeModels()
        models.detect = lambda *_args, **_kwargs: [{"label": "person", "confidence": 0.9, "box": [35, 35, 55, 85]}]
        depth = np.full((90, 90), 1.4, dtype=np.float32)
        payload, _, _ = process_frame(frame, models, depth_map_m=depth)
        center = payload["directions"][1]
        self.assertEqual(center["status"], "BLOCKED")
        self.assertEqual(center["nearest_obstacle_m"], 1.4)

    def test_far_intersecting_obstacle_is_uncertain_not_free_or_blocked(self):
        frame = np.full((90, 90, 3), 60, dtype=np.uint8)
        models = FakeModels()
        models.detect = lambda *_args, **_kwargs: [{"label": "person", "confidence": 0.9, "box": [35, 35, 55, 85]}]
        depth = np.full((90, 90), 8.0, dtype=np.float32)
        payload, _, _ = process_frame(frame, models, depth_map_m=depth)
        self.assertEqual(payload["directions"][1]["status"], "UNCERTAIN")

    def test_arbitrary_detected_class_is_reported_but_not_automatically_a_path_obstacle(self):
        frame = np.full((90, 90, 3), 60, dtype=np.uint8)
        models = FakeModels()
        models.detect = lambda *_args, **_kwargs: [{
            "label": "sports ball", "confidence": 0.81, "box": [35, 55, 55, 80],
            "distance_m": None, "navigation_obstacle": False,
        }]
        payload, _, _ = process_frame(frame, models)
        self.assertEqual(payload["object_message"], "OBJECTS DETECTED (1): sports ball")
        self.assertEqual(payload["detected_objects"][0]["label"], "sports ball")
        self.assertEqual(payload["obstacles"], [])

    def test_empty_scene_reports_no_object_detected(self):
        frame = np.full((90, 90, 3), 60, dtype=np.uint8)
        payload, _, _ = process_frame(frame, FakeModels())
        self.assertEqual(payload["object_message"], "NO OBJECT DETECTED")
        self.assertEqual(payload["detected_objects"], [])


if __name__ == "__main__":
    unittest.main()
