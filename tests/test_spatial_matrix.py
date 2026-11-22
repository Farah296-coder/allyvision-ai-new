import unittest

import numpy as np

from allyvision.pipeline import process_frame
from allyvision.spatial_matrix import build_spatial_matrix


class SpatialMatrixTests(unittest.TestCase):
    def test_object_extent_is_assigned_to_every_overlapped_cell(self):
        classes = np.ones((90, 90), dtype=np.uint8)
        confidence = np.full((90, 90), 0.9, dtype=np.float32)
        obj = {"label": "car", "confidence": 0.88, "box": [25, 25, 65, 60], "navigation_obstacle": True,
               "distance_m": 4.5, "distance_spread": 0.2}
        matrix = build_spatial_matrix((90, 90, 3), classes, confidence, {1}, {0}, [obj])
        self.assertEqual(len(matrix), 9)
        self.assertEqual([c["id"] for c in matrix], [
            "TOP_LEFT", "TOP_CENTER", "TOP_RIGHT", "MIDDLE_LEFT", "MIDDLE_CENTER",
            "MIDDLE_RIGHT", "BOTTOM_LEFT", "BOTTOM_CENTER", "BOTTOM_RIGHT",
        ])
        occupied = {cell["id"] for cell in matrix if cell["objects"]}
        self.assertEqual(occupied, {
            "TOP_LEFT", "TOP_CENTER", "TOP_RIGHT", "MIDDLE_LEFT", "MIDDLE_CENTER", "MIDDLE_RIGHT",
        })
        for cell in matrix:
            if cell["objects"]:
                self.assertEqual(cell["status"], "BLOCKED")
                self.assertEqual(cell["objects"][0]["estimated_distance_m"], 4.5)
                self.assertIn("overlap_bbox_xyxy", cell["objects"][0])

    def test_depth_is_hidden_when_distance_spread_is_unreliable(self):
        classes = np.ones((30, 30), dtype=np.uint8)
        confidence = np.full((30, 30), 0.9, dtype=np.float32)
        obj = {"label": "person", "confidence": 0.9, "box": [2, 2, 8, 8], "navigation_obstacle": True,
               "distance_m": 2.0, "distance_spread": 0.8}
        matrix = build_spatial_matrix((30, 30, 3), classes, confidence, {1}, set(), [obj])
        self.assertIsNone(matrix[0]["objects"][0]["estimated_distance_m"])

    def test_matrix_is_attached_to_each_pipeline_result_and_render(self):
        class Models:
            device = "cpu"
            walkable_ids = {1}
            road_ids = {0}

            def segment(self, _image):
                return np.ones((90, 90), dtype=np.uint8), np.full((90, 90), 0.9, dtype=np.float32)

            def detect(self, _frame, **_kwargs):
                return [{"label": "person", "confidence": 0.9, "box": [10, 10, 35, 45],
                         "navigation_obstacle": True, "distance_m": None}]

        frame = np.full((90, 90, 3), 60, dtype=np.uint8)
        payload, preview, _ = process_frame(frame, Models())
        self.assertEqual(len(payload["spatial_matrix"]), 9)
        self.assertGreater(preview.shape[0], 0)
        self.assertEqual(payload["spatial_matrix"][0]["objects"][0]["class"], "person")


if __name__ == "__main__":
    unittest.main()
