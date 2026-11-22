import unittest

import numpy as np

from allyvision.navigation import TemporalStatusSmoother, analyze_directions, choose_recommendation


class NavigationLogicTests(unittest.TestCase):
    def setUp(self):
        self.confidence = np.full((90, 90), 0.9, dtype=np.float32)

    def test_clear_road_corridors_are_free(self):
        classes = np.ones((90, 90), dtype=np.uint8)  # Cityscapes sidewalk ID
        result = analyze_directions(classes, self.confidence, {1}, {0}, [])
        self.assertEqual([r.status for r in result], ["FREE", "FREE", "FREE"])
        self.assertEqual(choose_recommendation(result), "CENTER")

    def test_near_field_obstacle_blocks_only_its_corridor(self):
        classes = np.ones((90, 90), dtype=np.uint8)
        obstacle = {"label": "person", "confidence": 0.9, "box": [5, 60, 25, 85]}
        result = analyze_directions(classes, self.confidence, {1}, {0}, [obstacle])
        self.assertEqual([r.status for r in result], ["BLOCKED", "FREE", "FREE"])

    def test_distant_object_does_not_block_walking_corridor(self):
        classes = np.ones((90, 90), dtype=np.uint8)
        obstacle = {"label": "person", "confidence": 0.9, "box": [5, 10, 25, 40]}
        result = analyze_directions(classes, self.confidence, {1}, {0}, [obstacle])
        self.assertEqual([r.status for r in result], ["FREE", "FREE", "FREE"])

    def test_confident_nonwalkable_surface_is_blocked(self):
        classes = np.full((90, 90), 2, dtype=np.uint8)  # Cityscapes building ID
        result = analyze_directions(classes, self.confidence, {1}, {0}, [])
        self.assertEqual([r.status for r in result], ["BLOCKED", "BLOCKED", "BLOCKED"])

    def test_mixed_surface_is_uncertain(self):
        classes = np.full((90, 90), 2, dtype=np.uint8)
        classes[85:, :] = 1
        result = analyze_directions(classes, self.confidence, {1}, {0}, [])
        self.assertEqual([r.status for r in result], ["UNCERTAIN", "UNCERTAIN", "UNCERTAIN"])

    def test_center_is_recommended_when_it_is_the_only_open_sidewalk(self):
        classes = np.full((90, 90), 2, dtype=np.uint8)
        classes[46:, 30:60] = 1
        result = analyze_directions(classes, self.confidence, {1}, {0}, [])
        self.assertEqual([r.status for r in result], ["BLOCKED", "FREE", "BLOCKED"])
        self.assertEqual(choose_recommendation(result), "CENTER")

    def test_road_surface_alone_is_not_reported_as_free(self):
        classes = np.zeros((90, 90), dtype=np.uint8)  # Cityscapes road ID
        result = analyze_directions(classes, self.confidence, {1}, {0}, [])
        self.assertEqual([r.status for r in result], ["UNCERTAIN", "UNCERTAIN", "UNCERTAIN"])
        self.assertIsNone(choose_recommendation(result))

    def test_free_requires_repeated_frames_but_blocked_is_immediate(self):
        classes = np.ones((90, 90), dtype=np.uint8)
        smoother = TemporalStatusSmoother(free_confirm_frames=2)
        first = smoother.apply(analyze_directions(classes, self.confidence, {1}, {0}, []))
        self.assertEqual([r.status for r in first], ["UNCERTAIN"] * 3)
        second = smoother.apply(analyze_directions(classes, self.confidence, {1}, {0}, []))
        self.assertEqual([r.status for r in second], ["FREE"] * 3)
        obstacle = {"label": "person", "confidence": 0.9, "box": [5, 60, 25, 85]}
        blocked = smoother.apply(analyze_directions(classes, self.confidence, {1}, {0}, [obstacle]))
        self.assertEqual(blocked[0].status, "BLOCKED")


if __name__ == "__main__":
    unittest.main()
