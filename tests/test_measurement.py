import json
import math
import unittest

import numpy as np

from road_damage.measurement import measure_masks, skeleton_length


class SkeletonTests(unittest.TestCase):
    def test_straight_diagonal_and_single_pixel_graph_lengths(self):
        self.assertEqual(skeleton_length(np.array([[1, 1, 1, 1, 1]], bool)), 4.0)
        self.assertAlmostEqual(skeleton_length(np.eye(5, dtype=bool)), 4 * math.sqrt(2))
        self.assertEqual(skeleton_length(np.ones((1, 1), bool)), 0.0)

    def test_corner_has_no_diagonal_shortcut(self):
        corner = np.array([[1, 1], [0, 1]], bool)
        self.assertEqual(skeleton_length(corner), 2.0)

    def test_branch_counts_each_edge_once(self):
        branch = np.zeros((5, 5), dtype=bool)
        branch[2, :] = True
        branch[:, 2] = True
        self.assertEqual(skeleton_length(branch), 8.0)


class MaskMeasurementTests(unittest.TestCase):
    def test_rectangle_area_and_overlapping_union_on_nonsquare_image(self):
        masks = np.zeros((2, 6, 10), dtype=bool)
        masks[0, 1:4, 2:6] = True
        masks[1, 1:4, 4:8] = True
        result = measure_masks(masks, task="pothole", original_shape=(6, 10))
        self.assertEqual([item["area_px2"] for item in result["instances"]], [12, 12])
        self.assertEqual(result["total_area_px2"], 18)
        json.dumps(result)

    def test_crack_union_does_not_count_overlapping_instances_twice(self):
        masks = np.zeros((2, 5, 10), dtype=bool)
        masks[0, 2, 1:6] = True
        masks[1, 2, 4:9] = True
        result = measure_masks(masks, task="crack", original_shape=(5, 10))
        self.assertEqual([item["length_px"] for item in result["instances"]], [4.0, 4.0])
        self.assertEqual(result["total_length_px"], 7.0)

    def test_class_filter_and_no_predictions(self):
        masks = np.ones((2, 4, 6), dtype=bool)
        filtered = measure_masks(masks, task="pothole", class_ids=[0, 1], target_class_id=1)
        self.assertEqual(filtered["instance_count"], 1)
        self.assertEqual(filtered["instances"][0]["instance_index"], 1)
        empty = measure_masks([], task="pothole", original_shape=(4, 6))
        self.assertEqual(empty["total_area_px2"], 0)
        self.assertEqual(empty["instances"], [])
        empty_crack = measure_masks(np.zeros((1, 4, 6)), task="crack")
        self.assertEqual(empty_crack["total_length_px"], 0.0)

    def test_resolution_mismatch_and_invalid_input_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "differs from original"):
            measure_masks(np.ones((1, 5, 7)), original_shape=(10, 14))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            measure_masks(np.array([[np.nan]]))
        with self.assertRaisesRegex(ValueError, "one class ID"):
            measure_masks(np.zeros((2, 5, 7)), class_ids=[0])
        with self.assertRaisesRegex(ValueError, "required"):
            measure_masks(np.zeros((2, 5, 7)), target_class_id=0)


if __name__ == "__main__":
    unittest.main()
