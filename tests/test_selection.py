import unittest

from road_damage.selection import select_models


def report(model, accuracy, latency, train=100, **overrides):
    value = {"model": f"yolov8{model}-seg", "dataset": "crack", "stage": "pilot30",
             "status": "completed", "epochs_requested": 30, "epochs_completed": 30,
             "validation": {"mask": {"map50_95": accuracy}},
             "benchmark": {"mean_ms": latency}, "train_wall_seconds": train}
    value.update(overrides)
    return value


class SelectionTests(unittest.TestCase):
    def test_balanced_selection_and_frontier(self):
        result = select_models([report("n", .40, 5, 100), report("s", .415, 7, 150), report("m", .42, 12, 200)])
        self.assertEqual(result["recommended"], "yolov8n-seg")
        self.assertEqual(result["best_accuracy"], "yolov8m-seg")
        self.assertEqual(len(result["pareto_frontier"]), 3)

    def test_accuracy_guard_excludes_fast_but_weak(self):
        result = select_models([report("n", .35, 5), report("s", .405, 7), report("m", .42, 12)])
        self.assertEqual(result["recommended"], "yolov8s-seg")

    def test_partial_and_failed_do_not_rank(self):
        for third in (report("m", .42, 12, status="failed"), report("m", .42, 12, epochs_completed=29)):
            result = select_models([report("n", .40, 5), report("s", .41, 7), third])
            self.assertFalse(result["complete"])
            self.assertIsNone(result["recommended"])

    def test_nonfinite_metric_cannot_win(self):
        self.assertFalse(select_models([report("n", float("nan"), 5), report("s", .41, 7), report("m", .42, 12)])["complete"])

    def test_dominated_model_and_tie_break(self):
        result = select_models([report("n", .42, 5, 100), report("s", .42, 5, 120), report("m", .41, 9, 200)])
        self.assertEqual(result["recommended"], "yolov8n-seg")
        self.assertEqual(result["pareto_frontier"], ["yolov8n-seg"])

    def test_mixed_datasets_rejected(self):
        with self.assertRaises(ValueError):
            select_models([report("n", .40, 5), report("s", .41, 7, dataset="pothole")])

    def test_different_samples_not_comparable(self):
        records = [report("n", .40, 5), report("s", .41, 7), report("m", .42, 12)]
        records[-1]["benchmark"]["manifest_sha256"] = "different"
        self.assertFalse(select_models(records)["complete"])


if __name__ == "__main__":
    unittest.main()
