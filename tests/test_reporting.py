"""Report acceptance tests: truthful pending state and separate comparisons."""

import json
from pathlib import Path
import tempfile
import unittest

from road_damage.reporting import build_comparison


def result(dataset, model, accuracy, latency):
    return {
        "schema_version": 1, "dataset": dataset, "model": model,
        "stage": "pilot30", "status": "completed", "epochs_requested": 30,
        "epochs_completed": 30, "train_wall_seconds": 7200, "batch": 8,
        "validation": {"split": "val", "mask": {
            "precision": 0.8, "recall": 0.7, "map50": 0.75,
            "map50_95": accuracy,
        }},
        "benchmark": {"mean_ms": latency, "median_ms": latency,
                      "p95_ms": latency + 1, "fps": 1000 / latency},
    }


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def put(self, dataset, model, accuracy, latency, **overrides):
        report = result(dataset, model, accuracy, latency)
        report.update(overrides)
        path = self.root / "runs" / "seg" / "pilot30" / dataset / model.replace("-", "_") / "report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_empty_report_is_portable_and_has_no_invented_zero_metrics(self):
        outcome = build_comparison(self.root)
        html = Path(outcome["html_path"]).read_text(encoding="utf-8")
        self.assertEqual(outcome["status"], "incomplete")
        self.assertEqual(outcome["completed"], 0)
        self.assertEqual(outcome["expected"], 6)
        self.assertEqual(html.count("Chưa đề xuất mô hình"), 2)
        self.assertNotIn("<td>0.00</td>", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("<script src", html)
        self.assertIn('lang="vi"', html)
        self.assertIn("không phải điểm 0", html)

    def test_six_complete_runs_choose_within_dataset_not_global_accuracy(self):
        for dataset, scores in (("crack", (.40, .415, .42)), ("pothole", (.65, .685, .70))):
            for model, score, latency in zip(("yolov8n-seg", "yolov8s-seg", "yolov8m-seg"), scores, (5, 7, 12)):
                self.put(dataset, model, score, latency)
        outcome = build_comparison(self.root)
        self.assertEqual(outcome["status"], "completed")
        self.assertEqual(outcome["datasets"]["crack"]["selection"]["recommended"], "yolov8n-seg")
        self.assertEqual(outcome["datasets"]["pothole"]["selection"]["recommended"], "yolov8s-seg")
        html = Path(outcome["html_path"]).read_text(encoding="utf-8")
        self.assertEqual(html.count("<svg "), 4)
        self.assertIn("kết quả sơ bộ", html)
        self.assertIn("80.00", html)
        self.assertIn("../runs/seg/pilot30/crack/yolov8n_seg/report.json", html)

    def test_short_run_does_not_create_recommendation(self):
        self.put("crack", "yolov8n-seg", .4, 5)
        self.put("crack", "yolov8s-seg", .42, 7)
        self.put("crack", "yolov8m-seg", .45, 9, epochs_completed=29)
        outcome = build_comparison(self.root)
        self.assertEqual(outcome["datasets"]["crack"]["completed"], 2)
        self.assertIsNone(outcome["datasets"]["crack"]["selection"]["recommended"])

    def test_nonfinite_values_missing_benchmark_and_test_split_cannot_win(self):
        self.put("crack", "yolov8n-seg", float("nan"), 5)
        self.put("crack", "yolov8s-seg", .42, 7, benchmark={"mean_ms": 7})
        invalid = result("crack", "yolov8m-seg", .45, 9)["validation"]
        invalid["split"] = "test"
        self.put("crack", "yolov8m-seg", .45, 9, validation=invalid)
        self.assertEqual(build_comparison(self.root)["completed"], 0)

    def test_unreadable_or_wrong_identity_is_visible_not_fatal(self):
        path = self.put("crack", "yolov8n-seg", .4, 5)
        path.write_text("{", encoding="utf-8")
        self.put("crack", "yolov8s-seg", .42, 7, stage="final100")
        outcome = build_comparison(self.root)
        self.assertEqual(outcome["completed"], 0)
        self.assertEqual(outcome["results"][0]["status"], "invalid")
        self.assertEqual(outcome["results"][1]["status"], "invalid")

    def test_mismatched_benchmark_cannot_be_marked_complete(self):
        for dataset in ("crack", "pothole"):
            for model, latency in zip(("yolov8n-seg", "yolov8s-seg", "yolov8m-seg"), (5, 7, 12)):
                report = result(dataset, model, .42, latency)
                report["benchmark"]["manifest_sha256"] = model if dataset == "crack" else "same"
                self.put(dataset, model, .42, latency, benchmark=report["benchmark"])
        outcome = build_comparison(self.root)
        self.assertEqual(outcome["completed"], 6)
        self.assertEqual(outcome["status"], "incomplete")
        html = Path(outcome["html_path"]).read_text(encoding="utf-8")
        self.assertIn("Chưa bảo đảm cùng ngân sách", html)

    def test_untrusted_source_text_is_escaped(self):
        self.put("crack", "yolov8n-seg", .4, 5, errors=["<script>alert('source')</script>"])
        html = Path(build_comparison(self.root)["html_path"]).read_text(encoding="utf-8")
        self.assertNotIn("<script>alert", html)
        self.assertIn("&lt;script&gt;", html)


if __name__ == "__main__":
    unittest.main()
