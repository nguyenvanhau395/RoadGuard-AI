import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import test_interface as interface  # noqa: E402


class MvpInterfaceTests(unittest.TestCase):
    @staticmethod
    def _email_report_fixture():
        return {
            "input": {"name": "doan-duong-01.mp4", "kind": "video"},
            "calibration": {
                "enabled": True, "road_width_m": 3.5, "road_length_m": 10,
            },
            "summary": {
                "instances": 2,
                "by_class": {"Nứt dọc": 1, "Ổ gà": 1},
                "longitudinal_crack_length_px": 850.25,
                "transverse_crack_length_px": 0,
                "total_crack_length_px": 850.25,
                "total_pothole_area_px2": 4500,
                "total_crack_length_m": 2.75,
                "total_pothole_area_m2": 0.18,
                "segmentation": {"measured": 2, "no_mask": 0},
            },
            "detections_unique": [
                {
                    "track_id": 4, "class_code": "D00", "class_name": "Nứt dọc",
                    "confidence": 0.88, "frame": 29, "measurement_status": "measured",
                    "measurement_type": "length_px", "measurement_value_px": 850.25,
                    "measurement_value_metric": 2.75, "measurement_unit_metric": "m",
                },
                {
                    "track_id": 9, "class_code": "D40", "class_name": "Ổ gà",
                    "confidence": 0.82, "frame": 493, "measurement_status": "measured",
                    "measurement_type": "area_px2", "measurement_value_px": 4500,
                    "measurement_value_metric": 0.18, "measurement_unit_metric": "m²",
                },
            ],
        }

    def test_all_three_checkpoints_exist(self):
        payload = interface._model_payload()
        self.assertEqual([item["id"] for item in payload], ["det", "seg_crack", "seg_pothole"])
        self.assertTrue(all(item["exists"] for item in payload))

    def test_upload_size_limit_is_applied_to_file_bytes(self):
        self.assertEqual(
            interface._validate_upload_size(interface.MAX_UPLOAD_BYTES),
            interface.MAX_UPLOAD_BYTES,
        )
        with self.assertRaisesRegex(ValueError, "200 MB"):
            interface._validate_upload_size(interface.MAX_UPLOAD_BYTES + 1)

    def test_analysis_job_runs_in_background_and_exposes_result(self):
        started = threading.Event()
        release = threading.Event()

        def fake_analysis(*_args, **_kwargs):
            started.set()
            release.wait(timeout=2)
            return {"run_id": "result-01", "summary": {"instances": 2}}

        with tempfile.TemporaryDirectory() as directory:
            upload = Path(directory) / "input.mp4"
            upload.write_bytes(b"video")
            with patch.object(interface, "_execute_analysis", side_effect=fake_analysis), patch.object(
                interface, "_cleanup_uploaded_file"
            ) as cleanup:
                job_id = interface._start_analysis_job(
                    upload, "input.mp4", "pipeline", "det", 0.25, 0.12, None
                )
                self.assertTrue(started.wait(timeout=1))
                self.assertEqual(interface._job_payload(job_id)["status"], "running")
                release.set()
                deadline = time.time() + 2
                while time.time() < deadline:
                    payload = interface._job_payload(job_id)
                    if payload["status"] == "completed":
                        break
                    time.sleep(0.01)

        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["result"]["run_id"], "result-01")
        cleanup.assert_called_once_with(upload)

    def test_history_lists_newest_reports_and_reopens_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = [
                ("older", "2026-10-20T01:00:00+00:00", "old.jpg", "image", 1),
                ("newer", "2026-10-22T09:00:00+00:00", "new.mp4", "video", 4),
            ]
            for run_id, created_at, filename, kind, instances in fixtures:
                run_dir = root / run_id
                run_dir.mkdir()
                media_name = "result.mp4" if kind == "video" else "result.jpg"
                (run_dir / media_name).write_bytes(b"media")
                (run_dir / "report.json").write_text(
                    json.dumps({
                        "schema_version": 1,
                        "created_at": created_at,
                        "mode": "pipeline",
                        "input": {"name": filename, "kind": kind, "width": 640, "height": 360},
                        "summary": {"instances": instances, "segmentation": {"measured": instances, "no_mask": 0}},
                        "detections": [],
                        "detections_unique": [],
                        "density": [],
                        "elapsed_ms": 1234,
                    }),
                    encoding="utf-8",
                )

            history = interface._history_payload(root=root, limit=10)
            reopened = interface._history_result("newer", root=root)

        self.assertEqual([item["run_id"] for item in history], ["newer", "older"])
        self.assertEqual(history[0]["input_name"], "new.mp4")
        self.assertEqual(history[0]["instances"], 4)
        self.assertEqual(reopened["url"], "/media/newer/result.mp4")
        self.assertEqual(reopened["summary"]["instances"], 4)
        self.assertEqual(reopened["csv_url"], "/api/export/newer?format=csv")

    def test_frontend_guards_large_uploads_and_has_history_ui(self):
        html = interface.WEB_FILE.read_text(encoding="utf-8")
        self.assertIn("MAX_UPLOAD_BYTES", html)
        self.assertIn("response.text()", html)
        self.assertIn('id="historyList"', html)
        self.assertIn("/api/history", html)

    def test_local_chatbot_env_loads_server_settings_without_overwriting_process_env(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".roadguard.env"
            env_file.write_text(
                "CHATBOT_API_KEY=file-key\n"
                "CHATBOT_BASE_URL=https://example.test/v1\n"
                "CHATBOT_MODEL=test-model\n"
                "REPORT_SMTP_HOST=smtp.example.test\n"
                "REPORT_SMTP_PORT=587\n"
                "UNRELATED=value\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"CHATBOT_API_KEY": "process-key"}, clear=True):
                interface._load_local_chatbot_env(env_file)
                self.assertEqual(os.environ["CHATBOT_API_KEY"], "process-key")
                self.assertEqual(os.environ["CHATBOT_BASE_URL"], "https://example.test/v1")
                self.assertEqual(os.environ["CHATBOT_MODEL"], "test-model")
                self.assertEqual(os.environ["REPORT_SMTP_HOST"], "smtp.example.test")
                self.assertEqual(os.environ["REPORT_SMTP_PORT"], "587")
                self.assertNotIn("UNRELATED", os.environ)

    def test_report_exports_have_expected_content(self):
        report = {
            "input": {"name": "sample.jpg"},
            "detections": [{
                "frame": 0, "timestamp_s": 0, "class_code": "D10",
                "class_name": "Nứt ngang", "confidence": 0.91,
                "x1": 1, "y1": 2, "x2": 20, "y2": 24,
                "measurement_type": "length_px", "measurement_value_px": 18.5,
            }],
            "summary": {
                "instances": 1,
                "total_crack_length_px": 18.5,
                "total_pothole_area_px2": 0,
            },
        }
        self.assertIn("Nứt ngang".encode("utf-8"), interface._csv_bytes(report))
        xlsx = interface._xlsx_bytes(report)
        self.assertTrue(xlsx.startswith(b"PK"))

    def test_detection_mapping_and_density(self):
        class Boxes:
            xyxy = [[0, 0, 100, 80]]
            cls = [0]
            conf = [0.8]

        class Result:
            boxes = Boxes()
            names = {0: "D00"}

        rows = interface._detection_rows(Result())
        self.assertEqual(rows[0]["class_name"], "Nứt dọc")
        density = interface._density(rows, 200, 100)
        self.assertAlmostEqual(density[0]["x"], 0.25)

    def test_direct_segmentation_rows_include_real_measurements(self):
        class Boxes:
            xyxy = np.array([[1, 1, 8, 5]])
            cls = np.array([0])
            conf = np.array([0.91])

            def __len__(self):
                return 1

        class Masks:
            data = np.pad(np.ones((1, 2, 4)), ((0, 0), (1, 3), (2, 4)))

        class Result:
            boxes = Boxes()
            masks = Masks()
            names = {0: "Pothole"}
            orig_shape = (6, 10)

        rows = interface._segment_rows(Result(), "seg_pothole")
        self.assertEqual(rows[0]["measurement_status"], "measured")
        self.assertEqual(rows[0]["measurement_type"], "area_px2")
        self.assertEqual(rows[0]["measurement_value_px"], 8.0)
        summary = interface._summarize(rows)
        self.assertEqual(summary["total_pothole_area_px2"], 8.0)
        self.assertEqual(summary["segmentation"]["measured"], 1)

    def test_missing_seg_mask_is_not_reported_as_zero_measurement(self):
        class Boxes:
            xyxy = np.array([[1, 1, 8, 5]])
            cls = np.array([0])
            conf = np.array([0.75])

            def __len__(self):
                return 1

        class Result:
            boxes = Boxes()
            masks = None
            names = {0: "crack"}
            orig_shape = (6, 10)

        rows = interface._segment_rows(Result(), "seg_crack")
        self.assertIsNone(rows[0]["measurement_value_px"])
        self.assertEqual(rows[0]["measurement_status"], "no_mask")

    def test_pipeline_rejects_detection_when_primary_seg_has_no_mask(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        row = {
            "class_code": "D00", "class_name": "Nứt dọc",
            "x1": 40, "y1": 40, "x2": 60, "y2": 60,
        }
        empty = np.zeros((36, 36), dtype=bool)
        model = Mock()
        model.predict.return_value = [Mock(boxes=None)]
        with patch.object(interface, "_load_model", return_value=model), patch.object(
            interface, "_measure_segment", return_value=(0.0, empty)
        ):
            interface._measure_detection(row, image, 0.12)
        self.assertEqual(row["measurement_status"], "no_mask")
        self.assertIsNone(row["measurement_value_px"])
        self.assertFalse(interface._pipeline_row_is_accepted(row))

    def test_pipeline_rejects_primary_mask_outside_det_box(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        row = {
            "class_code": "D00", "class_name": "Nứt dọc",
            "x1": 40, "y1": 40, "x2": 60, "y2": 60,
        }
        outside = np.zeros((36, 36), dtype=bool)
        outside[:5, :5] = True
        model = Mock()
        model.predict.return_value = [Mock(boxes=None)]
        with patch.object(interface, "_load_model", return_value=model), patch.object(
            interface, "_measure_segment", return_value=(25.0, outside)
        ):
            interface._measure_detection(row, image, 0.12)
        self.assertEqual(row["measurement_status"], "no_mask")
        self.assertFalse(interface._pipeline_row_is_accepted(row))

    def test_pipeline_keeps_det_only_alligator_crack(self):
        row = {"class_code": "D20", "measurement_status": "not_applicable"}
        self.assertTrue(interface._pipeline_row_is_accepted(row))

    def test_pipeline_accepts_only_primary_mask_overlapping_det_box(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        row = {
            "class_code": "D40", "class_name": "Ổ gà", "confidence": 0.7,
            "x1": 40, "y1": 40, "x2": 60, "y2": 60,
        }
        inside = np.zeros((36, 36), dtype=bool)
        inside[14:20, 14:20] = True
        model = Mock()
        model.predict.return_value = [Mock(boxes=None)]
        with patch.object(interface, "_load_model", return_value=model), patch.object(
            interface, "_measure_segment", return_value=(36.0, inside)
        ):
            interface._measure_detection(row, image, 0.12)
        self.assertEqual(row["measurement_status"], "measured")
        self.assertEqual(row["segment_strategy"], "primary_strict_overlap")
        self.assertTrue(interface._pipeline_row_is_accepted(row))
        summary = interface._summarize([row])
        self.assertEqual(summary["segmentation"]["measured"], 1)
        interface._draw_rows(image, [row])
        self.assertNotIn("_segment_visual", row)
        self.assertGreater(int(image.sum()), 0)

    def test_chatbot_has_local_report_fallback_without_secret(self):
        report = {
            "summary": {
                "instances": 2,
                "by_class": {"Ổ gà": 2},
                "total_crack_length_px": 0,
                "total_pothole_area_px2": 1250,
            },
            "detections": [],
        }
        previous = os.environ.pop("CHATBOT_API_KEY", None)
        try:
            reply = interface._chatbot(report, "Có bao nhiêu ổ gà?")
        finally:
            if previous is not None:
                os.environ["CHATBOT_API_KEY"] = previous
        self.assertIn("2 ổ gà", reply)
        self.assertIn("1250 px²", reply)

    def test_video_detection_seg_and_drawing_run_on_every_frame(self):
        class Boxes:
            xyxy = np.array([[1, 1, 8, 5]])
            cls = np.array([0])
            conf = np.array([0.8])
            id = np.array([7])

            def __len__(self):
                return 1

        class Result:
            boxes = Boxes()
            names = {0: "D00"}

        model = Mock()
        model.track.return_value = [Result()]
        frame = np.zeros((8, 10, 3), dtype=np.uint8)
        def mark_measured(row, *_args, **_kwargs):
            row["measurement_status"] = "measured"

        with patch.object(interface, "_measure_detection") as measure, patch.object(interface, "_draw_rows") as draw:
            measure.side_effect = mark_measured
            first = interface._process_pipeline_video_frame(
                model, frame.copy(), 1, 25.0, 0.25, 0.12
            )
            self.assertEqual(len(first), 1)
            measure.assert_called_once()
            draw.assert_called_once()

        with patch.object(interface, "_measure_detection") as measure, patch.object(interface, "_draw_rows") as draw:
            measure.side_effect = mark_measured
            second = interface._process_pipeline_video_frame(
                model, frame.copy(), 2, 25.0, 0.25, 0.12
            )
            self.assertEqual(len(second), 1)
            measure.assert_called_once()
            draw.assert_called_once()
        self.assertEqual(model.track.call_count, 2)
        self.assertEqual(model.track.call_args.kwargs["iou"], 0.5)
        self.assertEqual(model.track.call_args.kwargs["tracker"], "bytetrack.yaml")

    def test_video_summary_counts_each_bytetrack_id_once(self):
        rows = [
            {
                "track_id": 11, "class_code": "D00", "class_name": "Nứt dọc",
                "measurement_type": "length_px", "measurement_value_px": 12,
                "measurement_value_metric": 0.8, "measurement_unit_metric": "m",
                "measurement_status": "measured", "confidence": 0.8,
            },
            {
                "track_id": 11, "class_code": "D00", "class_name": "Nứt dọc",
                "measurement_type": "length_px", "measurement_value_px": 18,
                "measurement_value_metric": 1.2, "measurement_unit_metric": "m",
                "measurement_status": "measured", "confidence": 0.9,
            },
        ]
        summary = interface._summarize(rows)
        self.assertEqual(summary["instances"], 1)
        self.assertEqual(summary["raw_frame_detections"], 2)
        self.assertEqual(summary["unique_track_count"], 1)
        self.assertEqual(summary["total_crack_length_px"], 18)
        self.assertEqual(summary["total_crack_length_m"], 1.2)

    def test_video_density_has_one_realtime_marker_per_track(self):
        rows = [
            {
                "track_id": 11, "class_code": "D20", "class_name": "Da cá sấu",
                "frame": 10, "timestamp_s": 1.0, "center_x": 40, "center_y": 50,
                "confidence": 0.70, "measurement_status": "measured",
            },
            {
                "track_id": 11, "class_code": "D20", "class_name": "Da cá sấu",
                "frame": 12, "timestamp_s": 1.2, "center_x": 60, "center_y": 55,
                "confidence": 0.92,
            },
            {
                "track_id": 12, "class_code": "D40", "class_name": "Ổ gà",
                "frame": 50, "timestamp_s": 5.0, "center_x": 150, "center_y": 80,
                "confidence": 0.81, "measurement_status": "measured",
            },
        ]
        density = interface._density(rows, 200, 100, total_frames=101)
        self.assertEqual(len(density), 2)
        first = next(point for point in density if point["track_id"] == 11)
        self.assertEqual(first["observations"], 2)
        self.assertEqual(first["timestamp_s"], 1.0)
        self.assertEqual(first["last_timestamp_s"], 1.2)
        self.assertEqual(first["confidence"], 0.92)
        self.assertAlmostEqual(first["x"], 0.1)
        self.assertAlmostEqual(first["y"], 0.25)

    def test_transient_unsegmented_tracks_are_not_reported_as_damage(self):
        transient = [
            {
                "track_id": 41, "class_code": "D40", "class_name": "Ổ gà",
                "frame": frame, "timestamp_s": frame / 60, "center_x": 100,
                "center_y": 200, "confidence": 0.65,
                "measurement_status": "no_mask",
            }
            for frame in (492, 493)
        ]
        confirmed_by_seg = [{
            "track_id": 53, "class_code": "D40", "class_name": "Ổ gà",
            "frame": 726, "timestamp_s": 12.1, "center_x": 300,
            "center_y": 400, "confidence": 0.73,
            "measurement_status": "measured", "measurement_type": "area_px2",
            "measurement_value_px": 120,
        }]
        rows = transient + confirmed_by_seg
        summary = interface._summarize(rows)
        density = interface._density(rows, 640, 480, total_frames=800)
        self.assertEqual(summary["instances"], 1)
        self.assertEqual(summary["by_class"], {"Ổ gà": 1})
        self.assertEqual([point["track_id"] for point in density], [53])

    def test_track_id_switch_near_same_damage_is_merged(self):
        state = {"counted_ids": set(), "counts": {}, "aliases": {}, "tracks": {}, "merged_ids": set()}
        first = [{
            "track_id": 5, "class_code": "D40", "x1": 100, "y1": 100,
            "x2": 160, "y2": 160,
        }]
        interface._canonicalize_track_rows(first, state, 10, 30.0, (640, 640))
        switched = [{
            "track_id": 9, "class_code": "D40", "x1": 106, "y1": 103,
            "x2": 166, "y2": 163,
        }]
        interface._canonicalize_track_rows(switched, state, 12, 30.0, (640, 640))
        self.assertEqual(switched[0]["track_id"], 5)
        self.assertEqual(switched[0]["raw_track_id"], 9)
        self.assertTrue(switched[0]["track_merged"])
        self.assertEqual(state["merged_ids"], {9})

        far = [{
            "track_id": 10, "class_code": "D40", "x1": 480, "y1": 480,
            "x2": 540, "y2": 540,
        }]
        interface._canonicalize_track_rows(far, state, 13, 30.0, (640, 640))
        self.assertEqual(far[0]["track_id"], 10)

    def test_track_reidentification_uses_three_second_video_window(self):
        state = {"counted_ids": set(), "counts": {}, "aliases": {}, "tracks": {}, "merged_ids": set()}
        first = [{
            "track_id": 4, "class_code": "D20", "x1": 500, "y1": 2900,
            "x2": 650, "y2": 3100,
        }]
        interface._canonicalize_track_rows(first, state, 185, 60.0, (3840, 2160))
        reappeared = [{
            "track_id": 25, "class_code": "D20", "x1": 680, "y1": 2890,
            "x2": 830, "y2": 3090,
        }]
        # 135 frames at 60 FPS = 2.25 s: outside the old 2 s window,
        # but still the same nearby physical defect.
        interface._canonicalize_track_rows(reappeared, state, 320, 60.0, (3840, 2160))
        self.assertEqual(reappeared[0]["track_id"], 4)
        self.assertTrue(reappeared[0]["track_merged"])

    def test_chatbot_accepts_runtime_api_configuration_without_persisting_key(self):
        report = {"input": {"name": "test.jpg"}, "summary": {"instances": 0}, "detections": []}
        response = Mock()
        response.read.return_value = json.dumps({"choices": [{"message": {"content": "OK"}}]}).encode()
        context = Mock()
        context.__enter__ = Mock(return_value=response)
        context.__exit__ = Mock(return_value=False)
        with patch.object(interface, "urlopen", return_value=context) as open_url:
            reply = interface._chatbot(
                report, "Tóm tắt", api_key_override="test-key",
                base_url_override="https://api.example.com/v1",
                model_override="model_free",
            )
        self.assertEqual(reply, "OK")
        request = open_url.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.example.com/v1/chat/completions")
        self.assertEqual(request.headers["Authorization"], "Bearer test-key")
        self.assertIn("RoadGuardAI/1.0", request.headers["User-agent"])
        self.assertEqual(request.headers["Accept"], "application/json")

    def test_chatbot_uses_local_reply_when_provider_payload_is_invalid(self):
        report = {
            "summary": {
                "instances": 1, "by_class": {"Ổ gà": 1},
                "total_pothole_area_px2": 25, "total_crack_length_px": 0,
            },
            "detections_unique": [],
        }
        response = Mock()
        response.read.return_value = b'{"unexpected": true}'
        context = Mock()
        context.__enter__ = Mock(return_value=response)
        context.__exit__ = Mock(return_value=False)
        with patch.object(interface, "urlopen", return_value=context):
            reply = interface._chatbot(report, "Có bao nhiêu ổ gà?", api_key_override="test-key")
        self.assertIn("1 ổ gà", reply)
        self.assertIn("phản hồi không đúng định dạng", reply)

    def test_server_chat_key_cannot_be_redirected_by_client_url_or_model(self):
        payload = {
            "api_key": "", "base_url": "https://malicious.example/v1",
            "model": "wrong-model",
        }
        self.assertEqual(interface._chat_runtime_overrides(payload), (None, None, None))
        self.assertEqual(
            interface._chat_runtime_overrides({
                "api_key": "runtime-key",
                "base_url": "https://provider.example/v1",
                "model": "runtime-model",
            }),
            ("runtime-key", "https://provider.example/v1", "runtime-model"),
        )

    def test_chatbot_accepts_content_part_response(self):
        payload = {"choices": [{"message": {"content": [{"type": "text", "text": "Đã phân tích."}]}}]}
        self.assertEqual(interface._chat_response_text(payload), "Đã phân tích.")

    def test_email_narrative_asks_llm_for_analysis_and_recommendations(self):
        report = self._email_report_fixture()
        narrative = (
            "Đoạn đường có hai hư hỏng đã được SEG xác nhận. "
            "Ưu tiên xử lý ổ gà tại frame 493 để giảm nguy cơ mất an toàn."
        )
        with patch.object(interface, "_chatbot", return_value=narrative) as chatbot:
            result = interface._generate_email_narrative(report)

        self.assertEqual(result, narrative)
        chatbot.assert_called_once()
        self.assertIs(chatbot.call_args.args[0], report)
        prompt = chatbot.call_args.args[1].casefold()
        self.assertIn("phân tích", prompt)
        self.assertIn("khuyến nghị", prompt)
        self.assertIn("không", prompt)  # Prompt phải nhắc mô hình không bịa số liệu.

    def test_email_narrative_falls_back_to_local_report_if_llm_raises(self):
        report = self._email_report_fixture()
        with patch.object(interface, "_chatbot", side_effect=RuntimeError("provider unavailable")):
            narrative = interface._generate_email_narrative(report)

        self.assertIsInstance(narrative, str)
        self.assertIn("2", narrative)
        self.assertIn("Nứt dọc", narrative)
        self.assertIn("Ổ gà", narrative)

    def test_professional_email_contains_llm_narrative_and_report_attachments(self):
        report = self._email_report_fixture()
        narrative = "Phân tích AI: cần ưu tiên vá ổ gà tại frame 493."
        message = interface._build_report_email(
            report,
            recipient="nguoinhan@example.com",
            sender="roadguard@example.com",
            narrative=narrative,
        )

        self.assertEqual(message["From"], "roadguard@example.com")
        self.assertEqual(message["To"], "nguoinhan@example.com")
        self.assertIn("doan-duong-01.mp4", message["Subject"])
        plain = message.get_body(preferencelist=("plain",)).get_content()
        html = message.get_body(preferencelist=("html",)).get_content()
        self.assertIn(narrative, plain)
        self.assertIn("Phân tích AI", html)
        self.assertIn("2,75", html)
        self.assertIn("0,18", html)
        self.assertEqual(
            {part.get_filename() for part in message.iter_attachments()},
            {"bao_cao.csv", "bao_cao.xlsx"},
        )

    def test_send_report_email_uses_generated_narrative_without_real_smtp(self):
        report = self._email_report_fixture()
        smtp = Mock()
        smtp.__enter__ = Mock(return_value=smtp)
        smtp.__exit__ = Mock(return_value=False)
        env = {
            "REPORT_SMTP_HOST": "smtp.example.test",
            "REPORT_SMTP_PORT": "587",
            "REPORT_SMTP_FROM": "roadguard@example.com",
            "REPORT_SMTP_USER": "roadguard@example.com",
            "REPORT_SMTP_PASSWORD": "abcd efgh ijkl mnop",
            "REPORT_SMTP_TLS": "1",
        }
        narrative = "Nội dung phân tích do mô hình ngôn ngữ tạo."
        with patch.dict(os.environ, env, clear=True), patch.object(
            interface, "_generate_email_narrative", return_value=narrative,
        ) as generate, patch.object(interface.smtplib, "SMTP", return_value=smtp) as smtp_class:
            interface._send_report_email(report, "nguoinhan@example.com")

        generate.assert_called_once_with(report)
        smtp_class.assert_called_once_with("smtp.example.test", 587, timeout=30)
        smtp.starttls.assert_called_once_with()
        smtp.login.assert_called_once_with("roadguard@example.com", "abcdefghijklmnop")
        smtp.send_message.assert_called_once()
        sent_message = smtp.send_message.call_args.args[0]
        self.assertIn(
            narrative,
            sent_message.get_body(preferencelist=("plain",)).get_content(),
        )

    def test_four_point_calibration_converts_crack_length_to_metres(self):
        calibration = interface._prepare_calibration({
            "points": [[0, 0], [100, 0], [100, 200], [0, 200]],
            "road_width_m": 10,
            "road_length_m": 20,
        }, (200, 100))
        mask = np.zeros((20, 20), dtype=bool)
        mask[10, 3:14] = True  # ten skeleton edges at 0.1 metre each
        value = interface._metric_crack_length(mask, calibration)
        self.assertAlmostEqual(value, 1.0, places=5)

    def test_four_point_calibration_converts_pothole_area_to_square_metres(self):
        calibration = interface._prepare_calibration({
            "points": [[0, 0], [100, 0], [100, 200], [0, 200]],
            "road_width_m": 10,
            "road_length_m": 20,
        }, (200, 100))
        mask = np.zeros((30, 30), dtype=bool)
        mask[5:15, 4:24] = True  # 200 px² at 0.01 m² per pixel
        value = interface._metric_mask_area(mask, calibration)
        self.assertAlmostEqual(value, 2.0, places=4)

    def test_invalid_crossed_calibration_points_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "vùng đường lồi"):
            interface._prepare_calibration({
                "points": [[0, 0], [100, 200], [100, 0], [0, 200]],
                "road_width_m": 10,
                "road_length_m": 20,
            }, (200, 100))

    def test_locked_upload_cleanup_retries_without_failing_inference(self):
        locked = PermissionError(13, "file is in use")
        with patch.object(Path, "unlink", side_effect=[locked, None]) as unlink, patch.object(interface.time, "sleep") as sleep:
            removed = interface._unlink_with_retries(Path("temporary-video.mov"), attempts=3, delay_s=0.01)
        self.assertTrue(removed)
        self.assertEqual(unlink.call_count, 2)
        sleep.assert_called_once_with(0.01)


if __name__ == "__main__":
    unittest.main()
