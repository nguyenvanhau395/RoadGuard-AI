"""Local browser interface for testing the three approved road-damage models.

Run from the project root with::

    python scripts/test_interface.py

The server deliberately uses Python's standard library so it does not add a
second web framework to the training project. Ultralytics is imported lazily
only when a test is submitted.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import os
import re
import smtplib
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse, parse_qs
from urllib.request import Request, urlopen
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from road_damage.measurement import measure_masks

WEB_FILE = ROOT / "web" / "test_interface.html"
MEDIA_ROOT = ROOT / ".interface_runs"
LOCAL_ENV_FILE = ROOT / ".roadguard.env"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 2 * 1024 * 1024
TEST_IOU = 0.5  # Chuẩn kiểm thử ảnh/video từ archive/notebooks/CV_42k_det .ipynb
TRACK_REID_SECONDS = 3.0  # Nối lại track bị mất ngắn hạn theo FPS thực của video.
MIN_TRACK_OBSERVATIONS = 3  # Loại track chớp nhoáng trừ khi SEG đã xác nhận.
CHATBOT_SYSTEM_PROMPT = """Bạn là trợ lý phân tích báo cáo hư hỏng mặt đường RoadGuard AI.

Quy tắc bắt buộc:
- Trả lời bằng tiếng Việt, rõ ràng và ngắn gọn.
- Chỉ sử dụng số liệu có trong báo cáo được cung cấp; không tự suy đoán số liệu bị thiếu.
- Nếu có hiệu chuẩn, ưu tiên mét/m² và ghi pixel trong ngoặc để đối chiếu. Nếu chưa hiệu chuẩn, chỉ dùng pixel.
- Với video, DET và SEG chạy trên mọi frame; ByteTrack đếm mỗi track_id chuẩn một lần và báo cáo lấy mask đo tốt nhất của mỗi track.
- Chỉ coi D00/D10/D40 là hư hỏng hợp lệ khi SEG chính tạo được mask; detection không có mask bị loại khỏi kết quả, không gọi giá trị thiếu là 0.
- Khi được hỏi mức độ ưu tiên, cân nhắc loại hư hỏng, kích thước đo được, confidence và vị trí frame/track; nêu lý do dựa trên dữ liệu.
- Nếu câu hỏi nằm ngoài báo cáo, hãy nói rõ rằng báo cáo không đủ dữ liệu để kết luận.
"""

EMAIL_REPORT_PROMPT = """Hãy viết phần phân tích chuyên môn để chèn trực tiếp vào email báo cáo hư hỏng mặt đường.

Yêu cầu:
- Viết bằng tiếng Việt, giọng văn kỹ thuật, dễ hiểu, khoảng 250-450 từ.
- Có các mục rõ ràng: Đánh giá tổng quan; Phát hiện đáng chú ý; Thứ tự ưu tiên xử lý; Khuyến nghị kiểm tra/khắc phục; Giới hạn dữ liệu.
- Phân tích dựa trên loại hư hỏng, kích thước đo được, confidence, frame/track và trạng thái hiệu chuẩn có trong báo cáo.
- D00/D10 dùng chiều dài; D40 dùng diện tích; D20 chỉ được nhận xét theo số lượng, confidence và vị trí vì không có phép đo SEG tương ứng.
- Không bịa số liệu, vị trí, nguyên nhân, mức độ nguy hiểm hoặc tiêu chuẩn kỹ thuật không có trong dữ liệu.
- Không gọi dữ liệu thiếu là 0; nếu chưa hiệu chuẩn thì chỉ dùng pixel và nêu rõ chưa thể kết luận kích thước thực tế.
- Không viết tiêu đề email, lời chào, chữ ký, Markdown dạng bảng hay khối mã; chỉ trả về nội dung phân tích.
"""

DET_CLASSES = {
    "D00": "Nứt dọc",
    "D10": "Nứt ngang",
    "D20": "Nứt da cá sấu",
    "D40": "Ổ gà",
}
CRACK_CODES = {"D00", "D10"}
SEG_REQUIRED_CODES = {*CRACK_CODES, "D40"}

MODELS = {
    "det": {
        "label": "DET · YOLOv8m",
        "role": "DET",
        "task": "detect",
        "checkpoint": Path("models/runtime/roadguard_det.pt"),
    },
    "seg_crack": {
        "label": "SEG · Crack · YOLOv8s-seg",
        "role": "SEG",
        "task": "segment",
        "checkpoint": Path("models/runtime/crack_seg.pt"),
    },
    "seg_pothole": {
        "label": "SEG · Pothole · YOLOv8m-seg",
        "role": "SEG",
        "task": "segment",
        "checkpoint": Path("models/runtime/pothole_seg.pt"),
    },
}

_MODEL_CACHE: dict[str, object] = {}
_MODEL_LOCK = threading.Lock()
_TRACKING_LOCK = threading.Lock()
_ANALYSIS_LOCK = threading.Lock()
_JOB_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, object]] = {}


def _load_local_chatbot_env(path: Path = LOCAL_ENV_FILE) -> None:
    """Load server-only chatbot settings without exposing the key to the UI."""
    if not path.is_file():
        return
    allowed = {
        "CHATBOT_API_KEY", "CHATBOT_BASE_URL", "CHATBOT_MODEL",
        "REPORT_SMTP_HOST", "REPORT_SMTP_PORT", "REPORT_SMTP_USER",
        "REPORT_SMTP_PASSWORD", "REPORT_SMTP_FROM", "REPORT_SMTP_TLS",
    }
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key not in allowed or not value:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _ffmpeg_executable() -> str:
    """Return the bundled ffmpeg binary used for browser-safe H.264 video."""
    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover - project lock includes it
        raise RuntimeError("Thiếu imageio-ffmpeg để mã hóa video H.264.") from exc
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    if not Path(executable).is_file():
        raise RuntimeError("Không tìm thấy ffmpeg của imageio-ffmpeg.")
    return executable


class _BrowserVideoWriter:
    """Stream BGR frames to an H.264/yuv420p MP4 understood by web browsers."""

    def __init__(self, output: Path, fps: float, width: int, height: int):
        self.output = output
        self.width = int(width)
        self.height = int(height)
        self.closed = False
        command = [
            _ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", f"{max(float(fps), 1.0):.6f}", "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-vf", "scale=w='min(1920,iw)':h='min(1920,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame) -> None:
        import numpy as np

        if self.closed or self.process.stdin is None:
            raise RuntimeError("Bộ ghi video đã đóng.")
        array = np.asarray(frame)
        if array.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Frame video có shape {array.shape}, cần {(self.height, self.width, 3)}."
            )
        try:
            self.process.stdin.write(np.ascontiguousarray(array).tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError("ffmpeg dừng khi đang mã hóa video.") from exc

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process.stdin is not None:
            self.process.stdin.close()
            self.process.stdin = None
        _, stderr = self.process.communicate()
        if self.process.returncode != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Không mã hóa được video H.264: {message or self.process.returncode}")
        if not self.output.is_file() or self.output.stat().st_size == 0:
            raise RuntimeError("ffmpeg không tạo được video kết quả.")


def _transcode_browser_video(source: Path, destination: Path) -> Path:
    """Convert an Ultralytics/OpenCV video to fast-start H.264 MP4."""
    command = [
        _ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-map", "0:v:0", "-an", "-c:v", "libx264",
        "-preset", "fast", "-crf", "18",
        "-vf", "scale=w='min(1920,iw)':h='min(1920,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Không chuyển được video sang H.264: {message or completed.returncode}")
    return destination


def _parse_byte_range(value: str | None, size: int) -> tuple[int, int] | None:
    """Parse one RFC 7233 byte range; return inclusive start/end."""
    if not value:
        return None
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
    if not match or size <= 0:
        raise ValueError("Range không hợp lệ.")
    first, last = match.groups()
    if not first and not last:
        raise ValueError("Range không hợp lệ.")
    if first:
        start = int(first)
        end = int(last) if last else size - 1
    else:
        suffix = int(last)
        if suffix <= 0:
            raise ValueError("Range không hợp lệ.")
        start, end = max(0, size - suffix), size - 1
    if start >= size or end < start:
        raise ValueError("Range vượt quá kích thước file.")
    return start, min(end, size - 1)


def _safe_name(value: str, fallback: str = "upload") -> str:
    value = Path(value or fallback).name
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value or fallback


def _unlink_with_retries(path: Path, attempts: int = 6, delay_s: float = 0.2) -> bool:
    """Remove an uploaded file without turning a successful run into WinError 32.

    OpenCV/Ultralytics and antivirus scanners can retain a Windows file handle
    briefly after video decoding has finished. Cleanup is best-effort and must
    not replace a valid inference response with a server error.
    """
    for attempt in range(max(1, attempts)):
        try:
            path.unlink(missing_ok=True)
            return True
        except OSError as exc:
            if attempt + 1 >= attempts:
                return False
            # WinError 32/5 and POSIX EACCES are commonly transient locks.
            if getattr(exc, "winerror", None) not in {5, 32} and getattr(exc, "errno", None) != 13:
                return False
            time.sleep(delay_s)
    return False


def _cleanup_uploaded_file(path: Path) -> None:
    """Try now, then continue cleanup in a daemon thread if Windows is busy."""
    if _unlink_with_retries(path):
        return

    def deferred_cleanup() -> None:
        if not _unlink_with_retries(path, attempts=60, delay_s=1.0):
            print(f"[cleanup] Chưa thể xóa file upload đang bị khóa: {path}")

    threading.Thread(
        target=deferred_cleanup,
        name=f"upload-cleanup-{path.stem[:12]}",
        daemon=True,
    ).start()


def _model_payload() -> list[dict[str, object]]:
    payload = []
    for key, item in MODELS.items():
        checkpoint = ROOT / item["checkpoint"]
        payload.append(
            {
                "id": key,
                "label": item["label"],
                "role": item["role"],
                "task": item["task"],
                "checkpoint": item["checkpoint"].as_posix(),
                "exists": checkpoint.is_file(),
            }
        )
    return payload


def _load_model(model_id: str):
    if model_id not in MODELS:
        raise ValueError("Model không hợp lệ.")
    if model_id not in _MODEL_CACHE:
        checkpoint = ROOT / MODELS[model_id]["checkpoint"]
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on user's env
            raise RuntimeError(
                "Môi trường hiện tại chưa có ultralytics. "
                "Hãy chạy scripts/run_test_interface.ps1 hoặc dùng .conda\\python.exe."
            ) from exc
        with _MODEL_LOCK:
            if model_id not in _MODEL_CACHE:
                _MODEL_CACHE[model_id] = YOLO(str(checkpoint), task=MODELS[model_id]["task"])
    return _MODEL_CACHE[model_id]


def _parse_multipart(content_type: str, body: bytes) -> dict[str, tuple[str | None, bytes]]:
    """Parse one multipart request without relying on the removed cgi module."""
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("utf-8") + b"\r\n"
        b"MIME-Version: 1.0\r\n\r\n" + body
    )
    fields: dict[str, tuple[str | None, bytes]] = {}
    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        name = part.get_param("name", header="content-disposition")
        if not name or "form-data" not in disposition:
            continue
        fields[name] = (part.get_filename(), part.get_payload(decode=True) or b"")
    return fields


def _is_video(filename: str, content_type: str = "") -> bool:
    return content_type.startswith("video/") or Path(filename).suffix.lower() in {
        ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"
    }


def _count_instances(result) -> int:
    boxes = getattr(result, "boxes", None)
    try:
        return int(len(boxes)) if boxes is not None else 0
    except TypeError:
        return 0


def _to_numpy(value):
    """Convert an Ultralytics tensor to a NumPy array without assuming CUDA."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return value


def _detection_rows(result, frame_index: int = 0, timestamp_s: float = 0.0) -> list[dict[str, object]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = _to_numpy(getattr(boxes, "xyxy", []))
    classes = _to_numpy(getattr(boxes, "cls", []))
    confidences = _to_numpy(getattr(boxes, "conf", []))
    raw_track_ids = getattr(boxes, "id", None)
    track_ids = _to_numpy(raw_track_ids) if raw_track_ids is not None else []
    names = getattr(result, "names", {}) or {}
    rows: list[dict[str, object]] = []
    for index, box in enumerate(xyxy):
        class_id = int(classes[index]) if len(classes) > index else -1
        raw_name = str(names.get(class_id, class_id)).upper()
        code = raw_name if raw_name in DET_CLASSES else raw_name.split(" ")[0]
        label = DET_CLASSES.get(code, str(names.get(class_id, raw_name)))
        x1, y1, x2, y2 = [round(float(value), 2) for value in box[:4]]
        rows.append(
            {
                "frame": frame_index,
                "timestamp_s": round(float(timestamp_s), 3),
                "class_code": code,
                "class_name": label,
                "confidence": round(float(confidences[index]), 4) if len(confidences) > index else None,
                "track_id": int(track_ids[index]) if len(track_ids) > index else None,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": round((x1 + x2) / 2, 2),
                "center_y": round((y1 + y2) / 2, 2),
                "measurement_type": None,
                "measurement_value_px": None,
            }
        )
    return rows


def _seg_mask(result, shape):
    """Return the union of all SEG masks in original crop/image resolution."""
    stack = _seg_masks(result, shape)
    if len(stack) == 0:
        import numpy as np

        return np.zeros(shape[:2], dtype=bool)
    return stack.any(axis=0)


def _seg_masks(result, shape):
    """Return an N,H,W boolean mask stack aligned to the input image.

    ``retina_masks=True`` normally makes Ultralytics return masks at the
    original resolution.  The resize fallback also keeps this helper correct
    for older exports and mocked results used by tests.
    """
    import cv2
    import numpy as np

    masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "data", None) is None:
        return np.empty((0, *shape[:2]), dtype=bool)
    data = np.asarray(_to_numpy(masks.data))
    if data.size == 0:
        return np.empty((0, *shape[:2]), dtype=bool)
    if data.ndim == 2:
        data = data[None, ...]
    resized = []
    for mask in data:
        binary = (mask > 0.5).astype("uint8")
        if binary.shape != tuple(shape[:2]):
            binary = cv2.resize(binary, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        resized.append(binary.astype(bool))
    return np.stack(resized, axis=0)


def _measure_segment(result, shape, measurement_type: str) -> tuple[float, object]:
    """Return pixel measurement and the binary mask for one SEG result."""
    masks = _seg_masks(result, shape)
    task = "pothole" if measurement_type == "area_px2" else "crack"
    measured = measure_masks(masks, task=task, original_shape=tuple(shape[:2]))
    mask = masks.any(axis=0) if len(masks) else _seg_mask(result, shape)
    value_key = "total_area_px2" if task == "pothole" else "total_length_px"
    return float(measured[value_key]), mask


def _prepare_calibration(spec: dict[str, object] | None, image_shape) -> dict[str, object] | None:
    """Validate four road corners and build the image-to-road homography.

    Point order is near-left, near-right, far-right, far-left. The destination
    plane uses metres: road width on X and road-segment length on Y.
    """
    if not spec:
        return None
    import cv2
    import numpy as np

    try:
        road_width_m = float(spec["road_width_m"])
        road_length_m = float(spec["road_length_m"])
        points = np.asarray(spec["points"], dtype="float32")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Thông tin hiệu chuẩn không hợp lệ.") from exc
    if not np.isfinite([road_width_m, road_length_m]).all() or road_width_m <= 0 or road_length_m <= 0:
        raise ValueError("Chiều rộng và chiều dài đoạn đường phải lớn hơn 0.")
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("Hiệu chuẩn cần đúng 4 điểm ảnh hợp lệ.")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("Kích thước ảnh/video không hợp lệ để hiệu chuẩn.")
    if (points[:, 0] < 0).any() or (points[:, 0] > width).any() or (points[:, 1] < 0).any() or (points[:, 1] > height).any():
        raise ValueError("Điểm hiệu chuẩn nằm ngoài khung ảnh/video.")
    contour = points.reshape(-1, 1, 2)
    if abs(float(cv2.contourArea(contour))) < 25 or not cv2.isContourConvex(contour):
        raise ValueError("Bốn điểm hiệu chuẩn phải tạo thành một vùng đường lồi, không giao nhau.")
    destination = np.asarray(
        [[0.0, 0.0], [road_width_m, 0.0], [road_width_m, road_length_m], [0.0, road_length_m]],
        dtype="float32",
    )
    homography = cv2.getPerspectiveTransform(points, destination)
    if not np.isfinite(homography).all():
        raise ValueError("Không tính được ma trận quy đổi từ các điểm hiệu chuẩn.")
    return {
        "enabled": True,
        "points": points.tolist(),
        "road_width_m": road_width_m,
        "road_length_m": road_length_m,
        "homography": homography,
        "method": "four_point_homography",
    }


def _calibration_payload(calibration: dict[str, object] | None) -> dict[str, object] | None:
    """Return the serializable calibration metadata (without its matrix)."""
    if not calibration:
        return None
    return {
        "enabled": True,
        "points": calibration["points"],
        "road_width_m": calibration["road_width_m"],
        "road_length_m": calibration["road_length_m"],
        "method": calibration["method"],
        "point_order": ["near_left", "near_right", "far_right", "far_left"],
    }


def _metric_crack_length(mask, calibration: dict[str, object], offset_x: float = 0, offset_y: float = 0) -> float:
    """Measure skeleton graph edges after projecting endpoints onto the road plane."""
    import cv2
    import numpy as np
    from skimage.morphology import skeletonize

    pixels = skeletonize(np.asarray(mask, dtype=bool))
    edge_sets = []
    horizontal = pixels[:, :-1] & pixels[:, 1:]
    vertical = pixels[:-1, :] & pixels[1:, :]
    down_right = pixels[:-1, :-1] & pixels[1:, 1:] & ~pixels[:-1, 1:] & ~pixels[1:, :-1]
    down_left = pixels[:-1, 1:] & pixels[1:, :-1] & ~pixels[:-1, :-1] & ~pixels[1:, 1:]
    for matches, delta in ((horizontal, (1, 0)), (vertical, (0, 1)), (down_right, (1, 1)), (down_left, (-1, 1))):
        yx = np.argwhere(matches)
        if not len(yx):
            continue
        starts = np.column_stack((yx[:, 1] + offset_x, yx[:, 0] + offset_y))
        # Down-left matches are indexed from the top-right sub-array.
        if delta[0] == -1:
            starts[:, 0] += 1
        ends = starts + np.asarray(delta, dtype="float64")
        edge_sets.append(np.stack((starts, ends), axis=1))
    if not edge_sets:
        return 0.0
    edges = np.concatenate(edge_sets, axis=0).astype("float32")
    projected = cv2.perspectiveTransform(edges.reshape(-1, 1, 2), calibration["homography"]).reshape(-1, 2, 2)
    return float(np.linalg.norm(projected[:, 1] - projected[:, 0], axis=1).sum())


def _metric_mask_area(mask, calibration: dict[str, object], offset_x: float = 0, offset_y: float = 0) -> float:
    """Sum projected pixel-cell areas, preserving perspective and small masks."""
    import cv2
    import numpy as np

    yx = np.argwhere(np.asarray(mask, dtype=bool))
    if not len(yx):
        return 0.0
    total = 0.0
    # Chunking bounds memory for large pothole masks and full-resolution SEG.
    for start in range(0, len(yx), 50000):
        part = yx[start:start + 50000]
        x = part[:, 1].astype("float32") + float(offset_x)
        y = part[:, 0].astype("float32") + float(offset_y)
        cells = np.stack(
            (
                np.column_stack((x - .5, y - .5)),
                np.column_stack((x + .5, y - .5)),
                np.column_stack((x + .5, y + .5)),
                np.column_stack((x - .5, y + .5)),
            ),
            axis=1,
        ).astype("float32")
        world = cv2.perspectiveTransform(cells.reshape(-1, 1, 2), calibration["homography"]).reshape(-1, 4, 2)
        px, py = world[:, :, 0], world[:, :, 1]
        total += float((.5 * np.abs(np.sum(px * np.roll(py, -1, axis=1) - py * np.roll(px, -1, axis=1), axis=1))).sum())
    return total


def _metric_measurement(mask, measurement_type: str, calibration: dict[str, object] | None, offset_x: float = 0, offset_y: float = 0) -> tuple[float | None, str | None]:
    if not calibration or not getattr(mask, "any", lambda: False)():
        return None, None
    if measurement_type == "length_px":
        return _metric_crack_length(mask, calibration, offset_x, offset_y), "m"
    return _metric_mask_area(mask, calibration, offset_x, offset_y), "m2"


def _segment_rows(
    result,
    model_id: str,
    frame_index: int = 0,
    timestamp_s: float = 0.0,
    calibration: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Build exportable rows and measurements for direct SEG inference."""
    shape = tuple(getattr(result, "orig_shape", ()) or ())
    if len(shape) != 2:
        original = getattr(result, "orig_img", None)
        if original is None:
            return []
        shape = tuple(original.shape[:2])
    rows = _detection_rows(result, frame_index, timestamp_s)
    masks = _seg_masks(result, shape)
    measurement_type = "length_px" if model_id == "seg_crack" else "area_px2"
    task = "crack" if model_id == "seg_crack" else "pothole"
    class_code = "SEG-CRACK" if task == "crack" else "D40"
    class_name = "Vết nứt (SEG, chưa phân loại DET)" if task == "crack" else "Ổ gà"

    if not rows and len(masks):
        # Defensive fallback for an export that contains masks without boxes.
        for _ in range(len(masks)):
            rows.append({
                "frame": frame_index, "timestamp_s": round(float(timestamp_s), 3),
                "x1": 0.0, "y1": 0.0, "x2": float(shape[1]), "y2": float(shape[0]),
                "center_x": shape[1] / 2, "center_y": shape[0] / 2,
                "confidence": None,
            })

    for index, row in enumerate(rows):
        row["class_code"] = class_code
        row["class_name"] = class_name
        row["measurement_type"] = measurement_type
        row["segment_model"] = MODELS[model_id]["label"]
        if index < len(masks) and masks[index].any():
            measured = measure_masks(masks[index], task=task, original_shape=shape)
            key = "total_length_px" if task == "crack" else "total_area_px2"
            row["measurement_value_px"] = round(float(measured[key]), 2)
            metric_value, metric_unit = _metric_measurement(masks[index], measurement_type, calibration)
            row["measurement_value_metric"] = None if metric_value is None else round(metric_value, 4)
            row["measurement_unit_metric"] = metric_unit
            row["segment_mask_pixels"] = int(masks[index].sum())
            row["measurement_status"] = "measured"
        else:
            row["measurement_value_px"] = None
            row["measurement_value_metric"] = None
            row["measurement_unit_metric"] = None
            row["segment_mask_pixels"] = 0
            row["measurement_status"] = "no_mask"
    return rows


def _pipeline_row_is_accepted(row: dict[str, object]) -> bool:
    """Keep DET-only classes, but require a primary SEG mask where applicable."""
    code = str(row.get("class_code") or "")
    return code not in SEG_REQUIRED_CODES or row.get("measurement_status") == "measured"


def _track_is_confirmed(track_rows: list[dict[str, object]]) -> bool:
    observations = max(
        [len(track_rows), *(int(row.get("track_observations") or 0) for row in track_rows)]
    )
    return observations >= MIN_TRACK_OBSERVATIONS or any(
        row.get("measurement_status") == "measured" for row in track_rows
    )


def _deduplicate_tracked_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], bool]:
    """Select one representative measurement per ByteTrack ID.

    The notebook counts an object only the first time its track ID appears.
    For SEG reporting we keep the strongest measured mask for that same ID so
    a weak/empty first-frame crop does not discard a later valid measurement.
    """
    tracked: dict[tuple[str, int], list[dict[str, object]]] = {}
    untracked: list[dict[str, object]] = []

    def score(row: dict[str, object]) -> tuple[int, float, float]:
        status_rank = {"measured": 2, "not_applicable": 1}.get(
            str(row.get("measurement_status")), 0
        )
        measurement = row.get("measurement_value_metric")
        if measurement is None:
            measurement = row.get("measurement_value_px")
        return status_rank, float(measurement or 0), float(row.get("confidence") or 0)

    for row in rows:
        track_id = row.get("track_id")
        if track_id is None:
            untracked.append(row)
            continue
        key = (str(row.get("class_code") or ""), int(track_id))
        tracked.setdefault(key, []).append(row)
    if not tracked:
        return rows, False
    representatives: list[dict[str, object]] = []
    for track_rows in tracked.values():
        if not _track_is_confirmed(track_rows):
            continue
        representative = dict(max(track_rows, key=score))
        representative["track_observations"] = max(
            [len(track_rows), *(int(row.get("track_observations") or 0) for row in track_rows)]
        )
        representative["track_confirmed"] = True
        representatives.append(representative)
    return [*representatives, *untracked], True


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    summary_rows, tracking_deduplicated = _deduplicate_tracked_rows(rows)
    by_class: dict[str, int] = {}
    measurements: list[dict[str, object]] = []
    segmentation = {"measured": 0, "no_mask": 0, "not_applicable": 0}
    for row in summary_rows:
        label = str(row.get("class_name") or "Không xác định")
        by_class[label] = by_class.get(label, 0) + 1
        status = str(row.get("measurement_status") or "not_applicable")
        segmentation[status] = segmentation.get(status, 0) + 1
        value = row.get("measurement_value_px")
        if value is not None:
            measurements.append(
                {
                    "class_code": row.get("class_code"),
                    "class_name": label,
                    "measurement_type": row.get("measurement_type"),
                    "value_px": value,
                }
            )
    total_length = sum(
        float(item["value_px"])
        for item in measurements
        if item["measurement_type"] == "length_px"
    )
    total_area = sum(
        float(item["value_px"])
        for item in measurements
        if item["measurement_type"] == "area_px2"
    )
    longitudinal = sum(
        float(item["value_px"])
        for item in measurements
        if item["measurement_type"] == "length_px" and item.get("class_code") == "D00"
    )
    transverse = sum(
        float(item["value_px"])
        for item in measurements
        if item["measurement_type"] == "length_px" and item.get("class_code") == "D10"
    )
    metric_rows = [row for row in summary_rows if row.get("measurement_value_metric") is not None]
    total_length_m = sum(
        float(row["measurement_value_metric"])
        for row in metric_rows if row.get("measurement_unit_metric") == "m"
    )
    total_area_m2 = sum(
        float(row["measurement_value_metric"])
        for row in metric_rows if row.get("measurement_unit_metric") == "m2"
    )
    longitudinal_m = sum(
        float(row["measurement_value_metric"])
        for row in metric_rows if row.get("measurement_unit_metric") == "m" and row.get("class_code") == "D00"
    )
    transverse_m = sum(
        float(row["measurement_value_metric"])
        for row in metric_rows if row.get("measurement_unit_metric") == "m" and row.get("class_code") == "D10"
    )
    return {
        "instances": len(summary_rows),
        "raw_frame_detections": len(rows),
        "tracking_deduplicated": tracking_deduplicated,
        "unique_track_count": sum(1 for row in summary_rows if row.get("track_id") is not None),
        "by_class": by_class,
        "total_crack_length_px": round(total_length, 2),
        "longitudinal_crack_length_px": round(longitudinal, 2),
        "transverse_crack_length_px": round(transverse, 2),
        "total_pothole_area_px2": round(total_area, 2),
        "calibrated_measurements": len(metric_rows),
        "total_crack_length_m": round(total_length_m, 4),
        "longitudinal_crack_length_m": round(longitudinal_m, 4),
        "transverse_crack_length_m": round(transverse_m, 4),
        "total_pothole_area_m2": round(total_area_m2, 4),
        "segmentation": segmentation,
    }


def _density(
    rows: list[dict[str, object]],
    width: int,
    height: int,
    total_frames: int | None = None,
) -> list[dict[str, object]]:
    """Build one stable road-map marker per canonical track.

    Video inference produces one row per visible frame.  Rendering those rows
    directly made a single physical defect look like many defects on the road
    map.  Collapse tracked rows here, keep the first-seen time for playback,
    and use the median lane position so a marker does not drift frame-to-frame.
    """
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    density_rows: list[dict[str, object]] = []
    for row in rows:
        track_id = row.get("track_id")
        if track_id is None:
            density_rows.append(dict(row))
            continue
        key = (str(row.get("class_code") or ""), int(track_id))
        grouped.setdefault(key, []).append(row)

    for (_, track_id), track_rows in grouped.items():
        if not _track_is_confirmed(track_rows):
            continue
        first = min(track_rows, key=lambda item: int(item.get("frame") or 0))
        last = max(track_rows, key=lambda item: int(item.get("frame") or 0))
        strongest = max(track_rows, key=lambda item: float(item.get("confidence") or 0))
        center_values = sorted(float(item.get("center_x") or 0) for item in track_rows)
        middle = len(center_values) // 2
        median_center = (
            center_values[middle]
            if len(center_values) % 2
            else (center_values[middle - 1] + center_values[middle]) / 2
        )
        representative = dict(strongest)
        representative.update({
            "track_id": track_id,
            "frame": int(first.get("frame") or 0),
            "timestamp_s": float(first.get("timestamp_s") or 0),
            "last_timestamp_s": float(last.get("timestamp_s") or 0),
            "center_x": median_center,
            "observations": len(track_rows),
        })
        density_rows.append(representative)

    detected_max_frame = max((int(row.get("frame") or 0) for row in rows), default=0)
    max_frame = max(detected_max_frame, max(0, int(total_frames or 0) - 1))
    return [
        {
            # Video uses time/progress as the road axis and horizontal image
            # position as the lane axis. Images retain their 2D coordinates.
            "x": round(
                int(row.get("frame") or 0) / max_frame
                if max_frame else float(row["center_x"]) / max(width, 1), 4
            ),
            "y": round(
                float(row["center_x"]) / max(width, 1)
                if max_frame else float(row["center_y"]) / max(height, 1), 4
            ),
            "z": round(float(row.get("confidence") or 0), 4),
            "frame": int(row.get("frame") or 0),
            "timestamp_s": float(row.get("timestamp_s") or 0),
            "last_timestamp_s": float(row.get("last_timestamp_s") or row.get("timestamp_s") or 0),
            "track_id": row.get("track_id"),
            "observations": int(row.get("observations") or 1),
            "class_name": row["class_name"],
            "class_code": row["class_code"],
            "confidence": row["confidence"],
        }
        for row in density_rows
    ]


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _save_report(run_id: str, report: dict[str, object]) -> None:
    (MEDIA_ROOT / run_id / "report.json").write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _read_report(run_id: str, root: Path | None = None) -> dict[str, object]:
    report_path = (root or MEDIA_ROOT) / _safe_name(run_id) / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError("Không tìm thấy báo cáo của lượt test.")
    return json.loads(report_path.read_text(encoding="utf-8"))


def _validate_upload_size(size: int) -> int:
    size = int(size)
    if size <= 0 or size > MAX_UPLOAD_BYTES:
        raise ValueError("File phải lớn hơn 0 và không vượt quá 200 MB.")
    return size


def _result_media(run_id: str, root: Path | None = None) -> Path | None:
    base = root or MEDIA_ROOT
    run_dir = base / _safe_name(run_id)
    if not run_dir.is_dir():
        return None
    extensions = {".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi", ".mkv", ".webm"}
    candidates = sorted(
        (
            path for path in run_dir.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        ),
        key=lambda path: (not path.name.startswith("result"), path.name),
    )
    return candidates[0] if candidates else None


def _history_result(run_id: str, root: Path | None = None) -> dict[str, object]:
    base = root or MEDIA_ROOT
    report = _read_report(run_id, base)
    media = _result_media(run_id, base)
    if media is None:
        raise FileNotFoundError("Không tìm thấy media kết quả của lượt test.")
    mode = str(report.get("mode") or "single_model")
    checkpoint = str(report.get("model") or "")
    label = "Model đơn"
    if mode == "pipeline":
        label = "Pipeline DET → SEG"
        checkpoint = "3 checkpoint chính · chỉ giữ mask SEG chính"
    else:
        for item in MODELS.values():
            if item["checkpoint"].as_posix() == checkpoint:
                label = str(item["label"])
                break
    input_data = report.get("input") or {}
    result = {
        "run_id": _safe_name(run_id),
        "created_at": report.get("created_at"),
        "kind": input_data.get("kind", "image"),
        "url": f"/media/{_safe_name(run_id)}/{media.name}",
        "instances": (report.get("summary") or {}).get("instances", 0),
        "elapsed_ms": report.get("elapsed_ms", 0),
        "label": label,
        "checkpoint": checkpoint,
        "detections": report.get("detections", []),
        "detections_unique": report.get("detections_unique", []),
        "summary": report.get("summary", {}),
        "density": report.get("density", []),
        "calibration": report.get("calibration"),
        "seg_filter_policy": report.get("seg_filter_policy"),
        "frames": report.get("frames"),
        "detection_every": report.get("detection_every"),
        "merged_track_ids": report.get("merged_track_ids"),
        "report_url": f"/api/report/{_safe_name(run_id)}",
        "csv_url": f"/api/export/{_safe_name(run_id)}?format=csv",
        "xlsx_url": f"/api/export/{_safe_name(run_id)}?format=xlsx",
    }
    return result


def _history_payload(root: Path | None = None, limit: int = 50) -> list[dict[str, object]]:
    base = root or MEDIA_ROOT
    if not base.is_dir():
        return []
    items: list[dict[str, object]] = []
    for run_dir in base.iterdir():
        report_path = run_dir / "report.json"
        if not run_dir.is_dir() or not report_path.is_file():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        input_data = report.get("input") or {}
        summary = report.get("summary") or {}
        media = _result_media(run_dir.name, base)
        items.append({
            "run_id": run_dir.name,
            "created_at": report.get("created_at"),
            "input_name": input_data.get("name", "Không rõ tên file"),
            "kind": input_data.get("kind", "image"),
            "mode": report.get("mode", "single_model"),
            "instances": summary.get("instances", 0),
            "elapsed_ms": report.get("elapsed_ms", 0),
            "media_available": media is not None,
        })
    items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return items[:max(1, min(int(limit), 200))]


def _report_rows(report: dict[str, object]) -> list[list[object]]:
    headers = [
        "Frame", "Timestamp (s)", "Track ID chuẩn", "Track ID gốc", "Đã gộp track",
        "Số lần quan sát", "Đã xác nhận",
        "Mã lớp", "Loại hư hỏng", "Confidence",
        "X1", "Y1", "X2", "Y2", "Loại đo", "Giá trị pixel", "Giá trị quy đổi",
        "Đơn vị quy đổi", "Trạng thái SEG", "Mô hình SEG",
        "Chiến lược SEG", "Confidence mask",
    ]
    rows: list[list[object]] = [headers]
    detail_rows = report.get("detections_unique") or report.get("detections", [])
    for item in detail_rows:
        rows.append([
            item.get("frame"), item.get("timestamp_s"), item.get("track_id"),
            item.get("raw_track_id"), item.get("track_merged"), item.get("track_observations"),
            item.get("track_confirmed"), item.get("class_code"),
            item.get("class_name"), item.get("confidence"), item.get("x1"),
            item.get("y1"), item.get("x2"), item.get("y2"),
            item.get("measurement_type"), item.get("measurement_value_px"),
            item.get("measurement_value_metric"), item.get("measurement_unit_metric"),
            item.get("measurement_status"), item.get("segment_model"),
            item.get("segment_strategy"), item.get("segment_prediction_confidence"),
        ])
    summary = report.get("summary", {})
    rows.extend([
        [], ["Tổng instances", summary.get("instances")],
        ["Tổng detection theo frame", summary.get("raw_frame_detections")],
        ["Track ID duy nhất (ByteTrack)", summary.get("unique_track_count")],
        ["Tổng chiều dài nứt (px)", summary.get("total_crack_length_px")],
        ["Nứt dọc (px)", summary.get("longitudinal_crack_length_px")],
        ["Nứt ngang (px)", summary.get("transverse_crack_length_px")],
        ["Tổng diện tích ổ gà (px²)", summary.get("total_pothole_area_px2")],
        ["Tổng chiều dài nứt (m)", summary.get("total_crack_length_m")],
        ["Nứt dọc (m)", summary.get("longitudinal_crack_length_m")],
        ["Nứt ngang (m)", summary.get("transverse_crack_length_m")],
        ["Tổng diện tích ổ gà (m²)", summary.get("total_pothole_area_m2")],
        ["SEG đo được", summary.get("segmentation", {}).get("measured")],
        ["SEG không có mask", summary.get("segmentation", {}).get("no_mask")],
    ])
    calibration = report.get("calibration") or {}
    if calibration.get("enabled"):
        rows.extend([
            ["Chiều rộng đoạn đường hiệu chuẩn (m)", calibration.get("road_width_m")],
            ["Chiều dài đoạn đường hiệu chuẩn (m)", calibration.get("road_length_m")],
            ["Phương pháp quy đổi", "Homography 4 điểm"],
        ])
    return rows


def _csv_bytes(report: dict[str, object]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerows(_report_rows(report))
    return output.getvalue().encode("utf-8-sig")


def _xlsx_bytes(report: dict[str, object]) -> bytes:
    """Create a small dependency-free XLSX report with one readable sheet."""
    rows = _report_rows(report)
    sheet_rows = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for col_index, value in enumerate(row):
            if value is None:
                continue
            ref = f"{chr(65 + min(col_index, 25))}{row_index}"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                text = escape(str(value))
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        sheet_rows.append(f'<row r="{row_index}">' + "".join(cells) + "</row>")
    sheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>' + "".join(sheet_rows) + '</sheetData></worksheet>'
    ).encode("utf-8")
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Bao cao" sheetId="1" r:id="rId1"/></sheets></workbook>'
    ).encode("utf-8")
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '</Relationships>'
    ).encode("utf-8")
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    ).encode("utf-8")
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>'
    ).encode("utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", rels)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)
    return buffer.getvalue()


def _format_vi_number(value: object, digits: int = 2) -> str:
    """Format a report number with Vietnamese decimal/group separators."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    formatted = f"{number:,.{digits}f}"
    integer, _, decimal = formatted.partition(".")
    integer = integer.replace(",", ".")
    decimal = decimal.rstrip("0")
    return f"{integer},{decimal}" if decimal else integer


def _local_email_narrative(report: dict[str, object]) -> str:
    """Keep email delivery useful when the configured LLM is unavailable."""
    summary = report.get("summary", {})
    by_class = summary.get("by_class") or {}
    count_text = ", ".join(
        f"{name}: {count}" for name, count in by_class.items()
    ) or "không ghi nhận hư hỏng hợp lệ"
    calibrated = bool((report.get("calibration") or {}).get("enabled"))
    if calibrated:
        measured = (
            f"Tổng chiều dài nứt là {_format_vi_number(summary.get('total_crack_length_m'))} m "
            f"({_format_vi_number(summary.get('total_crack_length_px'))} px); diện tích ổ gà là "
            f"{_format_vi_number(summary.get('total_pothole_area_m2'))} m² "
            f"({_format_vi_number(summary.get('total_pothole_area_px2'))} px²)."
        )
    else:
        measured = (
            f"Tổng chiều dài nứt là {_format_vi_number(summary.get('total_crack_length_px'))} px; "
            f"diện tích ổ gà là {_format_vi_number(summary.get('total_pothole_area_px2'))} px². "
            "Dữ liệu chưa được hiệu chuẩn nên chưa thể kết luận kích thước thực tế theo mét/m²."
        )
    priorities: list[str] = []
    if by_class.get("Ổ gà", 0):
        priorities.append("ưu tiên kiểm tra các ổ gà và xác nhận phạm vi cần vá tại hiện trường")
    if by_class.get("Nứt da cá sấu", 0):
        priorities.append("khảo sát các vùng nứt da cá sấu để đánh giá mức suy giảm kết cấu")
    if by_class.get("Nứt dọc", 0) or by_class.get("Nứt ngang", 0):
        priorities.append("theo dõi và xử lý các đường nứt đã được SEG xác nhận")
    recommendation = "; ".join(priorities) or "tiếp tục theo dõi đoạn đường theo kế hoạch"
    return (
        "Đánh giá tổng quan\n"
        f"Hệ thống ghi nhận {summary.get('instances', 0)} hư hỏng hợp lệ ({count_text}). {measured}\n\n"
        "Thứ tự ưu tiên và khuyến nghị\n"
        f"Đề xuất {recommendation}. Cần đối chiếu trực tiếp tại hiện trường trước khi quyết định phương án sửa chữa.\n\n"
        "Giới hạn dữ liệu\n"
        "Nội dung này là bản dự phòng tự động từ số liệu DET/SEG vì dịch vụ LLM chưa trả được phần phân tích."
    )


def _generate_email_narrative(report: dict[str, object]) -> str:
    """Ask the configured LLM to author the report analysis for the email."""
    try:
        narrative = _chatbot(report, EMAIL_REPORT_PROMPT).strip()
        if not narrative:
            raise ValueError("LLM trả nội dung trống")
        if "Tóm tắt cục bộ;" in narrative or "Đang dùng báo cáo cục bộ" in narrative:
            return _local_email_narrative(report)
        return narrative
    except Exception:
        # Email delivery must not fail only because the analysis provider is
        # temporarily unavailable. SMTP errors are still surfaced normally.
        return _local_email_narrative(report)


def _clean_email_header(value: object, field_name: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned or "\r" in cleaned or "\n" in cleaned:
        raise ValueError(f"{field_name} không hợp lệ.")
    return cleaned


def _build_report_email(
    report: dict[str, object],
    recipient: str,
    sender: str,
    narrative: str,
) -> EmailMessage:
    """Build a Gmail-friendly multipart report with an escaped LLM analysis."""
    sender = _clean_email_header(sender, "Email người gửi")
    recipient = _clean_email_header(recipient, "Email người nhận")
    input_data = report.get("input", {})
    input_name = str(input_data.get("name") or "input").replace("\r", " ").replace("\n", " ")
    input_kind = "Video" if input_data.get("kind") == "video" else "Ảnh"
    summary = report.get("summary", {})
    segmentation = summary.get("segmentation") or {}
    calibration = report.get("calibration") or {}
    calibrated = bool(calibration.get("enabled"))
    by_class = summary.get("by_class") or {}
    narrative = str(narrative or "").strip() or _local_email_narrative(report)

    crack_px = _format_vi_number(summary.get("total_crack_length_px"))
    pothole_px = _format_vi_number(summary.get("total_pothole_area_px2"))
    crack_metric = _format_vi_number(summary.get("total_crack_length_m"))
    pothole_metric = _format_vi_number(summary.get("total_pothole_area_m2"))
    crack_display = f"{crack_metric} m ({crack_px} px)" if calibrated else f"{crack_px} px"
    pothole_display = f"{pothole_metric} m² ({pothole_px} px²)" if calibrated else f"{pothole_px} px²"
    class_lines = [f"- {name}: {count}" for name, count in by_class.items()]
    if not class_lines:
        class_lines = ["- Không ghi nhận hư hỏng hợp lệ"]
    calibration_text = (
        f"Đã hiệu chuẩn {_format_vi_number(calibration.get('road_width_m'))} × "
        f"{_format_vi_number(calibration.get('road_length_m'))} m"
        if calibrated else "Chưa hiệu chuẩn; giữ đơn vị pixel"
    )

    plain = (
        "ROADGUARD AI – BÁO CÁO PHÂN TÍCH HƯ HỎNG MẶT ĐƯỜNG\n\n"
        f"Input: {input_name}\n"
        f"Loại dữ liệu: {input_kind}\n"
        f"Hiệu chuẩn: {calibration_text}\n\n"
        "KẾT QUẢ TỔNG HỢP\n"
        f"- Tổng hư hỏng hợp lệ: {summary.get('instances', 0)}\n"
        + "\n".join(class_lines) + "\n"
        f"- Tổng chiều dài nứt: {crack_display}\n"
        f"- Tổng diện tích ổ gà: {pothole_display}\n"
        f"- SEG đo được/không có mask: {segmentation.get('measured', 0)}/"
        f"{segmentation.get('no_mask', 0)}\n\n"
        "PHÂN TÍCH VÀ KHUYẾN NGHỊ DO LLM SOẠN\n"
        f"{narrative}\n\n"
        "DỮ LIỆU ĐÍNH KÈM\n"
        "Báo cáo chi tiết được đính kèm ở định dạng CSV và Excel. "
        "Các kết luận sửa chữa cần được xác nhận bằng khảo sát hiện trường.\n"
    )

    def html_text(value: object) -> str:
        return escape(str(value), {'"': "&quot;", "'": "&#x27;"})

    class_rows = "".join(
        "<tr>"
        f"<td style=\"padding:8px 10px;border-bottom:1px solid #e5e7eb\">{html_text(name)}</td>"
        f"<td style=\"padding:8px 10px;border-bottom:1px solid #e5e7eb;text-align:right;font-weight:700\">{html_text(count)}</td>"
        "</tr>"
        for name, count in by_class.items()
    ) or (
        "<tr><td colspan=\"2\" style=\"padding:8px 10px;color:#64748b\">"
        "Không ghi nhận hư hỏng hợp lệ</td></tr>"
    )
    narrative_html = html_text(narrative).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
    html = f"""<!doctype html>
<html lang="vi"><body style="margin:0;padding:0;background:#f1f5f4;color:#172033;font-family:Arial,sans-serif">
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:#f1f5f4;padding:24px 10px"><tr><td align="center">
<table role="presentation" width="680" cellspacing="0" cellpadding="0" style="max-width:680px;width:100%;background:#ffffff;border:1px solid #dfe6e4;border-radius:14px;overflow:hidden">
<tr><td style="padding:24px 28px;background:#173d3a;color:#ffffff"><div style="font-size:12px;letter-spacing:1.4px;color:#9ee0d7;font-weight:700">ROADGUARD AI</div><h1 style="margin:7px 0 0;font-size:22px;line-height:1.3">Báo cáo phân tích hư hỏng mặt đường</h1></td></tr>
<tr><td style="padding:24px 28px">
<p style="margin:0 0 6px"><strong>Input:</strong> {html_text(input_name)}</p>
<p style="margin:0 0 6px"><strong>Loại dữ liệu:</strong> {html_text(input_kind)}</p>
<p style="margin:0 0 20px"><strong>Hiệu chuẩn:</strong> {html_text(calibration_text)}</p>
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="margin:0 0 20px;border:1px solid #e5e7eb;border-radius:10px;overflow:hidden">
<tr><td style="padding:12px;background:#f8faf9"><strong>Tổng hư hỏng hợp lệ</strong></td><td style="padding:12px;background:#f8faf9;text-align:right;font-size:20px;font-weight:700;color:#c2410c">{html_text(summary.get('instances', 0))}</td></tr>
{class_rows}
</table>
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="margin:0 0 22px"><tr>
<td width="50%" style="padding:12px;background:#eef8f6;border-right:6px solid #ffffff"><div style="font-size:12px;color:#52605e">Tổng chiều dài nứt</div><div style="margin-top:4px;font-size:17px;font-weight:700">{html_text(crack_display)}</div></td>
<td width="50%" style="padding:12px;background:#fff4ec;border-left:6px solid #ffffff"><div style="font-size:12px;color:#52605e">Tổng diện tích ổ gà</div><div style="margin-top:4px;font-size:17px;font-weight:700">{html_text(pothole_display)}</div></td>
</tr></table>
<div style="margin:0 0 10px;font-size:12px;letter-spacing:1px;color:#0e7490;font-weight:700">PHÂN TÍCH VÀ KHUYẾN NGHỊ DO LLM SOẠN</div>
<div style="padding:16px;border-left:4px solid #0e7490;background:#f4fafb;line-height:1.65;font-size:14px">{narrative_html}</div>
<p style="margin:20px 0 0;color:#64748b;font-size:12px;line-height:1.5">Báo cáo chi tiết được đính kèm ở định dạng CSV và Excel. Kết luận sửa chữa cần được xác nhận bằng khảo sát hiện trường.</p>
</td></tr></table></td></tr></table></body></html>"""

    message = EmailMessage()
    message["Subject"] = f"Báo cáo hư hỏng mặt đường · {input_name}"
    message["From"] = sender
    message["To"] = recipient
    message.set_content(plain)
    message.add_alternative(html, subtype="html")
    message.add_attachment(_csv_bytes(report), maintype="text", subtype="csv", filename="bao_cao.csv")
    message.add_attachment(_xlsx_bytes(report), maintype="application", subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="bao_cao.xlsx")
    return message


def _send_report_email(report: dict[str, object], recipient: str) -> None:
    host = os.environ.get("REPORT_SMTP_HOST")
    if not host:
        raise RuntimeError("Chưa cấu hình REPORT_SMTP_HOST cho email.")
    port = int(os.environ.get("REPORT_SMTP_PORT", "587"))
    sender = os.environ.get("REPORT_SMTP_FROM") or os.environ.get("REPORT_SMTP_USER")
    if not sender:
        raise RuntimeError("Chưa cấu hình REPORT_SMTP_FROM hoặc REPORT_SMTP_USER.")
    narrative = _generate_email_narrative(report)
    message = _build_report_email(report, recipient, sender, narrative)
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        if os.environ.get("REPORT_SMTP_TLS", "1") != "0":
            smtp.starttls()
        username = os.environ.get("REPORT_SMTP_USER")
        password = os.environ.get("REPORT_SMTP_PASSWORD")
        if username and not password:
            raise RuntimeError(
                "Thiếu REPORT_SMTP_PASSWORD. Gmail yêu cầu App Password 16 ký tự, "
                "không dùng mật khẩu đăng nhập thông thường."
            )
        if password and not username:
            raise RuntimeError("Thiếu REPORT_SMTP_USER cho tài khoản gửi email.")
        if username and password:
            smtp.login(username, password.replace(" ", ""))
        smtp.send_message(message)


def _local_report_reply(report: dict[str, object], prompt: str) -> str:
    """Deterministic fallback so report Q&A still works without an API key."""
    summary = report.get("summary", {})
    by_class = summary.get("by_class", {})
    question = prompt.casefold()
    calibrated = bool((report.get("calibration") or {}).get("enabled"))
    def measured(metric_key: str, metric_unit: str, pixel_key: str, pixel_unit: str) -> str:
        if calibrated:
            return f"{summary.get(metric_key, 0)} {metric_unit} ({summary.get(pixel_key, 0)} {pixel_unit})"
        return f"{summary.get(pixel_key, 0)} {pixel_unit}"
    count_text = ", ".join(f"{name}: {count}" for name, count in by_class.items()) or "không có hư hỏng"
    if "ổ gà" in question or "dien tich" in question or "diện tích" in question:
        return (
            f"Phát hiện {by_class.get('Ổ gà', 0)} ổ gà; tổng diện tích mask là "
            f"{measured('total_pothole_area_m2', 'm²', 'total_pothole_area_px2', 'px²')}."
        )
    if "nứt dọc" in question:
        return f"Tổng chiều dài nứt dọc đo được là {measured('longitudinal_crack_length_m', 'm', 'longitudinal_crack_length_px', 'px')}."
    if "nứt ngang" in question:
        return f"Tổng chiều dài nứt ngang đo được là {measured('transverse_crack_length_m', 'm', 'transverse_crack_length_px', 'px')}."
    if "frame" in question or "khung" in question:
        frames: dict[int, int] = {}
        for row in report.get("detections", []):
            frame = int(row.get("frame") or 0)
            frames[frame] = frames.get(frame, 0) + 1
        if frames:
            frame, count = max(frames.items(), key=lambda item: item[1])
            return f"Frame {frame} có nhiều detection nhất: {count} đối tượng."
    return (
        f"Input có {summary.get('instances', 0)} detection ({count_text}). "
        f"Tổng chiều dài nứt {measured('total_crack_length_m', 'm', 'total_crack_length_px', 'px')}; "
        f"diện tích ổ gà {measured('total_pothole_area_m2', 'm²', 'total_pothole_area_px2', 'px²')}."
    )


def _chat_response_text(data: object) -> str | None:
    """Extract text from common Chat Completions and Responses payloads."""
    if not isinstance(data, dict):
        return None
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = [
                str(part.get("text") or "").strip()
                for part in content if isinstance(part, dict)
            ]
            joined = "\n".join(part for part in parts if part)
            if joined:
                return joined
        text = choice.get("text") if isinstance(choice, dict) else None
        if isinstance(text, str) and text.strip():
            return text.strip()
    for key in ("output_text", "response", "content"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    output = data.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            for part in item.get("content", []) if isinstance(item, dict) else []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"].strip())
        joined = "\n".join(part for part in parts if part)
        if joined:
            return joined
    return None


def _chatbot(
    report: dict[str, object],
    prompt: str,
    api_key_override: str | None = None,
    base_url_override: str | None = None,
    model_override: str | None = None,
) -> str:
    api_key = (api_key_override or os.environ.get("CHATBOT_API_KEY") or "").strip()
    if not api_key:
        return _local_report_reply(report, prompt) + " (Tóm tắt cục bộ; chưa cấu hình CHATBOT_API_KEY.)"
    base_url = (base_url_override or os.environ.get("CHATBOT_BASE_URL") or "https://api.openai.com/v1").strip().rstrip("/")
    parsed_url = urlparse(base_url)
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise ValueError("URL chatbot phải là địa chỉ HTTPS hợp lệ.")
    model = (
        model_override
        or os.environ.get("CHATBOT_MODEL")
        or "gpt-4.1-mini"
    ).strip()
    if not model or len(model) > 100:
        raise ValueError("Tên model chatbot không hợp lệ.")
    summary = report.get("summary", {})
    detail_source = report.get("detections_unique") or report.get("detections", [])
    compact_details = [
        {
            key: row.get(key) for key in (
                "track_id", "class_code", "class_name", "confidence", "frame",
                "timestamp_s", "measurement_type", "measurement_value_px",
                "measurement_value_metric", "measurement_unit_metric", "measurement_status",
            )
        }
        for row in detail_source[:40]
    ]
    compact = {
        "input": report.get("input"),
        "summary": summary,
        "calibration": report.get("calibration"),
        "detections": compact_details,
    }
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": CHATBOT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Báo cáo hiện tại:\n{json.dumps(compact, ensure_ascii=False)}\n\nCâu hỏi: {prompt}"},
            ],
            "temperature": 0.2,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            # The 9router gateway rejects urllib's default user agent with
            # Cloudflare error 1010 even when the same API key is valid.
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RoadGuardAI/1.0",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        reply = _chat_response_text(data)
        if not reply:
            raise ValueError("empty chatbot response")
        return reply
    except HTTPError as exc:
        reason = {
            401: "API key bị từ chối",
            403: "API key không có quyền truy cập",
            429: "API đã vượt giới hạn yêu cầu",
        }.get(exc.code, f"API trả mã {exc.code}")
    except (URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError, OSError):
        reason = "không kết nối hoặc không đọc được phản hồi API"
    except (ValueError, KeyError, IndexError, TypeError):
        reason = "API trả phản hồi không đúng định dạng"
    return _local_report_reply(report, prompt) + f" (Đang dùng báo cáo cục bộ vì {reason}.)"


def _chat_runtime_overrides(payload: dict[str, object]) -> tuple[str | None, str | None, str | None]:
    """Use client URL/model only together with a client-supplied key.

    This keeps the server key bound to the server-configured provider and also
    lets an already-open browser tab (with an old model value) use the new
    server configuration safely.
    """
    api_key = str(payload.get("api_key", "")).strip()
    if len(api_key) > 4096:
        raise ValueError("API key chatbot không hợp lệ.")
    if not api_key:
        return None, None, None
    return (
        api_key,
        str(payload.get("base_url", "")).strip() or None,
        str(payload.get("model", "")).strip() or None,
    )


def _measure_detection(
    row: dict[str, object],
    image,
    confidence: float,
    calibration: dict[str, object] | None = None,
) -> None:
    """Measure a DET row using only the primary SEG mask overlapping its box."""
    import numpy as np

    code = str(row.get("class_code"))
    if code in CRACK_CODES:
        model_id, measurement_type = "seg_crack", "length_px"
    elif code == "D40":
        model_id, measurement_type = "seg_pothole", "area_px2"
    else:
        row["measurement_status"] = "not_applicable"
        return

    height, width = image.shape[:2]
    original = [int(round(float(row[key]))) for key in ("x1", "y1", "x2", "y2")]
    original[0], original[2] = max(0, original[0]), min(width, original[2])
    original[1], original[3] = max(0, original[1]), min(height, original[3])
    box_width = max(1, original[2] - original[0])
    box_height = max(1, original[3] - original[1])

    row["measurement_type"] = measurement_type
    row["segment_model"] = MODELS[model_id]["label"]
    row["segment_strategy"] = "primary_no_mask"
    row["segment_mask_pixels"] = 0
    row["segment_confidence"] = round(float(confidence), 3)
    row["segment_prediction_confidence"] = 0.0
    row["measurement_value_px"] = None
    row["measurement_value_metric"] = None
    row["measurement_unit_metric"] = None
    row["measurement_status"] = "no_mask"

    pad_x = max(8, round(box_width * 0.18))
    pad_y = max(8, round(box_height * 0.18))
    x1 = max(0, min(original[0] - pad_x, width - 1))
    x2 = max(1, min(original[2] + pad_x, width))
    y1 = max(0, min(original[1] - pad_y, height - 1))
    y2 = max(1, min(original[3] + pad_y, height))
    if x2 <= x1 or y2 <= y1:
        return
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return

    result = _load_model(model_id).predict(
        source=crop,
        conf=float(confidence),
        iou=TEST_IOU,
        imgsz=640,
        retina_masks=True,
        verbose=False,
    )[0]
    _, raw_mask = _measure_segment(result, crop.shape, measurement_type)
    raw_mask = np.asarray(raw_mask, dtype=bool)
    if not raw_mask.any():
        return

    core_x1 = max(0, original[0] - x1)
    core_x2 = min(raw_mask.shape[1], original[2] - x1)
    core_y1 = max(0, original[1] - y1)
    core_y2 = min(raw_mask.shape[0], original[3] - y1)
    mask = np.zeros_like(raw_mask, dtype=bool)
    mask[core_y1:core_y2, core_x1:core_x2] = raw_mask[
        core_y1:core_y2, core_x1:core_x2
    ]
    if not mask.any():
        return

    task = "pothole" if measurement_type == "area_px2" else "crack"
    measured = measure_masks(mask[None, ...], task=task, original_shape=mask.shape)
    value_key = "total_area_px2" if task == "pothole" else "total_length_px"
    value = float(measured[value_key])
    boxes = getattr(result, "boxes", None)
    if boxes is not None and getattr(boxes, "conf", None) is not None and len(boxes):
        row["segment_prediction_confidence"] = round(
            float(np.asarray(_to_numpy(boxes.conf)).max()), 4
        )
    row["segment_strategy"] = "primary_strict_overlap"
    row["segment_mask_pixels"] = int(mask.sum())
    row["measurement_value_px"] = round(value, 2)
    metric_value, metric_unit = _metric_measurement(
        mask, measurement_type, calibration, x1, y1
    )
    row["measurement_value_metric"] = (
        None if metric_value is None else round(metric_value, 4)
    )
    row["measurement_unit_metric"] = metric_unit
    row["measurement_status"] = "measured"
    # Removed by _draw_rows after rendering so arrays never enter JSON.
    row["_segment_visual"] = {"mask": mask, "x": int(x1), "y": int(y1)}


def _draw_rows(image, rows: list[dict[str, object]]) -> None:
    import cv2
    import numpy as np

    palette = {"D00": (255, 190, 70), "D10": (100, 220, 255), "D20": (90, 120, 255), "D40": (100, 230, 135)}
    for row in rows:
        color = palette.get(str(row["class_code"]), (220, 220, 220))
        visual = row.pop("_segment_visual", None)
        if isinstance(visual, dict):
            mask = np.asarray(visual.get("mask"), dtype=bool)
            offset_x, offset_y = int(visual.get("x", 0)), int(visual.get("y", 0))
            max_h = min(mask.shape[0], image.shape[0] - offset_y)
            max_w = min(mask.shape[1], image.shape[1] - offset_x)
            if max_h > 0 and max_w > 0:
                active = mask[:max_h, :max_w]
                roi = image[offset_y:offset_y + max_h, offset_x:offset_x + max_w]
                mask_color = np.asarray(color, dtype=np.float32)
                roi_pixels = roi[active].astype(np.float32)
                roi[active] = np.clip(
                    roi_pixels * 0.58 + mask_color * 0.42, 0, 255
                ).astype(np.uint8)
                contours, _ = cv2.findContours(
                    active.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(
                    roi, contours, -1, tuple(int(v) for v in mask_color), 2
                )
        x1, y1, x2, y2 = [int(round(float(row[key]))) for key in ("x1", "y1", "x2", "y2")]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{row['class_name']} {float(row.get('confidence') or 0):.2f}"
        if row.get("measurement_value_metric") is not None:
            unit = "m" if row.get("measurement_unit_metric") == "m" else "m2"
            text += f" · {float(row['measurement_value_metric']):.2f} {unit}"
        elif row.get("measurement_value_px") is not None:
            unit = "px" if row.get("measurement_type") == "length_px" else "px2"
            text += f" · {row['measurement_value_px']} {unit}"
        cv2.putText(image, text, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _draw_counts_overlay(image, counts: dict[str, int]) -> None:
    """Draw the notebook-standard cumulative TOTAL and per-class panel."""
    import cv2

    height, width = image.shape[:2]
    panel_width = min(440, max(210, width - 20))
    panel_height = min(290, max(180, height - 20))
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    scale = max(0.55, min(1.0, min(width, height) / 640))
    cv2.putText(
        image, f"TOTAL: {sum(int(value) for value in counts.values())}",
        (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scale,
        (0, 0, 255), max(2, round(3 * scale)), cv2.LINE_AA,
    )
    y = 88
    for code in DET_CLASSES:
        count = int(counts.get(code, 0))
        color = (0, 255, 120) if count > 0 else (180, 180, 180)
        cv2.putText(
            image, f"{code}: {count}", (20, y), cv2.FONT_HERSHEY_SIMPLEX,
            .82 * scale, color, max(1, round(2 * scale)), cv2.LINE_AA,
        )
        y += max(28, round(38 * scale))


def _reset_model_trackers(model) -> None:
    """Prevent persistent ByteTrack IDs from leaking into the next video."""
    predictor = getattr(model, "predictor", None)
    if predictor is None:
        return
    for tracker in getattr(predictor, "trackers", []) or []:
        reset = getattr(tracker, "reset", None)
        if callable(reset):
            reset()
    if hasattr(predictor, "vid_path"):
        try:
            predictor.vid_path = [None] * len(predictor.vid_path)
        except TypeError:
            predictor.vid_path = None


def _row_box_iou(first: dict[str, object], second: dict[str, object]) -> float:
    left = max(float(first["x1"]), float(second["x1"]))
    top = max(float(first["y1"]), float(second["y1"]))
    right = min(float(first["x2"]), float(second["x2"]))
    bottom = min(float(first["y2"]), float(second["y2"]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first["x2"]) - float(first["x1"])) * max(0.0, float(first["y2"]) - float(first["y1"]))
    second_area = max(0.0, float(second["x2"]) - float(second["x1"])) * max(0.0, float(second["y2"]) - float(second["y1"]))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _canonicalize_track_rows(
    rows: list[dict[str, object]],
    tracking_state: dict[str, object],
    frame_index: int,
    fps: float,
    frame_shape,
) -> None:
    """Merge short ByteTrack ID switches that refer to the same road defect.

    A new raw ID is joined to a recent canonical track only when class matches
    and its box overlaps the predicted box or remains spatially close with a
    compatible scale. Tracks already updated in the current frame are excluded
    so nearby simultaneous defects are not collapsed into one.
    """
    aliases: dict[int, int] = tracking_state.setdefault("aliases", {})
    tracks: dict[int, dict[str, object]] = tracking_state.setdefault("tracks", {})
    merged_ids: set[int] = tracking_state.setdefault("merged_ids", set())
    height, width = int(frame_shape[0]), int(frame_shape[1])
    diagonal = max(1.0, (width * width + height * height) ** 0.5)
    max_gap = max(3, round(max(fps, 1.0) * TRACK_REID_SECONDS))

    for row in rows:
        raw_id = int(row["track_id"])
        row["raw_track_id"] = raw_id
        canonical_id = aliases.get(raw_id)
        center_x = (float(row["x1"]) + float(row["x2"])) / 2
        center_y = (float(row["y1"]) + float(row["y2"])) / 2
        area = max(1.0, (float(row["x2"]) - float(row["x1"])) * (float(row["y2"]) - float(row["y1"])))

        if canonical_id is None:
            candidates = []
            for candidate_id, track in tracks.items():
                gap = frame_index - int(track["last_frame"])
                if gap <= 0 or gap > max_gap or track["class_code"] != row.get("class_code"):
                    continue
                predicted_x = float(track["center_x"]) + float(track.get("velocity_x", 0.0)) * gap
                predicted_y = float(track["center_y"]) + float(track.get("velocity_y", 0.0)) * gap
                distance = ((center_x - predicted_x) ** 2 + (center_y - predicted_y) ** 2) ** 0.5 / diagonal
                previous_area = max(1.0, float(track["area"]))
                area_ratio = area / previous_area
                overlap = _row_box_iou(row, track["box"])
                if overlap >= 0.10 or (distance <= 0.08 and 0.20 <= area_ratio <= 5.0):
                    candidates.append((0 if overlap >= 0.10 else 1, distance, -overlap, candidate_id))
            if candidates:
                canonical_id = min(candidates)[-1]
                aliases[raw_id] = canonical_id
                if raw_id != canonical_id:
                    merged_ids.add(raw_id)
            else:
                canonical_id = raw_id
                aliases[raw_id] = canonical_id

        previous = tracks.get(canonical_id)
        velocity_x = velocity_y = 0.0
        if previous is not None:
            gap = max(1, frame_index - int(previous["last_frame"]))
            instant_x = (center_x - float(previous["center_x"])) / gap
            instant_y = (center_y - float(previous["center_y"])) / gap
            velocity_x = .65 * float(previous.get("velocity_x", 0.0)) + .35 * instant_x
            velocity_y = .65 * float(previous.get("velocity_y", 0.0)) + .35 * instant_y
        row["track_id"] = canonical_id
        row["track_merged"] = raw_id != canonical_id
        row["track_observations"] = int(previous.get("hits", 0) if previous else 0) + 1
        tracks[canonical_id] = {
            "class_code": row.get("class_code"),
            "last_frame": frame_index,
            "center_x": center_x,
            "center_y": center_y,
            "velocity_x": velocity_x,
            "velocity_y": velocity_y,
            "area": area,
            "hits": row["track_observations"],
            "box": {key: row[key] for key in ("x1", "y1", "x2", "y2")},
        }


def _pipeline_report(run_id: str, filename: str, rows: list[dict[str, object]], width: int, height: int, elapsed_ms: int, kind: str, url: str, **extra) -> dict[str, object]:
    unique_rows, _ = _deduplicate_tracked_rows(rows)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "pipeline",
        "input": {"name": filename, "kind": kind, "width": width, "height": height},
        "pipeline_models": {key: item["checkpoint"].as_posix() for key, item in MODELS.items()},
        "seg_filter_policy": "primary_mask_required",
        "detections": rows,
        "detections_unique": unique_rows,
        "summary": _summarize(rows),
        "density": _density(rows, width, height, extra.get("frames")),
        "elapsed_ms": elapsed_ms,
        **extra,
    }
    _save_report(run_id, report)
    return {
        "run_id": run_id,
        "kind": kind,
        "url": url,
        "instances": report["summary"]["instances"],
        "elapsed_ms": elapsed_ms,
        "label": "Pipeline DET → SEG",
        "checkpoint": "3 checkpoint chính · chỉ giữ mask SEG chính",
        "detections": report["detections"],
        "detections_unique": report["detections_unique"],
        "summary": report["summary"],
        "density": report["density"],
        "frames": report.get("frames"),
        "detection_every": report.get("detection_every"),
        "seg_sampled_every": report.get("seg_sampled_every"),
        "seg_every_frame": report.get("seg_every_frame"),
        "tracker": report.get("tracker"),
        "track_merge": report.get("track_merge"),
        "merged_track_ids": report.get("merged_track_ids"),
        "iou": report.get("iou"),
        "unique_track_counts": report.get("unique_track_counts"),
        "calibration": report.get("calibration"),
        "seg_filter_policy": report.get("seg_filter_policy"),
        "report_url": f"/api/report/{run_id}",
        "csv_url": f"/api/export/{run_id}?format=csv",
        "xlsx_url": f"/api/export/{run_id}?format=xlsx",
    }


def _run_pipeline_image(
    source: Path,
    filename: str,
    confidence: float,
    seg_confidence: float = 0.12,
    calibration_spec: dict[str, object] | None = None,
) -> dict[str, object]:
    import cv2

    image = cv2.imread(str(source))
    if image is None:
        raise ValueError("Không đọc được ảnh đầu vào.")
    run_id = uuid.uuid4().hex[:12]
    output_dir = MEDIA_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    calibration = _prepare_calibration(calibration_spec, image.shape[:2])
    det_model = _load_model("det")
    det_result = det_model.predict(
        source=image, conf=confidence, iou=TEST_IOU, imgsz=640, verbose=False
    )[0]
    rows = _detection_rows(det_result)
    for row in rows:
        _measure_detection(row, image, seg_confidence, calibration)
    rows = [row for row in rows if _pipeline_row_is_accepted(row)]
    plotted = image.copy()
    _draw_rows(plotted, rows)
    image_counts: dict[str, int] = {}
    for row in rows:
        code = str(row.get("class_code") or "")
        image_counts[code] = image_counts.get(code, 0) + 1
    _draw_counts_overlay(plotted, image_counts)
    output = output_dir / "result.jpg"
    if not cv2.imwrite(str(output), plotted):
        raise RuntimeError("Không ghi được ảnh pipeline.")
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    return _pipeline_report(
        run_id, filename, rows, image.shape[1], image.shape[0], elapsed_ms, "image", f"/media/{run_id}/{output.name}",
        calibration=_calibration_payload(calibration),
        iou=TEST_IOU,
    )


def _process_pipeline_video_frame(
    det_model,
    frame,
    frame_index: int,
    fps: float,
    confidence: float,
    seg_confidence: float,
    calibration: dict[str, object] | None = None,
    tracking_state: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Run DET, specialist SEG measurement, and annotation on every frame."""
    det_result = det_model.track(
        source=frame,
        persist=True,
        conf=confidence,
        iou=TEST_IOU,
        tracker="bytetrack.yaml",
        imgsz=640,
        verbose=False,
    )[0]
    # Match the notebook: only tracked boxes participate in video counting,
    # drawing, SEG measurement, and the final unique-object summary.
    frame_rows = [
        row for row in _detection_rows(det_result, frame_index, frame_index / fps)
        if row.get("track_id") is not None
    ]
    state = tracking_state if tracking_state is not None else {
        "counted_ids": set(), "counts": {}, "aliases": {}, "tracks": {}, "merged_ids": set()
    }
    _canonicalize_track_rows(frame_rows, state, frame_index, fps, frame.shape[:2])
    for row in frame_rows:
        _measure_detection(row, frame, seg_confidence, calibration)
        row["track_confirmed"] = (
            int(row.get("track_observations") or 0) >= MIN_TRACK_OBSERVATIONS
            or row.get("measurement_status") == "measured"
        )
    accepted_rows = [
        row for row in frame_rows
        if row.get("track_confirmed") and _pipeline_row_is_accepted(row)
    ]
    counted_ids = state.setdefault("counted_ids", set())
    counts = state.setdefault("counts", {})
    for row in accepted_rows:
        track_id = int(row["track_id"])
        code = str(row.get("class_code") or "")
        count_key = (code, track_id)
        if count_key not in counted_ids:
            counted_ids.add(count_key)
            counts[code] = int(counts.get(code, 0)) + 1
    _draw_rows(frame, accepted_rows)
    _draw_counts_overlay(frame, counts)
    return accepted_rows


def _run_pipeline_video(
    source: Path,
    filename: str,
    confidence: float,
    seg_confidence: float = 0.12,
    calibration_spec: dict[str, object] | None = None,
) -> dict[str, object]:
    import cv2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("Không mở được video đầu vào.")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Video không có kích thước hợp lệ.")
    calibration = _prepare_calibration(calibration_spec, (height, width))
    run_id = uuid.uuid4().hex[:12]
    output_dir = MEDIA_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "result.mp4"
    started = time.perf_counter()
    det_model = _load_model("det")
    writer = _BrowserVideoWriter(output, fps, width, height)
    rows: list[dict[str, object]] = []
    tracking_state: dict[str, object] = {
        "counted_ids": set(), "counts": {}, "aliases": {}, "tracks": {}, "merged_ids": set()
    }
    frame_index = 0
    with _TRACKING_LOCK:
        _reset_model_trackers(det_model)
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                # DET and the appropriate specialist SEG model run on every frame.
                # This intentionally avoids temporal sampling so the report and
                # delivered video represent continuous frame-by-frame inspection.
                rows.extend(_process_pipeline_video_frame(
                    det_model, frame, frame_index, fps, confidence, seg_confidence,
                    calibration, tracking_state,
                ))
                writer.write(frame)
                frame_index += 1
        finally:
            capture.release()
            writer.close()
            _reset_model_trackers(det_model)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    return _pipeline_report(
        run_id, filename, rows, width, height, elapsed_ms, "video", f"/media/{run_id}/{output.name}",
        frames=frame_index,
        detection_every=1,
        seg_sampled_every=1,
        sampled_every=1,
        seg_every_frame=True,
        tracker="bytetrack.yaml",
        track_merge=f"class + <= {TRACK_REID_SECONDS:g}s gap + predicted spatial overlap",
        merged_track_ids=len(tracking_state["merged_ids"]),
        iou=TEST_IOU,
        unique_track_counts=tracking_state["counts"],
        fps=round(fps, 2),
        calibration=_calibration_payload(calibration),
    )


def _run_inference(
    model_id: str,
    source: Path,
    filename: str,
    confidence: float,
    calibration_spec: dict[str, object] | None = None,
) -> dict[str, object]:
    model_info = MODELS[model_id]
    model = _load_model(model_id)
    run_id = uuid.uuid4().hex[:12]
    output_dir = MEDIA_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    is_video = _is_video(filename)
    kwargs = {
        "source": str(source),
        "conf": confidence,
        "imgsz": 640,
        "retina_masks": model_info["task"] == "segment",
        "iou": TEST_IOU,
        "verbose": False,
        "save": is_video,
        "project": str(MEDIA_ROOT),
        "name": run_id,
        "exist_ok": True,
    }
    if is_video:
        results = list(model.predict(stream=True, **kwargs))
        media_files = [
            path for path in output_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        ]
        if not media_files:
            raise RuntimeError("Model không tạo được file video kết quả.")
        raw_output = media_files[0]
        output = _transcode_browser_video(raw_output, output_dir / "result_h264.mp4")
        if raw_output != output:
            raw_output.unlink(missing_ok=True)
        instances = sum(_count_instances(result) for result in results)
        kind = "video"
        rows: list[dict[str, object]] = []
        fps = 0.0
        try:
            import cv2

            capture = cv2.VideoCapture(str(source))
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            capture.release()
        except Exception:
            fps = 0.0
        if results:
            height, width = tuple(getattr(results[0], "orig_shape", (0, 0)))
        else:
            width = height = 0
        calibration = _prepare_calibration(calibration_spec, (height, width)) if width and height else None
        for frame_index, result in enumerate(results):
            timestamp = frame_index / fps if fps else 0.0
            if model_info["task"] == "segment":
                rows.extend(_segment_rows(result, model_id, frame_index, timestamp, calibration))
            else:
                rows.extend(_detection_rows(result, frame_index, timestamp))
    else:
        import cv2

        results = model.predict(**kwargs)
        if not results:
            raise RuntimeError("Model không trả về kết quả.")
        plotted = results[0].plot()
        output = output_dir / "result.jpg"
        if not cv2.imwrite(str(output), plotted):
            raise RuntimeError("Không ghi được ảnh kết quả.")
        instances = _count_instances(results[0])
        kind = "image"
        original = getattr(results[0], "orig_img", plotted)
        height, width = original.shape[:2]
        calibration = _prepare_calibration(calibration_spec, (height, width))
        rows = (
            _segment_rows(results[0], model_id, calibration=calibration)
            if model_info["task"] == "segment"
            else _detection_rows(results[0])
        )

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "single_model",
        "input": {"name": filename, "kind": kind, "width": width, "height": height},
        "model": model_info["checkpoint"].as_posix(),
        "detections": rows,
        "summary": _summarize(rows),
        "density": _density(rows, width or 1, height or 1),
        "calibration": _calibration_payload(calibration),
        "elapsed_ms": elapsed_ms,
    }
    _save_report(run_id, report)
    return {
        "run_id": run_id,
        "kind": kind,
        "url": f"/media/{run_id}/{output.name}",
        "instances": instances,
        "elapsed_ms": elapsed_ms,
        "label": model_info["label"],
        "checkpoint": model_info["checkpoint"].as_posix(),
        "detections": report["detections"],
        "summary": report["summary"],
        "density": report["density"],
        "calibration": report["calibration"],
        "report_url": f"/api/report/{run_id}",
        "csv_url": f"/api/export/{run_id}?format=csv",
        "xlsx_url": f"/api/export/{run_id}?format=xlsx",
    }


def _execute_analysis(
    input_path: Path,
    filename: str,
    mode: str,
    model_id: str,
    confidence: float,
    seg_confidence: float,
    calibration_spec: dict[str, object] | None,
) -> dict[str, object]:
    if mode == "pipeline":
        runner = _run_pipeline_video if _is_video(filename) else _run_pipeline_image
        return runner(input_path, filename, confidence, seg_confidence, calibration_spec)
    return _run_inference(model_id, input_path, filename, confidence, calibration_spec)


def _run_analysis_job(
    job_id: str,
    input_path: Path,
    filename: str,
    mode: str,
    model_id: str,
    confidence: float,
    seg_confidence: float,
    calibration_spec: dict[str, object] | None,
) -> None:
    with _JOB_LOCK:
        _JOBS[job_id]["status"] = "running"
        _JOBS[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
    try:
        with _ANALYSIS_LOCK:
            result = _execute_analysis(
                input_path, filename, mode, model_id, confidence,
                seg_confidence, calibration_spec,
            )
        with _JOB_LOCK:
            _JOBS[job_id].update({
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "result": result,
            })
    except Exception as exc:  # pragma: no cover - exercised through job status boundary
        with _JOB_LOCK:
            _JOBS[job_id].update({
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"Lỗi suy luận: {exc}",
            })
    finally:
        _cleanup_uploaded_file(input_path)


def _start_analysis_job(
    input_path: Path,
    filename: str,
    mode: str,
    model_id: str,
    confidence: float,
    seg_confidence: float,
    calibration_spec: dict[str, object] | None,
) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _JOB_LOCK:
        completed = [key for key, value in _JOBS.items() if value.get("status") in {"completed", "failed"}]
        while len(_JOBS) >= 100 and completed:
            _JOBS.pop(completed.pop(0), None)
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "filename": filename,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    worker = threading.Thread(
        target=_run_analysis_job,
        args=(
            job_id, input_path, filename, mode, model_id, confidence,
            seg_confidence, calibration_spec,
        ),
        name=f"roadguard-{job_id}",
        daemon=True,
    )
    worker.start()
    return job_id


def _job_payload(job_id: str) -> dict[str, object]:
    safe_id = _safe_name(job_id)
    with _JOB_LOCK:
        job = _JOBS.get(safe_id)
        if job is None:
            raise FileNotFoundError("Không tìm thấy lượt phân tích đang chạy.")
        return dict(job)


class Handler(BaseHTTPRequestHandler):
    server_version = "RoadDamageTest/1.0"

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        print(f"[{self.log_date_time_string()}] {format % args}")

    def _send_bytes(self, payload: bytes, content_type: str, status: int = 200, extra_headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, value: object, status: int = 200) -> None:
        self._send_bytes(
            json.dumps(value, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def _send_file(self, candidate: Path, content_type: str, *, head_only: bool = False) -> None:
        """Stream a file and honor byte ranges required by HTML5 video."""
        size = candidate.stat().st_size
        try:
            byte_range = _parse_byte_range(self.headers.get("Range"), size)
        except ValueError:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start, end = byte_range if byte_range is not None else (0, size - 1)
        length = end - start + 1
        self.send_response(206 if byte_range is not None else 200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if byte_range is not None:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only:
            return
        try:
            with candidate.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # Browsers commonly cancel a range once enough metadata is read.
            return

    def _resolve_media(self, request_path: str) -> Path:
        relative = Path(*request_path.removeprefix("/media/").split("/"))
        candidate = (MEDIA_ROOT / relative).resolve()
        try:
            candidate.relative_to(MEDIA_ROOT.resolve())
        except ValueError as exc:
            raise ValueError("Đường dẫn không hợp lệ.") from exc
        if not candidate.is_file():
            raise FileNotFoundError("Không tìm thấy file kết quả.")
        return candidate

    def do_HEAD(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if not path.startswith("/media/"):
            self.send_response(404)
            self.end_headers()
            return
        try:
            candidate = self._resolve_media(path)
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            self._send_file(candidate, content_type, head_only=True)
        except ValueError:
            self.send_response(400)
            self.end_headers()
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            self._send_bytes(WEB_FILE.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/models":
            self._send_json(_model_payload())
            return
        if path == "/api/history":
            try:
                limit = int(parse_qs(parsed.query).get("limit", ["50"])[0])
                self._send_json(_history_payload(limit=limit))
            except ValueError:
                self._send_json({"error": "Giới hạn lịch sử không hợp lệ."}, 400)
            return
        if path.startswith("/api/history/"):
            try:
                self._send_json(_history_result(path.removeprefix("/api/history/")))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, 400)
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, 404)
            return
        if path.startswith("/api/jobs/"):
            try:
                self._send_json(_job_payload(path.removeprefix("/api/jobs/")))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, 400)
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, 404)
            return
        if path.startswith("/api/report/"):
            try:
                self._send_json(_read_report(_safe_name(path.removeprefix("/api/report/"))))
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, 404)
            return
        if path.startswith("/api/export/"):
            run_id = _safe_name(path.removeprefix("/api/export/").strip("/"))
            format_name = parse_qs(parsed.query).get("format", ["csv"])[0].lower()
            try:
                report = _read_report(run_id)
                if format_name == "xlsx":
                    payload, content_type, filename = _xlsx_bytes(report), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "bao_cao.xlsx"
                elif format_name == "csv":
                    payload, content_type, filename = _csv_bytes(report), "text/csv; charset=utf-8", "bao_cao.csv"
                else:
                    self._send_json({"error": "Chỉ hỗ trợ csv hoặc xlsx."}, 400)
                    return
                self._send_bytes(payload, content_type, extra_headers={"Content-Disposition": f'attachment; filename="{filename}"'})
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, 404)
            return
        if path.startswith("/media/"):
            try:
                candidate = self._resolve_media(path)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, 400)
                return
            except FileNotFoundError as exc:
                self._send_json({"error": "Không tìm thấy file kết quả."}, 404)
                return
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            self._send_file(candidate, content_type)
            return
        self._send_json({"error": "Không tìm thấy tài nguyên."}, 404)

    def do_POST(self) -> None:  # noqa: N802
        endpoint = urlparse(self.path).path
        if endpoint in {"/api/email", "/api/chat"}:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 2 * 1024 * 1024:
                    raise ValueError("Request không hợp lệ.")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                report = _read_report(str(payload.get("run_id", "")))
                if endpoint == "/api/email":
                    recipient = str(payload.get("recipient", "")).strip()
                    if not recipient or "@" not in recipient:
                        raise ValueError("Nhập email người nhận hợp lệ.")
                    _send_report_email(report, recipient)
                    self._send_json({"ok": True, "message": "Đã gửi báo cáo qua email."})
                else:
                    prompt = str(payload.get("message", "")).strip()
                    if not prompt:
                        raise ValueError("Nhập câu hỏi cho chatbot.")
                    api_key, base_url, chat_model = _chat_runtime_overrides(payload)
                    self._send_json({
                        "reply": _chatbot(
                            report,
                            prompt,
                            api_key_override=api_key,
                            base_url_override=base_url,
                            model_override=chat_model,
                        )
                    })
            except (ValueError, FileNotFoundError, RuntimeError) as exc:
                self._send_json({"error": str(exc)}, 400)
            except Exception as exc:  # pragma: no cover - defensive UI boundary
                self._send_json({"error": f"Lỗi dịch vụ: {exc}"}, 500)
            return
        if endpoint != "/api/test":
            self._send_json({"error": "Endpoint không hợp lệ."}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._send_json({"error": "File phải lớn hơn 0 và không vượt quá 200 MB."}, 413)
            return
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            self._send_json({"error": "Request phải là multipart/form-data."}, 415)
            return
        try:
            fields = _parse_multipart(content_type, self.rfile.read(length))
            model_id = fields.get("model", (None, b""))[1].decode("utf-8").strip()
            mode = fields.get("mode", (None, b"single"))[1].decode("utf-8").strip().lower()
            filename, file_bytes = fields.get("file", (None, b""))
            confidence = float(fields.get("confidence", (None, b"0.25"))[1].decode("utf-8"))
            seg_confidence = float(fields.get("seg_confidence", (None, b"0.12"))[1].decode("utf-8"))
            calibration_enabled = fields.get("calibration_enabled", (None, b"false"))[1].decode("utf-8").strip().lower() in {"1", "true", "yes"}
            calibration_spec = None
            if calibration_enabled:
                try:
                    calibration_points = json.loads(fields.get("calibration_points", (None, b"[]"))[1].decode("utf-8"))
                    calibration_spec = {
                        "points": calibration_points,
                        "road_width_m": float(fields.get("road_width_m", (None, b"0"))[1].decode("utf-8")),
                        "road_length_m": float(fields.get("road_length_m", (None, b"0"))[1].decode("utf-8")),
                    }
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    raise ValueError("Dữ liệu hiệu chuẩn 4 điểm không hợp lệ.") from exc
            if not filename or not file_bytes:
                raise ValueError("Hãy chọn một file ảnh hoặc video.")
            _validate_upload_size(len(file_bytes))
            if not 0.01 <= confidence <= 0.99:
                raise ValueError("Confidence phải nằm trong khoảng 0.01–0.99.")
            if not 0.01 <= seg_confidence <= 0.99:
                raise ValueError("SEG confidence phải nằm trong khoảng 0.01–0.99.")
            if mode not in {"single", "pipeline"}:
                raise ValueError("Mode test không hợp lệ.")
            if mode == "single" and model_id not in MODELS:
                raise ValueError("Model không hợp lệ.")
            input_dir = MEDIA_ROOT / "inputs"
            input_dir.mkdir(parents=True, exist_ok=True)
            input_path = input_dir / f"{uuid.uuid4().hex[:12]}_{_safe_name(filename)}"
            input_path.write_bytes(file_bytes)
            try:
                job_id = _start_analysis_job(
                    input_path, filename, mode, model_id, confidence,
                    seg_confidence, calibration_spec,
                )
            except Exception:
                _cleanup_uploaded_file(input_path)
                raise
            self._send_json({"job_id": job_id, "status": "queued"}, 202)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            self._send_json({"error": str(exc)}, 400)
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self._send_json({"error": f"Lỗi suy luận: {exc}"}, 500)


def main() -> None:
    _load_local_chatbot_env()
    parser = argparse.ArgumentParser(description="Local model testing interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not WEB_FILE.is_file():
        raise SystemExit(f"Không tìm thấy giao diện: {WEB_FILE}")
    MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Road Damage test interface: http://{args.host}:{args.port}")
    print("Nhấn Ctrl+C để dừng.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
