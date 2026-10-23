"""Refresh the factual status section of the single project Markdown log."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _json(path):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else None


def refresh_readme(root=None):
    root = Path(root).resolve() if root else Path(__file__).resolve().parents[2]
    path = root / "README.md"
    if not path.exists():
        return
    lines = [f"Cập nhật: {datetime.now(timezone.utc).isoformat()} (UTC).", "", "### Di chuyển đã kiểm chứng", "",
             "| Đường dẫn cũ | Đường dẫn mới | Số file | Tổng byte |", "|---|---|---:|---:|"]
    for record in _json(root / "reports/migration.json") or []:
        lines.append(f"| `{record['source']}` | `{record['destination']}` | {record['files']} | {record['bytes']} |")
    lines.extend(["", "### Dataset đã kiểm tra", "", "| Dataset | Split | Ảnh gốc | Sử dụng | Loại | Polygon |", "|---|---|---:|---:|---:|---:|"])
    for name in ("crack", "pothole"):
        audit = _json(root / f"reports/data_quality/{name}/audit.json")
        if audit:
            for split, stat in audit["splits"].items():
                lines.append(f"| {name} | {split} | {stat['total_images']} | {stat['usable_images']} | {stat['excluded_images']} | {stat['polygon_count']} |")
    lines.extend(["", "### Tiến độ pilot 30 epoch", "", "| Dataset | Model | Trạng thái | Epoch | Batch | Giờ train | Cập nhật UTC |", "|---|---|---|---:|---:|---:|---|"])
    details = []
    for dataset in ("crack", "pothole"):
        for size in ("n", "s", "m"):
            report_path = root / f"runs/seg/pilot30/{dataset}/yolov8{size}_seg/report.json"
            report = _json(report_path)
            if not report:
                lines.append(f"| {dataset} | YOLOv8{size}-seg | Chưa chạy | — | — | — | — |")
                continue
            hours = report.get("train_wall_seconds", 0) / 3600
            lines.append(f"| {dataset} | {report['model']} | {report['status']} / {report.get('phase','')} | {report.get('epochs_completed',0)}/30 | {report.get('batch','—')} | {hours:.3f} | {report.get('updated_at','')} |")
            details.extend(["", f"#### {dataset} / {report['model']}", "", f"- Nguồn chi tiết: `{report_path.relative_to(root).as_posix()}`.",
                            f"- Bắt đầu tạo lượt chạy: {report.get('created_at','')}; hoàn tất: {report.get('completed_at','chưa hoàn tất')}.",
                            f"- Fingerprint thí nghiệm: `{report.get('fingerprint','chưa có')}`."])
            for attempt in report.get("attempts", []):
                details.append(f"- Lần chạy từ epoch {attempt.get('resume_from_epoch',0)}: {attempt.get('started_at','')} → {attempt.get('ended_at','đang chạy')}; batch {attempt.get('batch_effective',attempt.get('batch_requested'))}; thời gian {attempt.get('wall_seconds','đang đo')} giây.")
            if report.get("validation"):
                m = report["validation"]["mask"]
                details.append(f"- Mask validation: mAP50={m['map50']:.5f}; mAP50–95={m['map50_95']:.5f}; P={m['precision']:.5f}; R={m['recall']:.5f}.")
            if report.get("benchmark"):
                b = report["benchmark"]
                details.append(f"- Benchmark: mean {b['mean_ms']:.3f} ms/ảnh, median {b['median_ms']:.3f}, p95 {b['p95_ms']:.3f}; {b['fps']:.2f} FPS.")
            for error in report.get("errors", []):
                message = str(error.get("message", "")).replace("\n", " ")
                details.append(f"- Lỗi {error.get('time','')} ({error.get('phase','')}): {error.get('type','')} — {message}.")
    lines.extend(details)
    start, end = "<!-- GENERATED_STATUS_START -->", "<!-- GENERATED_STATUS_END -->"
    content = path.read_text(encoding="utf-8")
    if start not in content or end not in content:
        return
    before = content.split(start, 1)[0]
    after = content.split(end, 1)[1]
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(before + start + "\n" + "\n".join(lines) + "\n" + end + after, encoding="utf-8")
    temporary.replace(path)
