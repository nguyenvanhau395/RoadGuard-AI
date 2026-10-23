"""Portable, source-grounded Vietnamese comparison of segmentation experiments.

This module reads the runtime's report.json artifacts. It never starts training,
re-evaluates checkpoints, or manufactures values for unfinished experiments.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import quote


MODELS = ("yolov8n-seg", "yolov8s-seg", "yolov8m-seg")
DATASETS = {
    "crack": ("Crack", "Phân đoạn vết nứt · phục vụ đo chiều dài"),
    "pothole": ("Pothole", "Phân đoạn ổ gà · phục vụ đo diện tích"),
}
COLORS = {"yolov8n-seg": "#087f8c", "yolov8s-seg": "#8863bf", "yolov8m-seg": "#c36a28"}
METRICS = ("precision", "recall", "map50", "map50_95")


def _number(value: Any, *, positive: bool = False) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def _metric(report: dict, key: str) -> Any:
    validation = report.get("validation")
    mask = validation.get("mask") if isinstance(validation, dict) else None
    return mask.get(key) if isinstance(mask, dict) else None


def _evaluated(report: dict, stage: str) -> bool:
    """Only complete, comparable records may affect any recommendation."""
    if report.get("status") != "completed" or report.get("stage") != stage:
        return False
    validation = report.get("validation")
    if not isinstance(validation, dict) or validation.get("split") != "val":
        return False
    epochs = report.get("epochs_completed")
    requested = report.get("epochs_requested")
    if stage == "pilot30" and (epochs != 30 or requested != 30):
        return False
    if not _number(epochs, positive=True):
        return False
    if not all(_number(_metric(report, key)) and _metric(report, key) <= 1 for key in METRICS):
        return False
    if not _number(report.get("train_wall_seconds"), positive=True):
        return False
    benchmark = report.get("benchmark")
    return isinstance(benchmark, dict) and all(_number(benchmark.get(key), positive=True) for key in ("mean_ms", "median_ms", "p95_ms", "fps"))


def _fmt(value: Any, decimals: int = 2, scale: float = 1) -> str:
    return f"{value * scale:,.{decimals}f}" if _number(value) else "—"


def _model_label(model: str | None) -> str:
    return model.replace("yolov8", "YOLOv8") if model else "Chưa xác định"


def _safe(value: Any) -> str:
    return escape(str(value), quote=True)


def _source_link(path: Path, output: Path, label: str) -> str:
    # os.path.relpath permits report regeneration from an arbitrary project root.
    import os

    relative = Path(os.path.relpath(path, output.parent)).as_posix()
    return f'<a href="{quote(relative, safe="/.:_-~")}">{_safe(label)}</a>'


def _read_report(path: Path, dataset: str, model: str, stage: str) -> dict:
    base = {"dataset": dataset, "model": model, "stage": stage, "status": "pending"}
    if not path.is_file():
        return {**base, "_source_exists": False, "_source": str(path)}
    try:
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(report, dict):
            raise ValueError("report.json phải là một JSON object")
        for key, expected in (("dataset", dataset), ("model", model), ("stage", stage)):
            if report.get(key) != expected:
                raise ValueError(f"{key} không khớp thư mục: {report.get(key)!r} != {expected!r}")
        return {**report, "_source_exists": True, "_source": str(path)}
    except (OSError, ValueError, TypeError) as exc:
        return {**base, "status": "invalid", "errors": [str(exc)], "_source_exists": True, "_source": str(path)}


def _status(report: dict, stage: str) -> tuple[str, str]:
    if _evaluated(report, stage):
        return "complete", "Đã đánh giá"
    raw = report.get("status", "pending")
    if raw == "completed":
        return "warning", "Thiếu dữ liệu so sánh"
    return {
        "pending": ("pending", "Chờ chạy"),
        "running": ("running", "Đang chạy"),
        "failed": ("warning", "Lỗi"),
        "invalid": ("warning", "Báo cáo không hợp lệ"),
    }.get(raw, ("pending", str(raw)))


def _scatter(reports: list[dict], stage: str) -> str:
    records = [r for r in reports if _evaluated(r, stage)]
    if not records:
        return '<div class="empty-chart">Biểu đồ xuất hiện khi có kết quả đánh giá hợp lệ.</div>'
    xs = [r["benchmark"]["mean_ms"] for r in records]
    ys = [_metric(r, "map50_95") * 100 for r in records]
    xmax = max(xs) * 1.22
    ymin = max(0, math.floor((min(ys) - 5) / 5) * 5)
    ymax = min(100, math.ceil((max(ys) + 5) / 5) * 5)
    if ymax <= ymin:
        ymax = ymin + 5
    left, top, width, height = 62, 29, 462, 210
    parts = ['<svg viewBox="0 0 560 298" role="img" aria-label="So sánh độ trễ suy luận và mask mAP50–95">']
    for tick in range(5):
        x = left + width * tick / 4
        y = top + height * tick / 4
        parts.append(f'<line class="grid" x1="{left}" y1="{y}" x2="{left + width}" y2="{y}"/>')
        parts.append(f'<text class="tick" x="{left - 10}" y="{y + 4}" text-anchor="end">{ymax - (ymax - ymin) * tick / 4:.1f}</text>')
        parts.append(f'<text class="tick" x="{x}" y="{top + height + 21}" text-anchor="middle">{xmax * tick / 4:.1f}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + height}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}"/>')
    parts.append('<text class="axis-title" x="62" y="15">Mask mAP50–95 (%) ↑</text>')
    parts.append('<text class="axis-title" x="293" y="285" text-anchor="middle">Độ trễ trung bình / ảnh (ms) · thấp hơn tốt hơn</text>')
    positions = []
    for record in records:
        model = record["model"]
        x = left + width * record["benchmark"]["mean_ms"] / xmax
        y = top + height * (ymax - _metric(record, "map50_95") * 100) / (ymax - ymin)
        shift = -12
        # Separate nearby text without distorting measured point positions.
        if any(abs(x - px) < 60 and abs(y - py) < 26 for px, py in positions):
            shift = 23
        positions.append((x, y))
        label = _model_label(model)
        title = f"{label}: {_metric(record, 'map50_95') * 100:.2f}%, {record['benchmark']['mean_ms']:.2f} ms"
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{COLORS[model]}" stroke="white" stroke-width="2"><title>{_safe(title)}</title></circle>')
        parts.append(f'<text class="point-label" x="{x:.2f}" y="{y + shift:.2f}" text-anchor="middle" fill="{COLORS[model]}">{_safe(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _duration_chart(reports: list[dict], stage: str) -> str:
    records = [r for r in reports if _evaluated(r, stage)]
    if not records:
        return '<div class="empty-chart">Chưa có thời gian huấn luyện hoàn tất để so sánh.</div>'
    maximum = max(r["train_wall_seconds"] / 3600 for r in records)
    lookup = {r["model"]: r for r in records}
    parts = ['<svg viewBox="0 0 560 298" role="img" aria-label="Thời gian huấn luyện theo mô hình">']
    parts.append('<text class="axis-title" x="24" y="22">Thời gian huấn luyện (giờ) · thấp hơn tốt hơn</text>')
    for index, model in enumerate(MODELS):
        y = 56 + index * 69
        parts.append(f'<text class="bar-label" x="24" y="{y + 18}">{_safe(_model_label(model))}</text>')
        parts.append(f'<rect x="160" y="{y}" width="290" height="28" rx="4" fill="#eef2f2"/>')
        if model in lookup:
            hours = lookup[model]["train_wall_seconds"] / 3600
            bar_width = 290 * hours / maximum
            parts.append(f'<rect x="160" y="{y}" width="{bar_width:.2f}" height="28" rx="4" fill="{COLORS[model]}"/>')
            parts.append(f'<text class="bar-label" x="466" y="{y + 19}">{hours:.2f} h</text>')
        else:
            parts.append(f'<text class="tick" x="174" y="{y + 19}">Chưa hoàn tất</text>')
    parts.append('<text class="tick" x="24" y="282">Thời gian thực; không bao gồm đánh giá và benchmark.</text></svg>')
    return "".join(parts)


def _table(reports: list[dict], stage: str) -> str:
    columns = ("Mô hình", "Trạng thái", "Epoch", "Batch", "Train (giờ)", "P (%)", "R (%)", "mAP50 (%)", "mAP50–95 (%)", "Mean (ms)", "P50 (ms)", "P95 (ms)", "FPS")
    rows = []
    for report in reports:
        status_class, status_label = _status(report, stage)
        complete = _evaluated(report, stage)
        metrics = report.get("validation", {}).get("mask", {}) if complete else {}
        benchmark = report.get("benchmark", {}) if complete else {}
        done = report.get("epochs_completed")
        requested = report.get("epochs_requested", 30 if stage == "pilot30" else 100)
        epoch_text = f"{done}/{requested}" if _number(done) else f"—/{requested}"
        values = [
            f'<span class="model-dot" style="background:{COLORS[report["model"]]}"></span>{_safe(_model_label(report["model"]))}',
            f'<span class="status {status_class}">{_safe(status_label)}</span>',
            _safe(epoch_text), _fmt(report.get("batch"), 0),
            _fmt(report.get("train_wall_seconds") if complete else None, 2, 1 / 3600),
            *[_fmt(metrics.get(key), 2, 100) for key in METRICS],
            *[_fmt(benchmark.get(key)) for key in ("mean_ms", "median_ms", "p95_ms", "fps")],
        ]
        rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
    return '<div class="table-scroll"><table><thead><tr>' + "".join(f"<th>{column}</th>" for column in columns) + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"


def _selection_cards(selection: dict, reports: list[dict], stage: str) -> str:
    if not selection.get("complete"):
        finished = sum(_evaluated(report, stage) for report in reports)
        reason = "Chờ cả ba lượt hoàn tất để áp dụng cùng một tiêu chí." if finished < 3 else "Chưa bảo đảm cùng ngân sách, dữ liệu và tập ảnh benchmark. Cần kiểm tra cấu hình trước khi chọn mô hình."
        return f'<div class="selection pending-selection"><strong>Chưa đề xuất mô hình</strong><span>{finished}/3 mô hình có đủ kết quả. {reason}</span></div>'
    lookup = {r["model"]: r for r in reports}
    cards = []
    specs = (
        ("Đề xuất cân bằng", "recommended", "recommended"),
        ("mAP50–95 cao nhất", "best_accuracy", ""),
        ("Suy luận nhanh nhất", "fastest_inference", ""),
    )
    for title, key, style in specs:
        model = selection.get(key)
        report = lookup.get(model, {})
        detail = f"{_fmt(_metric(report, 'map50_95'), 2, 100)}% mAP50–95 · {_fmt(report.get('benchmark', {}).get('mean_ms'))} ms/ảnh"
        cards.append(f'<div class="choice {style}"><span>{title}</span><strong>{_safe(_model_label(model))}</strong><small>{detail}</small></div>')
    return '<div class="choices">' + "".join(cards) + "</div>"


def _details(reports: list[dict], output: Path) -> str:
    parts = ['<details><summary>Nguồn kết quả, cấu hình và lỗi</summary><div class="source-grid">']
    for report in reports:
        path = Path(report["_source"])
        label = _model_label(report["model"])
        source = _source_link(path, output, "report.json") if report["_source_exists"] else "report.json chưa được tạo"
        local = {key: value for key, value in report.items() if not key.startswith("_")}
        parts.append(f'<div class="source"><strong>{_safe(label)}</strong><p>{source}</p><pre>{_safe(json.dumps(local, ensure_ascii=False, indent=2, default=str))}</pre></div>')
    parts.append("</div></details>")
    return "".join(parts)


CSS = """
:root{color-scheme:light;--ink:#183b42;--muted:#5d7076;--line:#dce5e6;--bg:#f2f6f6;--teal:#087f8c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 'Segoe UI',Arial,sans-serif}
header{background:#143c43;color:#fff;padding:44px max(28px,calc((100vw - 1400px)/2));border-bottom:6px solid #39b3b0}
.eyebrow{text-transform:uppercase;letter-spacing:.15em;font-size:12px;color:#a4dad8;font-weight:700}
h1{margin:10px 0 10px;font-size:clamp(27px,3vw,39px);line-height:1.2;font-weight:700}header p{max-width:880px;margin:0;color:#d5e7e8}
main{max-width:1456px;margin:auto;padding:28px}h2{font-size:25px;margin:0 0 4px}h3{font-size:16px;margin:0 0 14px}p{margin:8px 0}
.overview{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:22px}.kpi{background:white;border:1px solid var(--line);border-radius:10px;padding:18px 22px}
.kpi span{display:block;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}.kpi strong{display:block;font-size:27px;margin-top:3px}.kpi small{color:var(--muted)}
.notice{border-left:4px solid #c68d38;background:#fff8e9;border-radius:4px;padding:15px 20px;margin:0 0 22px}.notice.complete{background:#e4f5ee;border-color:#2a8566}
.protocol{background:#fff;border:1px solid var(--line);border-radius:10px;padding:22px 26px;margin-bottom:24px}.protocol p{max-width:1180px}.muted{color:var(--muted)}
.dataset{background:white;border:1px solid var(--line);border-radius:12px;padding:26px;margin-bottom:24px}.dataset-heading{display:flex;justify-content:space-between;gap:20px;align-items:center;margin-bottom:20px}.dataset-heading p{margin:0;color:var(--muted)}
.badge{background:#edf5f5;color:#236671;border-radius:30px;white-space:nowrap;padding:7px 13px;font-size:13px;font-weight:600}.choices{display:grid;grid-template-columns:1fr 1fr 1fr;gap:13px;margin:20px 0}.choice{border:1px solid var(--line);border-radius:8px;padding:17px 18px}.choice span,.choice small{display:block;color:var(--muted);font-size:12px}.choice strong{display:block;font-size:23px;margin:5px 0}.choice.recommended{background:#eaf7f5;border-color:#74bcb4}.choice.recommended span{color:#08776f;font-weight:700}
.selection{padding:17px 20px;border-radius:8px;background:#f4f6f6;margin:17px 0}.selection strong,.selection span{display:block}.selection span{font-size:13px;color:var(--muted);margin-top:3px}
.table-scroll{overflow:auto;border:1px solid var(--line);border-radius:8px;margin-top:18px}table{width:100%;border-collapse:collapse;white-space:nowrap;font-size:12px;font-variant-numeric:tabular-nums}th,td{padding:13px 10px;text-align:right;border-bottom:1px solid #e8eeee}th{background:#edf3f3;font-weight:600;color:#46616a}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}tr:last-child td{border:0}td:first-child{font-weight:650;color:#183b42}tbody tr:hover{background:#f7fafa}.model-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px}
.status{font-size:11px;display:inline-block;border-radius:20px;padding:3px 7px}.status.complete{background:#e3f4ed;color:#24664e}.status.pending{background:#eff1f3;color:#657279}.status.running{background:#e6f0ff;color:#305d92}.status.warning{background:#fff0d7;color:#9d5d19}
.table-note{font-size:12px;color:var(--muted);margin-top:9px}.charts{display:grid;grid-template-columns:1fr 1fr;gap:25px;margin-top:25px}.chart{border:1px solid #e5ecec;border-radius:8px;padding:17px 15px 5px}.chart h3{padding-left:8px;font-size:14px}svg{display:block;width:100%;height:auto}svg text{font-family:'Segoe UI',Arial,sans-serif}.grid{stroke:#e6eded;stroke-width:1}.axis{stroke:#a3b6ba}.tick{fill:#667d83;font-size:11px}.axis-title{fill:#566e76;font-size:11px}.point-label{font-size:10px;font-weight:650;paint-order:stroke;stroke:white;stroke-width:3px}.bar-label{fill:#3d5961;font-size:12px}.empty-chart{min-height:248px;display:flex;align-items:center;justify-content:center;padding:30px;text-align:center;color:#788b90;font-size:13px;background:linear-gradient(0deg,#f9fbfb,#fff)}
details{margin-top:20px;border-top:1px solid var(--line);padding-top:14px}summary{cursor:pointer;color:#2b6975;font-size:13px;font-weight:600}.source-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:15px;margin-top:14px}.source{min-width:0}.source p{font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f7;border:1px solid #e6ecee;border-radius:6px;font:11px/1.5 Consolas,monospace;padding:12px;max-height:350px;overflow:auto}a{color:#087d90}footer{font-size:12px;color:var(--muted);padding:4px 3px 30px}.method-list{padding-left:20px;margin-bottom:0}.method-list li{margin:5px 0}
@media(max-width:900px){header{padding:30px 20px}main{padding:18px}.dataset{padding:18px}.charts,.source-grid{grid-template-columns:1fr}.choices{grid-template-columns:1fr}.choice strong{font-size:22px}.overview{gap:9px}.kpi{padding:14px}.kpi strong{font-size:22px}.dataset-heading{align-items:flex-start}.chart h3{margin-bottom:8px}}
@media(max-width:550px){.overview{grid-template-columns:1fr}.dataset-heading{display:block}.badge{display:inline-block;margin-top:12px}.protocol{padding:18px}.table-scroll{margin-right:-6px}}
@media print{body{background:white;font-size:11px}header{padding:22px;color:#143c43;background:white;border-bottom:3px solid #087f8c}header p,.eyebrow{color:#436d73}main{padding:15px;max-width:none}h1{font-size:26px}.dataset{break-inside:avoid;padding:15px;page-break-before:auto}.overview{margin-bottom:15px}.charts{gap:10px}details{display:none}.table-scroll{overflow:visible}table{font-size:8px}th,td{padding:7px 4px}.choice strong{font-size:18px}.protocol{padding:15px}.empty-chart{min-height:100px}footer{padding-bottom:0}}
"""


def build_comparison(root: str | Path | None = None, stage: str = "pilot30") -> dict:
    """Write one standalone HTML file and return its truthful completion status.

    Scores are percentages for display, but the runtime's [0, 1] metrics are
    preserved in the returned results and embedded source records.
    """
    from .selection import select_models

    if stage not in {"pilot30", "final100"}:
        raise ValueError("stage phải là pilot30 hoặc final100")
    project = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    output = project / "reports" / f"seg_{stage}_evaluation.html"
    all_reports = []
    datasets = {}
    panels = []
    for dataset, (title, description) in DATASETS.items():
        reports = []
        for model in MODELS:
            path = project / "runs" / "seg" / stage / dataset / model.replace("-", "_") / "report.json"
            reports.append(_read_report(path, dataset, model, stage))
        valid = [report for report in reports if _evaluated(report, stage)]
        selection = select_models(valid, tolerance=0.02, expected_models=MODELS)
        # A defensive guard prevents a changed selection helper from suggesting
        # a winner while one requested run still lacks a valid benchmark.
        if len(valid) != len(MODELS):
            selection = {**selection, "complete": False, "recommended": None, "best_accuracy": None, "fastest_inference": None}
        datasets[dataset] = {"results": reports, "selection": selection, "completed": len(valid), "expected": len(MODELS)}
        all_reports.extend(reports)
        frontier = selection.get("pareto_frontier", [])
        frontier_note = f'<p class="table-note">Biên Pareto (mAP50–95 cao, độ trễ và thời gian train thấp): {_safe(", ".join(_model_label(model) for model in frontier))}.</p>' if selection.get("complete") and frontier else ""
        panels.append(f'''<section class="dataset" id="{dataset}">
<div class="dataset-heading"><div><h2>{title}</h2><p>{description}</p></div><span class="badge">{len(valid)}/3 mô hình đã đánh giá</span></div>
{_selection_cards(selection, reports, stage)}{_table(reports, stage)}
<p class="table-note">P/R và mAP là chỉ số <strong>mask</strong> trên validation. Mean/P50/P95 là độ trễ mỗi ảnh; FPS = 1000 / Mean. Dấu “—” là chưa có kết quả đủ điều kiện, không phải điểm 0.</p>
{frontier_note}<div class="charts"><div class="chart"><h3>Chất lượng và tốc độ suy luận</h3>{_scatter(reports, stage)}</div><div class="chart"><h3>Chi phí huấn luyện</h3>{_duration_chart(reports, stage)}</div></div>
{_details(reports, output)}</section>''')
    count = sum(_evaluated(report, stage) for report in all_reports)
    expected = len(DATASETS) * len(MODELS)
    complete = count == expected and all(dataset["selection"].get("complete") for dataset in datasets.values())
    generated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    pilot = stage == "pilot30"
    epochs = 30 if pilot else 100
    notice_title = "Đã đủ kết quả để so sánh" if complete else "Báo cáo đang cập nhật — chưa hoàn tất thí nghiệm"
    notice_body = f"{count}/{expected} lượt có đủ checkpoint đánh giá và số đo thời gian. "
    notice_body += "Mỗi bộ dữ liệu chỉ có đề xuất sau khi cả ba mô hình hoàn tất. "
    notice_body += "Đề xuất 30 epoch là kết quả sơ bộ, cần xác nhận lại sau giai đoạn 100 epoch." if pilot else "Đây là so sánh trên validation; kết quả test cuối cùng cần được báo cáo riêng."
    subtitle = "So sánh YOLOv8n-seg, YOLOv8s-seg và YOLOv8m-seg trên hai bài toán SEG; cùng số epoch và quy trình đánh giá."
    html = f'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="description" content="Đánh giá mô hình phân đoạn hư hỏng mặt đường: chất lượng mask, thời gian huấn luyện và tốc độ suy luận."><title>Đánh giá SEG · {epochs} epoch</title><style>{CSS}</style></head>
<body><header><div class="eyebrow">Road Damage · Model Evaluation</div><h1>Đánh giá mô hình SEG · {epochs} epoch</h1><p>{subtitle}</p></header>
<main><div class="overview"><div class="kpi"><span>Tiến độ đánh giá</span><strong>{count} / {expected} mô hình</strong><small>Hai bộ dữ liệu · ba kích thước YOLOv8</small></div><div class="kpi"><span>Ngân sách mỗi lượt</span><strong>{epochs} epoch</strong><small>{"Thử nghiệm ban đầu từ pretrained" if pilot else "Giai đoạn huấn luyện chính"}</small></div><div class="kpi"><span>Ngưỡng chọn cân bằng</span><strong>≤ 2 điểm %</strong><small>Chênh mask mAP50–95 so với tốt nhất</small></div></div>
<div class="notice {"complete" if complete else ""}"><strong>{notice_title}</strong><p>{notice_body}</p></div>
<section class="protocol"><h3>Chọn mô hình có chất lượng cao và thời gian tối ưu</h3><p>So sánh <strong>riêng từng bộ dữ liệu</strong>. Lấy mask mAP50–95 tốt nhất làm mốc, giữ các mô hình thấp hơn mốc tối đa <strong>0,02 (2 điểm phần trăm)</strong>, rồi chọn mô hình có <strong>độ trễ suy luận trung bình thấp nhất</strong>. Nếu độ trễ bằng nhau, ưu tiên thời gian train ngắn hơn, rồi mAP50–95 cao hơn.</p><p class="muted">Báo cáo thể hiện cả chi phí train và tốc độ xử lý thực tế. Mô hình đạt mAP cao nhất và mô hình suy luận nhanh nhất được nêu riêng để thấy đánh đổi. Không gộp điểm giữa crack và pothole vì đây là hai tập dữ liệu khác nhau.</p></section>
{"".join(panels)}
<section class="protocol"><h3>Quy ước đo và giới hạn kết luận</h3><ul class="method-list"><li><strong>Chất lượng:</strong> đánh giá lại best.pt trên split validation; P = Precision, R = Recall, mAP50 tại IoU 0,50 và mAP50–95 trung bình IoU 0,50–0,95. Các chỉ số trên bảng được hiển thị theo phần trăm. Box metrics gốc, nếu có, nằm trong phần nguồn kết quả.</li><li><strong>Thời gian train:</strong> thời gian thực của phần huấn luyện, gồm validation trong vòng train; đánh giá cuối và benchmark được tính riêng. Batch thực tế xuất hiện trong bảng để nhận biết khác biệt cấu hình.</li><li><strong>Độ trễ:</strong> batch 1, có warmup và đồng bộ GPU; đo lời gọi dự đoán gồm tiền xử lý, mô hình và hậu xử lý, loại thời gian đọc ảnh từ đĩa. P50 là trung vị; P95 mô tả độ trễ ở phân vị 95. Số ảnh, số lần lặp, phần cứng và phiên bản nằm trong dữ liệu nguồn của từng lượt.</li><li><strong>Chọn mô hình:</strong> chỉ dùng validation để lựa chọn; giữ test cho đánh giá cuối giai đoạn 100 epoch. Nhãn gốc của mỗi bộ dữ liệu được giữ nguyên; bảng so sánh là chỉ số mask tổng hợp theo các lớp đã huấn luyện.</li><li><strong>Diễn giải:</strong> {"30 epoch là vòng sàng lọc; thứ hạng có thể đổi khi huấn luyện lâu hơn. " if pilot else ""}Kết quả phân đoạn chưa chứng minh độ chính xác của phép đo chiều dài hoặc diện tích; đo mét/m² cần hiệu chuẩn riêng. Benchmark được so sánh trên cùng phần cứng và điều kiện tải.</li></ul></section>
<footer>Tạo lúc {_safe(generated)} · Stage: {_safe(stage)} · Tệp HTML độc lập, không dùng CDN. Đây là ảnh chụp trạng thái lúc tạo; chạy lại lệnh xuất báo cáo để cập nhật tiến độ. Nguồn JSON được nhúng trong mỗi bộ dữ liệu và liên kết tới file cục bộ tương ứng.</footer></main></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".html.tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(output)
    return {"html_path": str(output), "status": "completed" if complete else "incomplete", "stage": stage, "completed": count, "expected": expected, "generated_at": generated, "results": all_reports, "datasets": datasets}


def main() -> None:
    parser = argparse.ArgumentParser(description="Xuất một file HTML so sánh chất lượng và thời gian mô hình SEG")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--stage", choices=("pilot30", "final100"), default="pilot30")
    args = parser.parse_args()
    result = build_comparison(args.root, args.stage)
    print(json.dumps({key: result[key] for key in ("html_path", "status", "completed", "expected")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
