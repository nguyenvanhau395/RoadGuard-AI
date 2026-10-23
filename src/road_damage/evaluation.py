"""Validation metrics and a reproducible batch-one SEG latency benchmark."""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _validation_paths(data_yaml):
    import yaml

    data_yaml = Path(data_yaml)
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = Path(config.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    entries = config["val"]
    if isinstance(entries, str):
        entries = [entries]
    paths = []
    for entry in entries:
        source = Path(entry)
        if not source.is_absolute():
            source = base / source
        if source.suffix.lower() != ".txt":
            raise ValueError("Expected audited validation image manifest (.txt)")
        for line in source.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                candidate = Path(line.strip())
                paths.append(candidate if candidate.is_absolute() else source.parent / candidate)
    return sorted(set(p.resolve() for p in paths), key=lambda p: str(p).lower())


def benchmark_manifest(data_yaml, output_path, count=200, seed=42):
    """Persist one sample per dataset and refuse silently changed samples."""
    paths = _validation_paths(data_yaml)
    if not paths:
        raise ValueError("Validation manifest is empty")
    indices = sorted(random.Random(seed).sample(range(len(paths)), min(count, len(paths))))
    selected = [str(paths[index]) for index in indices]
    digest = hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest()
    value = {"seed": seed, "requested_images": count, "images": selected, "sha256": digest}
    output_path = Path(output_path)
    if output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous != value:
            raise RuntimeError(f"Benchmark sample changed: {output_path}. Use a new experiment stage.")
    else:
        _write_json(output_path, value)
    return value


def evaluate_checkpoint(checkpoint, data_yaml, output_dir, split="val"):
    """Use validation for pilot selection; the caller decides when test is legal."""
    from ultralytics import YOLO

    model = YOLO(str(checkpoint), task="segment")
    if model.task != "segment":
        raise ValueError("The checkpoint is not an instance segmentation model")
    args = dict(data=str(data_yaml), split=split, imgsz=640, batch=1, device=0,
                workers=0, half=True, rect=False, conf=0.001, iou=0.7,
                max_det=300, augment=False, plots=True, save_json=False,
                project=str(Path(output_dir).parent), name=Path(output_dir).name,
                exist_ok=True, verbose=False)
    metrics = model.val(**args)

    def values(metric):
        result = {"precision": float(metric.mp), "recall": float(metric.mr),
                  "map50": float(metric.map50), "map50_95": float(metric.map)}
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in result.values()):
            raise ValueError(f"Invalid evaluation metrics: {result}")
        return result

    class_names = {str(k): v for k, v in model.names.items()}
    per_class = []
    for index, class_id in enumerate(metrics.seg.ap_class_index):
        p, r, ap50, ap = metrics.seg.class_result(index)
        per_class.append({"class_id": int(class_id), "class_name": class_names[str(int(class_id))],
                          "precision": float(p), "recall": float(r),
                          "map50": float(ap50), "map50_95": float(ap)})
    return {"split": split, "checkpoint": str(Path(checkpoint).resolve()),
            "mask": values(metrics.seg), "box": values(metrics.box),
            "class_names": class_names, "mask_per_class": per_class,
            "args": args, "precision_recall_rule": "Ultralytics mean metrics at the validation F1-optimal confidence",
            "checkpoint_selection": "highest validation mask mAP50-95 during training"}


def benchmark_checkpoint(checkpoint, manifest, output_dir, warmup=10, repeats=3):
    """Time warmed predict calls on predecoded images, including mask generation.

    Batch 1, GPU synchronized, FP16, fixed square input, original-size masks.
    Disk I/O, overlays, skeleton/area measurement and report writing are excluded.
    """
    import cv2
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for comparable pilot benchmarks")
    images = []
    for source in manifest["images"]:
        image = cv2.imread(source)
        if image is None:
            raise ValueError(f"Cannot decode benchmark image: {source}")
        images.append(image)
    model = YOLO(str(checkpoint), task="segment")
    args = dict(imgsz=640, batch=1, device=0, half=True, rect=False,
                conf=0.25, iou=0.7, max_det=300, retina_masks=True,
                augment=False, verbose=False, save=False, show=False,
                stream=False)

    def predict(image):
        predictions = model.predict(source=image, **args)
        result = predictions[0]
        if result.masks is not None and tuple(result.masks.data.shape[-2:]) != tuple(result.orig_shape):
            raise AssertionError("Prediction mask does not match original image dimensions")
        return result

    for index in range(warmup):
        predict(images[index % len(images)])
    torch.cuda.synchronize(0)
    timings = []
    forward = []
    samples = []
    for repeat in range(repeats):
        for image_index, image in enumerate(images):
            torch.cuda.synchronize(0)
            start = time.perf_counter()
            result = predict(image)
            torch.cuda.synchronize(0)
            elapsed_ms = (time.perf_counter() - start) * 1000
            timings.append(elapsed_ms)
            inference_ms = float(result.speed["inference"])
            forward.append(inference_ms)
            samples.append({"repeat": repeat + 1, "image_index": image_index,
                            "end_to_end_ms": elapsed_ms, "forward_ms": inference_ms})
    timings_sorted = sorted(timings)
    position = .95 * (len(timings_sorted) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    p95 = timings_sorted[lower] + (timings_sorted[upper] - timings_sorted[lower]) * (position - lower)
    mean = statistics.mean(timings)
    result = {"mean_ms": mean, "median_ms": statistics.median(timings), "p95_ms": p95,
              "stdev_ms": statistics.pstdev(timings), "fps": 1000 / mean,
              "forward_mean_ms": statistics.mean(forward), "images": len(images),
              "repeats": repeats, "warmup": warmup, "timed_predictions": len(timings),
              "manifest_sha256": manifest["sha256"], "args": args,
              "scope": "end_to_end_predict_excluding_disk",
              "excluded": ["disk_decode", "overlay", "length_area_measurement", "report_writing"],
              "gpu": torch.cuda.get_device_name(0)}
    output_dir = Path(output_dir)
    _write_json(output_dir / "samples.json", samples)
    _write_json(output_dir / "summary.json", result)
    del model, images
    torch.cuda.empty_cache()
    return result
