"""Save a small, common set of SEG overlays and original-pixel measurements."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .measurement import measure_masks


def generate_examples(checkpoint, manifest, output_dir, dataset, count=9):
    import cv2
    from ultralytics import YOLO
    from skimage.morphology import skeletonize

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(checkpoint), task="segment")
    target = None
    if dataset == "pothole":
        matches = [int(k) for k, v in model.names.items() if str(v).strip().casefold() == "pothole"]
        if len(matches) != 1:
            raise ValueError("Exactly one Pothole class is required for area measurement")
        target = matches[0]
    records = []
    for index, source in enumerate(manifest["images"][:count]):
        image = cv2.imread(source)
        if image is None:
            raise ValueError(f"Could not load example image: {source}")
        result = model.predict(image, imgsz=640, batch=1, device=0, half=True,
                               rect=False, retina_masks=True, conf=.25, iou=.7,
                               verbose=False, save=False)[0]
        masks = result.masks.data.cpu().numpy() if result.masks is not None else np.empty((0, *result.orig_shape), dtype=bool)
        classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else np.empty(0, dtype=int)
        measurement = measure_masks(masks, task=dataset, class_ids=classes,
                                    target_class_id=target, original_shape=result.orig_shape)
        overlay = result.plot()
        if dataset == "crack":
            union = (masks > .5).any(axis=0) if len(masks) else np.zeros(result.orig_shape, bool)
            overlay[skeletonize(union)] = (0, 0, 255)
            caption = f"Skeleton length: {measurement['total_length_px']:.1f} px"
        else:
            caption = f"Pothole area: {measurement['total_area_px2']} px^2"
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 32), (25, 35, 40), -1)
        cv2.putText(overlay, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        path = output / f"{index+1:02d}_{Path(source).stem}.jpg"
        if not cv2.imwrite(str(path), overlay):
            raise RuntimeError(f"Failed to write overlay: {path}")
        records.append({"source": source, "overlay": str(path), **measurement})
    summary = {"dataset": dataset, "checkpoint": str(checkpoint), "measurements": records,
               "conf": .25, "iou": .7, "units": "original-image pixels; no metric calibration"}
    path = output / "measurements.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"summary_path": str(path), "images": [r["overlay"] for r in records], "count": len(records)}
