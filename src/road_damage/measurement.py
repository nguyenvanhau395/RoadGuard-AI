"""Measure predicted masks in original-image pixels, with no metric calibration.

Use Ultralytics predict(..., retina_masks=True), then pass masks.data.cpu().numpy()
and result.orig_shape. Crack length is the length of a skeleton graph, not road
length. Measurements of different video frames must never be summed as unique
physical road damage without registration/tracking.
"""

from __future__ import annotations

import math

import numpy as np
from skimage.morphology import skeletonize


def skeleton_length(skeleton: np.ndarray) -> float:
    """Count each horizontal/vertical/diagonal skeleton edge exactly once.

    Diagonals have length sqrt(2); omit a diagonal if either intervening
    orthogonal pixel already provides a route around that corner. An isolated
    skeleton pixel has zero graph length.
    """
    pixels = np.asarray(skeleton, dtype=bool)
    if pixels.ndim != 2:
        raise ValueError("Skeleton must be a two-dimensional array")
    horizontal = np.count_nonzero(pixels[:, :-1] & pixels[:, 1:])
    vertical = np.count_nonzero(pixels[:-1, :] & pixels[1:, :])
    down_right = pixels[:-1, :-1] & pixels[1:, 1:] & ~pixels[:-1, 1:] & ~pixels[1:, :-1]
    down_left = pixels[:-1, 1:] & pixels[1:, :-1] & ~pixels[:-1, :-1] & ~pixels[1:, 1:]
    diagonal = np.count_nonzero(down_right) + np.count_nonzero(down_left)
    return float(horizontal + vertical + math.sqrt(2) * diagonal)


def measure_masks(
    masks: np.ndarray,
    *,
    task: str = "crack",
    class_ids: list[int] | np.ndarray | None = None,
    target_class_id: int | None = None,
    original_shape: tuple[int, int] | list[int] | None = None,
) -> dict:
    """Measure instances and their union; mask shape is N,H,W or H,W.

    Floating masks are thresholded at >0.5. If original_shape is provided,
    mismatched mask resolution raises instead of silently measuring resized
    pixels. Supply target_class_id for the original Pothole class when an export
    has multiple classes. Returned values are JSON serializable.
    """
    if task not in ("crack", "pothole"):
        raise ValueError("task must be 'crack' or 'pothole'")
    values = np.asarray(masks)
    if values.size == 0 and values.ndim < 3 and original_shape is not None:
        values = np.empty((0, *original_shape), dtype=bool)
    if values.ndim == 2:
        values = values[np.newaxis, ...]
    if values.ndim != 3 or values.shape[1] < 1 or values.shape[2] < 1:
        raise ValueError("Masks must have shape N,H,W with positive image dimensions")
    if not np.isfinite(values).all():
        raise ValueError("Masks contain non-finite values")
    shape = tuple(int(value) for value in values.shape[1:])
    if original_shape is not None and tuple(original_shape) != shape:
        raise ValueError(f"Mask shape {shape} differs from original image {tuple(original_shape)}")
    binary = values > 0.5
    classes = [None] * len(binary) if class_ids is None else list(class_ids)
    if len(classes) != len(binary):
        raise ValueError("class_ids must contain one class ID per instance")
    for class_id in classes:
        if class_id is not None and (not np.isfinite(class_id) or int(class_id) != class_id or class_id < 0):
            raise ValueError("Class IDs must be nonnegative integers")
    if target_class_id is not None and class_ids is None:
        raise ValueError("class_ids are required when filtering target_class_id")
    selected = [index for index, class_id in enumerate(classes) if target_class_id is None or class_id == target_class_id]
    union = np.zeros(shape, dtype=bool)
    instances = []
    for index in selected:
        mask = binary[index]
        union |= mask
        result = {"instance_index": index, "class_id": None if classes[index] is None else int(classes[index])}
        if task == "crack":
            result["length_px"] = skeleton_length(skeletonize(mask))
        else:
            result["area_px2"] = int(np.count_nonzero(mask))
        instances.append(result)
    summary = {"task": task, "unit": "px" if task == "crack" else "px2", "original_shape": list(shape), "instances": instances, "instance_count": len(instances)}
    if task == "crack":
        summary["total_length_px"] = skeleton_length(skeletonize(union))
    else:
        summary["total_area_px2"] = int(np.count_nonzero(union))
    return summary
