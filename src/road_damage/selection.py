"""Selection rules for completed, comparable segmentation pilot runs."""
from __future__ import annotations

import math


DEFAULT_MODELS = ("yolov8n-seg", "yolov8s-seg", "yolov8m-seg")


def _numbers(report):
    return (
        float(report["validation"]["mask"]["map50_95"]),
        float(report["benchmark"]["mean_ms"]),
        float(report["train_wall_seconds"]),
    )


def select_models(reports, tolerance=0.02, expected_models=DEFAULT_MODELS):
    """Recommend lowest latency within an absolute mAP tolerance, per dataset.

    No recommendation is issued until every expected model has completed the
    requested epoch budget and has finite, positive timing measurements.
    """
    if not 0 <= tolerance <= 1:
        raise ValueError("tolerance must be an absolute fraction between 0 and 1")
    records = list(reports)
    if len({r.get("dataset") for r in records}) > 1:
        raise ValueError("Compare models within one dataset, never across datasets")
    valid = {}
    for report in records:
        if report.get("status") != "completed":
            continue
        try:
            accuracy, latency, training = _numbers(report)
            if not all(math.isfinite(v) for v in (accuracy, latency, training)):
                continue
            if not 0 <= accuracy <= 1 or min(latency, training) <= 0:
                continue
            if report.get("epochs_completed") != report.get("epochs_requested"):
                continue
            if report["model"] in valid:
                raise ValueError(f"Duplicate model report: {report['model']}")
            valid[report["model"]] = report
        except (KeyError, TypeError):
            continue
    missing = sorted(set(expected_models) - set(valid))
    result = {
        "complete": not missing,
        "best_accuracy": None,
        "fastest_inference": None,
        "recommended": None,
        "pareto_frontier": [],
        "tolerance": tolerance,
        "missing_models": missing,
        "reason": "Awaiting all requested models; incomplete comparisons are not ranked.",
    }
    if missing:
        return result
    candidates = [valid[name] for name in expected_models]
    signatures = {
        (r.get("stage"), r.get("epochs_requested"), r.get("dataset_fingerprint"),
         r.get("benchmark", {}).get("manifest_sha256"))
        for r in candidates
    }
    if len(signatures) > 1:
        result.update(complete=False, reason="Runs use different stage, budget, dataset, or benchmark samples.")
        return result
    best = min(candidates, key=lambda r: (-_numbers(r)[0], _numbers(r)[1], r["model"]))
    fastest = min(candidates, key=lambda r: (_numbers(r)[1], _numbers(r)[2], -_numbers(r)[0], r["model"]))
    eligible = [r for r in candidates if _numbers(r)[0] >= _numbers(best)[0] - tolerance - 1e-12]
    recommended = min(eligible, key=lambda r: (_numbers(r)[1], _numbers(r)[2], -_numbers(r)[0], r["model"]))
    frontier = []
    for candidate in candidates:
        accuracy, latency, training = _numbers(candidate)
        dominated = False
        for other in candidates:
            oa, ol, ot = _numbers(other)
            if oa >= accuracy and ol <= latency and ot <= training and (oa > accuracy or ol < latency or ot < training):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate["model"])
    result.update(
        best_accuracy=best["model"], fastest_inference=fastest["model"],
        recommended=recommended["model"], pareto_frontier=frontier,
        reason=(f"Heuristic: lowest batch-1 latency within {tolerance * 100:g} percentage points "
                "of best validation mask mAP50-95; ties use shorter training then higher mAP. "
                "This is not a universal optimum."),
    )
    return result
