"""Wait for the authorized pilot queue, then verify and finalize its report."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1", YOLO_CONFIG_DIR=str(ROOT / "reports/runtime/ultralytics"))

from road_damage.runner import process_alive, gpu_lease, refresh_outputs
from road_damage.training import atomic_json, read_json, utc_now


def main():
    queue = read_json(ROOT / "reports/runtime/queue_process.json", {})
    status_path = ROOT / "reports/runtime/finalization.json"
    atomic_json(status_path, {"status": "waiting_for_queue", "pid": os.getpid(), "started_at": utc_now(), "queue_pid": queue.get("pid")})
    while process_alive(queue.get("pid"), queue.get("process_created")):
        time.sleep(30)
    lease_path = ROOT / "runs/seg/.gpu.lock"
    # A detached training worker may outlive an interrupted coordinator.
    while True:
        lease = read_json(lease_path, {})
        if not process_alive(lease.get("worker_pid"), lease.get("worker_created")):
            break
        time.sleep(30)
    status = refresh_outputs(ROOT, "pilot30")
    if status["completed"] != 6:
        atomic_json(status_path, {"status": "retrying_incomplete_runs_once", "updated_at": utc_now()})
        with (ROOT / "reports/runtime/retry.stdout.log").open("a", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-s", "-u", "-m", "road_damage.runner", "--stage", "pilot30"],
                           cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=False)
        status = refresh_outputs(ROOT, "pilot30")
    if status["completed"] == 6:
        from road_damage.evaluation import benchmark_manifest
        from road_damage.examples import generate_examples
        with gpu_lease(ROOT):
            for dataset in ("crack", "pothole"):
                for size in ("n", "s", "m"):
                    run = ROOT / f"runs/seg/pilot30/{dataset}/yolov8{size}_seg"
                    report = read_json(run / "report.json")
                    if not (run / "examples/measurements.json").exists():
                        manifest = benchmark_manifest(ROOT / f"configs/seg/{dataset}.yaml", run.parent / "benchmark_manifest.json")
                        report["examples"] = generate_examples(run / "weights/best.pt", manifest, run / "examples", dataset)
                        atomic_json(run / "report.json", report)
    status = refresh_outputs(ROOT, "pilot30")
    from road_damage.reporting import build_comparison
    comparison = build_comparison(ROOT, "pilot30")
    # Archive runtime files produced by the initial Ultralytics setup/AMP check.
    for relative, destination_relative in (("Ultralytics", "reports/runtime/initial_ultralytics"),
                                            ("yolo26n.pt", "reports/runtime/amp/yolo26n.pt")):
        source, destination = (ROOT / relative).resolve(), (ROOT / destination_relative).resolve()
        if not source.is_relative_to(ROOT.resolve()) or not destination.is_relative_to(ROOT.resolve()):
            raise RuntimeError("Runtime archive paths must remain inside the project")
        if source.exists() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
    atomic_json(status_path, {"status": comparison["status"], "completed": comparison["completed"],
                             "expected": 6, "finished_at": utc_now(), "report": comparison["html_path"],
                             "main100_started": False})
    print(json.dumps({"status": comparison["status"], "completed": comparison["completed"], "report": comparison["html_path"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        atomic_json(ROOT / "reports/runtime/finalization.json", {"status": "failed", "updated_at": utc_now(), "error": f"{type(error).__name__}: {error}"})
        raise
