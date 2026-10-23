"""Sequential, resumable SEG experiment queue with one project GPU lease."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .training import (DATASETS, MODELS, STAGES, atomic_json, canonical_model,
                       dataset_identity, experiment_identity, project_root as resolve_root,
                       read_json, run_directory, training_config, utc_now)


def process_alive(pid, created=None):
    import psutil
    try:
        process = psutil.Process(int(pid))
        return process.is_running() and (created is None or abs(process.create_time() - created) < 1)
    except (psutil.Error, TypeError, ValueError):
        return False


@contextmanager
def gpu_lease(root):
    import psutil
    path = root / "runs" / "seg" / ".gpu.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}:{time.time_ns()}"
    lease = {"token": token, "pid": os.getpid(), "created": psutil.Process().create_time(), "started_at": utc_now()}
    for _ in range(3):
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(lease, handle)
            break
        except FileExistsError:
            try:
                previous = read_json(path, {})
            except (OSError, ValueError):
                raise RuntimeError(f"GPU lock unreadable: {path}; inspect it before retrying")
            if process_alive(previous.get("pid"), previous.get("created")) or process_alive(previous.get("worker_pid"), previous.get("worker_created")):
                raise RuntimeError("Another project SEG process is using the GPU; run only one notebook/queue at a time")
            path.unlink(missing_ok=True)
    else:
        raise RuntimeError("Could not acquire project GPU lease")
    try:
        yield path, lease
    finally:
        if read_json(path, {}).get("token") == token and not process_alive(lease.get("worker_pid"), lease.get("worker_created")):
            path.unlink(missing_ok=True)


def refresh_outputs(root, stage):
    reports = []
    for dataset in DATASETS:
        for model in MODELS:
            report_path = run_directory(root, dataset, model, stage) / "report.json"
            report = read_json(report_path)
            reports.append(report or {"dataset": dataset, "model": f"yolov8{model}-seg", "stage": stage, "status": "pending", "epochs_completed": 0})
    status = {"stage": stage, "updated_at": utc_now(), "completed": sum(r.get("status") == "completed" for r in reports), "requested_runs": 6, "runs": reports}
    atomic_json(root / "runs" / "seg" / stage / "status.json", status)
    try:
        from .reporting import build_comparison
        build_comparison(root=root, stage=stage)
        from .worklog import refresh_readme
        refresh_readme(root)
    except Exception as error:
        # Reporting errors are visible and do not discard an hours-long training result.
        atomic_json(root / "reports" / "report_generation_error.json", {"time": utc_now(), "type": type(error).__name__, "message": str(error)})
        print(f"Report update failed: {type(error).__name__}: {error}", flush=True)
    return status


def run_experiment(dataset, model, stage="pilot30", project_root=None):
    root = resolve_root(project_root)
    model = canonical_model(model)
    run_dir = run_directory(root, dataset, model, stage)
    report_path = run_dir / "report.json"
    with gpu_lease(root) as (lease_path, lease):
        yaml_path = root / "configs" / "seg" / f"{dataset}.yaml"
        if not yaml_path.exists():
            from .dataset import prepare_dataset
            try:
                prepare_dataset(dataset, root=root)
            except FileNotFoundError:
                pending = {"dataset": dataset, "model": f"yolov8{model}-seg", "stage": stage,
                           "status": "pending_data", "epochs_requested": STAGES[stage]["epochs"],
                           "epochs_completed": 0, "updated_at": utc_now()}
                atomic_json(report_path, pending)
                refresh_outputs(root, stage)
                return pending
        config = training_config(root, dataset, model, stage)
        identity = dataset_identity(yaml_path)
        fingerprint, provenance = experiment_identity(config, identity)
        report = read_json(report_path)
        if report and report.get("fingerprint") and report["fingerprint"] != fingerprint:
            raise RuntimeError(f"Config/dataset/pretrained weights changed for {run_dir}; existing run will not be overwritten")
        if report and report.get("status") == "completed":
            if not (run_dir / "weights/best.pt").is_file() or not (run_dir / "weights/last.pt").is_file():
                raise RuntimeError("Completed run is missing a checkpoint")
            refresh_outputs(root, stage)
            print(f"Already completed: {dataset}/{model}/{stage}", flush=True)
            return report
        if not report or not report.get("fingerprint"):
            report = {"schema_version": 1, "dataset": dataset, "model": f"yolov8{model}-seg", "stage": stage,
                      "status": "queued", "epochs_requested": STAGES[stage]["epochs"], "epochs_completed": 0,
                      "created_at": utc_now(), "updated_at": utc_now(), "train_wall_seconds": 0,
                      "fingerprint": fingerprint, "dataset_fingerprint": identity["sha256"],
                      "dataset_identity": identity, "provenance": provenance,
                      "config": config, "batch": config["batch"], "attempts": [], "errors": []}
            atomic_json(report_path, report)
        batch = int(report.get("batch", config["batch"]))
        while True:
            run_dir.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            (root / "reports/runtime/ultralytics").mkdir(parents=True, exist_ok=True)
            environment.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1", PYTHONUNBUFFERED="1",
                               YOLO_CONFIG_DIR=str(root / "reports/runtime/ultralytics"))
            command = [sys.executable, "-s", "-u", "-m", "road_damage.runner", "--root", str(root),
                       "--stage", stage, "--worker", "--dataset", dataset, "--model", model, "--batch", str(batch)]
            print(f"Starting {dataset}/{model}: {STAGES[stage]['epochs']} epochs, batch {batch}. Log: {run_dir / 'console.log'}", flush=True)
            with (run_dir / "console.log").open("a", encoding="utf-8") as log:
                log.write(f"\nWorker started {utc_now()}, batch={batch}\n")
                log.flush()
                # Ultralytics AMP checks resolve auxiliary pretrained models from cwd.
                process = subprocess.Popen(command, cwd=root / "models/pretrained", env=environment, stdout=log, stderr=subprocess.STDOUT)
                import psutil
                lease.update(worker_pid=process.pid, worker_created=psutil.Process(process.pid).create_time())
                atomic_json(lease_path, lease)
                try:
                    while True:
                        try:
                            code = process.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            refresh_outputs(root, stage)
                            current = read_json(report_path, {})
                            print(f"{dataset}/{model}: {current.get('phase','starting')}, epoch {current.get('epochs_completed',0)}/{STAGES[stage]['epochs']}", flush=True)
                except KeyboardInterrupt:
                    # Retain the lease while the worker is alive. The worker continues
                    # saving checkpoints and can be resumed safely if later stopped.
                    print(f"Queue interrupted; worker PID {process.pid} is still active. See its console.log.", flush=True)
                    raise
            lease.pop("worker_pid", None)
            lease.pop("worker_created", None)
            atomic_json(lease_path, lease)
            report = read_json(report_path, report)
            refresh_outputs(root, stage)
            if code == 0 and report.get("status") == "completed":
                return report
            if code == 42 and batch > 1:
                batch //= 2
                report.update(batch=batch, status="queued", updated_at=utc_now())
                report.setdefault("batch_changes", []).append({"time": utc_now(), "new_batch": batch, "reason": "CUDA OOM"})
                atomic_json(report_path, report)
                continue
            if report.get("status") in ("running", "queued"):
                report.update(status="interrupted", updated_at=utc_now(), worker_exit_code=code)
                atomic_json(report_path, report)
                refresh_outputs(root, stage)
            return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--stage", choices=STAGES, default="pilot30")
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=list(DATASETS))
    parser.add_argument("--models", choices=MODELS, nargs="+", default=list(MODELS))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=DATASETS, help=argparse.SUPPRESS)
    parser.add_argument("--model", choices=MODELS, help=argparse.SUPPRESS)
    parser.add_argument("--batch", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = resolve_root(args.root)
    if args.worker:
        from .training import train_and_evaluate_worker
        return train_and_evaluate_worker(root, args.dataset, args.model, args.stage, args.batch)
    incomplete = False
    for dataset in args.datasets:
        for model in args.models:
            result = run_experiment(dataset, model, args.stage, root)
            incomplete = incomplete or result.get("status") != "completed"
    refresh_outputs(root, args.stage)
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
