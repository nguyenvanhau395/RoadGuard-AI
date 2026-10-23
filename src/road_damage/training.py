"""Isolated training worker, checkpoint provenance and resumable run state."""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


MODELS = ("n", "s", "m")
DATASETS = ("crack", "pothole")
STAGES = {"pilot30": {"epochs": 30, "patience": 0}, "main100": {"epochs": 100, "patience": 20}}


def project_root(root=None):
    return Path(root).resolve() if root else Path(__file__).resolve().parents[2]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default


def canonical_model(model):
    value = str(model).lower().removesuffix(".pt").replace("_", "-")
    for size in MODELS:
        if value in (size, f"yolov8{size}-seg"):
            return size
    raise ValueError("model must be n, s, m or a corresponding YOLOv8 SEG model name")


def run_directory(root, dataset, model, stage):
    if dataset not in DATASETS or stage not in STAGES:
        raise ValueError(f"Unknown dataset/stage: {dataset}/{stage}")
    return project_root(root) / "runs" / "seg" / stage / dataset / f"yolov8{canonical_model(model)}_seg"


def training_config(root, dataset, model, stage="pilot30", batch=None):
    root, size = project_root(root), canonical_model(model)
    run_dir = run_directory(root, dataset, size, stage)
    return {
        "task": "segment", "mode": "train",
        "model": str(root / "models" / "pretrained" / f"yolov8{size}-seg.pt"),
        "data": str(root / "configs" / "seg" / f"{dataset}.yaml"),
        "epochs": STAGES[stage]["epochs"], "patience": STAGES[stage]["patience"],
        "imgsz": 640, "batch": batch or {"n": 8, "s": 8, "m": 4}[size],
        "device": 0, "workers": 0, "amp": True, "cache": False,
        "seed": 42, "deterministic": True, "optimizer": "AdamW", "lr0": .001,
        "lrf": .01, "momentum": .937, "weight_decay": .0005,
        "warmup_epochs": 3.0, "warmup_momentum": .8, "warmup_bias_lr": .1,
        "box": 7.5, "cls": .5, "dfl": 1.5, "nbs": 64, "cos_lr": False,
        "close_mosaic": 10, "hsv_h": .015, "hsv_s": .7, "hsv_v": .4,
        "degrees": 0.0, "translate": .1, "scale": .5, "shear": 0.0,
        "perspective": 0.0, "flipud": 0.0, "fliplr": .5,
        "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0,
        "rect": False, "multi_scale": 0.0, "single_cls": False,
        "overlap_mask": True, "mask_ratio": 4, "fraction": 1.0,
        "pretrained": True, "freeze": None, "val": True,
        "save": True, "save_period": 10, "plots": True,
        "profile": False, "compile": False, "verbose": True,
        "project": str(run_dir.parent), "name": run_dir.name, "exist_ok": True,
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_identity(data_yaml):
    """Cover config, split membership, image stat and annotation content."""
    import yaml

    config_path = Path(data_yaml)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = Path(config.get("path", config_path.parent))
    if not base.is_absolute():
        base = (config_path.parent / base).resolve()
    digest = hashlib.sha256(config_path.read_bytes())
    manifests = {}
    for split in ("train", "val", "test"):
        entries = config.get(split, [])
        if isinstance(entries, str):
            entries = [entries]
        manifests[split] = []
        for entry in entries:
            source = Path(entry)
            if not source.is_absolute():
                source = base / source
            if source.suffix.lower() != ".txt":
                raise ValueError("Use the audited image manifests produced by dataset.prepare_dataset")
            digest.update(source.read_bytes())
            manifests[split].append(str(source.resolve()))
            for line in source.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                image = Path(line.strip())
                if not image.is_absolute():
                    image = source.parent / image
                stat = image.stat()
                digest.update(f"{image}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
                pieces = list(image.parts)
                indices = [i for i, piece in enumerate(pieces) if piece.lower() == "images"]
                if not indices:
                    raise ValueError(f"Image has no images directory for label mapping: {image}")
                pieces[indices[-1]] = "labels"
                label = Path(*pieces).with_suffix(".txt")
                digest.update(label.read_bytes())
    return {"sha256": digest.hexdigest(), "yaml_path": str(config_path.resolve()),
            "manifests": manifests, "class_names": config["names"]}


def experiment_identity(config, data_identity):
    payload = {"config": config, "dataset_sha256": data_identity["sha256"],
               "pretrained_sha256": file_sha256(config["model"]),
               "checkpoint_selection": "validation_mask_map50_95", "schema_version": 1}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest(), payload


def environment_info():
    import torch
    import ultralytics

    return {"python": platform.python_version(), "python_executable": sys.executable,
            "platform": platform.platform(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None}


def verify_environment():
    import torch

    info = environment_info()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; CPU fallback is disabled")
    expected = {"python": "3.11", "torch": "2.11.0", "cuda_runtime": "12.8", "ultralytics": "8.4.30"}
    mismatches = [f"{key}={info[key]} (expected {value})" for key, value in expected.items()
                  if not str(info[key]).startswith(value)]
    if mismatches:
        raise RuntimeError("Environment differs from the pilot specification: " + "; ".join(mismatches))
    probe = torch.ones((32, 32), device="cuda:0")
    if float((probe @ probe).mean()) != 32:
        raise RuntimeError("CUDA numerical probe failed")
    torch.cuda.synchronize(0)
    return info


def csv_epochs(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{str(k).strip(): v for k, v in row.items()} for row in reader]
    return [int(float(row["epoch"])) for row in rows if row.get("epoch")]


def train_and_evaluate_worker(root, dataset, model, stage, batch):
    """Called only in a subprocess under the queue's GPU lock."""
    root = project_root(root)
    run_dir = run_directory(root, dataset, model, stage)
    report_path = run_dir / "report.json"
    report = read_json(report_path)
    if not report or not report.get("fingerprint"):
        raise RuntimeError("Worker requires the parent runner's prepared report and fingerprint")
    report.update(status="running", phase="environment_check", worker_pid=os.getpid(), updated_at=utc_now())
    atomic_json(report_path, report)
    phase = "environment_check"
    try:
        import torch
        from ultralytics import YOLO
        from ultralytics.models.yolo.segment import SegmentationTrainer
        from .evaluation import benchmark_checkpoint, benchmark_manifest, evaluate_checkpoint

        environment = verify_environment()
        previous_environment = report.get("environment")
        if previous_environment and any(previous_environment.get(k) != environment.get(k)
                                        for k in ("python", "torch", "cuda_runtime", "ultralytics", "gpu")):
            raise RuntimeError("Environment changed since this run started; refuse unsafe resume")
        report["environment"] = environment
        config = training_config(root, dataset, model, stage, batch)
        best, last = run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"
        report["checkpoint"] = {"best": str(best), "last": str(last)}
        saved_epochs = int(report.get("epochs_completed", 0))
        target_epochs = STAGES[stage]["epochs"]
        training_done = saved_epochs == target_epochs and best.exists() and last.exists()
        if stage == "main100" and report.get("training_finished_at") and best.exists() and last.exists():
            training_done = True
        if not training_done:
            phase = "training"
            report.update(phase=phase, batch=batch, config=config, updated_at=utc_now())
            atomic_json(report_path, report)
            if saved_epochs and not last.exists():
                raise RuntimeError("Saved progress exists but last.pt is missing; refusing to silently restart")
            resume = last.exists()
            if resume:
                checkpoint_state = torch.load(last, map_location="cpu", weights_only=False)
                actual_epoch = int(checkpoint_state.get("epoch", -1)) + 1
                del checkpoint_state
                if actual_epoch != saved_epochs:
                    raise RuntimeError(f"Checkpoint epoch {actual_epoch} does not match report {saved_epochs}")

            class MaskSelectionTrainer(SegmentationTrainer):
                final_eval_seconds = 0.0

                def validate(self):
                    previous_best = self.best_fitness
                    metrics, _ = super().validate()
                    if metrics is None:
                        return None, None
                    fitness = float(metrics["metrics/mAP50-95(M)"])
                    self.best_fitness = max(previous_best if previous_best is not None else float("-inf"), fitness)
                    return metrics, fitness

                def final_eval(self):
                    started = time.perf_counter()
                    try:
                        return super().final_eval()
                    finally:
                        self.final_eval_seconds += time.perf_counter() - started

            trained_model = YOLO(str(last if resume else config["model"]), task="segment")
            if trained_model.task != "segment":
                raise ValueError("Expected YOLO instance segmentation weights")
            previous_seconds = float(report.get("train_wall_seconds", 0))
            train_start = time.perf_counter()
            attempt = {"started_at": utc_now(), "batch_requested": batch, "resume_from_epoch": saved_epochs,
                       "checkpoint_source": str(last if resume else config["model"])}
            report.setdefault("attempts", []).append(attempt)

            def record_checkpoint(trainer):
                report["epochs_completed"] = int(trainer.epoch) + 1
                report["batch"] = int(trainer.batch_size)
                report["train_wall_seconds"] = previous_seconds + time.perf_counter() - train_start
                report["updated_at"] = utc_now()
                report["latest_mask_map50_95"] = float(trainer.metrics["metrics/mAP50-95(M)"])
                report["best_mask_map50_95"] = float(trainer.best_fitness)
                atomic_json(report_path, report)

            trained_model.add_callback("on_model_save", record_checkpoint)
            try:
                trained_model.train(trainer=MaskSelectionTrainer, resume=str(last) if resume else False, **config)
                attempt["status"] = "completed"
                report["training_finished_at"] = utc_now()
            finally:
                trainer = getattr(trained_model, "trainer", None)
                final_eval_seconds = getattr(trainer, "final_eval_seconds", 0.0)
                elapsed = time.perf_counter() - train_start - final_eval_seconds
                report["train_wall_seconds"] = previous_seconds + elapsed
                attempt.update(ended_at=utc_now(), wall_seconds=elapsed,
                               automatic_final_evaluation_seconds=final_eval_seconds,
                               batch_effective=int(getattr(trainer, "batch_size", batch)))
                atomic_json(report_path, report)
            observed_epochs = csv_epochs(run_dir / "results.csv")
            if stage == "pilot30" and observed_epochs != list(range(1, target_epochs + 1)):
                raise RuntimeError(f"Pilot must contain exactly epochs 1..30; observed {observed_epochs}")
            report["epochs_completed"] = max(observed_epochs, default=0)
            report["epoch_budget_met"] = report["epochs_completed"] == target_epochs
            del trained_model, trainer
            gc.collect()
            torch.cuda.empty_cache()
        if not best.exists() or not last.exists():
            raise RuntimeError("Training did not produce both best.pt and last.pt")
        if stage == "pilot30" and csv_epochs(run_dir / "results.csv") != list(range(1, 31)):
            raise RuntimeError("Pilot checkpoint budget cannot be verified from results.csv")
        phase = "validation"
        report.update(phase=phase, updated_at=utc_now())
        atomic_json(report_path, report)
        report["validation"] = evaluate_checkpoint(best, config["data"], run_dir / "evaluation" / "val", split="val")
        # Pilot never reads test labels for metrics or model selection.
        if stage == "main100":
            report["test"] = evaluate_checkpoint(best, config["data"], run_dir / "evaluation" / "test", split="test")
        phase = "benchmark"
        report.update(phase=phase, updated_at=utc_now())
        atomic_json(report_path, report)
        manifest = benchmark_manifest(config["data"], run_dir.parent / "benchmark_manifest.json")
        report["benchmark"] = benchmark_checkpoint(best, manifest, run_dir / "benchmark")
        # Examples are outside the benchmark timing region and use the same samples.
        from .examples import generate_examples
        phase = "examples"
        report.update(phase=phase, updated_at=utc_now())
        atomic_json(report_path, report)
        report["examples"] = generate_examples(best, manifest, run_dir / "examples", dataset)
        report.update(status="completed", phase="completed", completed_at=utc_now(), updated_at=utc_now())
        report["checkpoint"]["best_sha256"] = file_sha256(best)
        atomic_json(report_path, report)
        return 0
    except BaseException as error:
        oom = "out of memory" in str(error).lower() and "cuda" in str(error).lower()
        report.update(status="oom" if oom and phase == "training" else "failed", phase=phase, updated_at=utc_now())
        report.setdefault("errors", []).append({"time": utc_now(), "phase": phase,
                                                 "type": type(error).__name__, "message": str(error),
                                                 "traceback": traceback.format_exc()})
        atomic_json(report_path, report)
        traceback.print_exc()
        return 42 if oom and phase == "training" else 1


def run_experiment(dataset, model, stage="pilot30", project_root=None):
    """Notebook convenience entry point; always isolates GPU work in a process."""
    from .runner import run_experiment as execute
    return execute(dataset, model, stage=stage, project_root=project_root)
