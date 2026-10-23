"""One real forward/backward SEG batch per dataset/model, outside pilot runs."""
import gc
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
runtime = ROOT / "reports/runtime/ultralytics"
runtime.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(runtime)

import torch
import yaml
from ultralytics.models.yolo.segment import SegmentationTrainer
from road_damage.training import verify_environment


def main():
    environment = verify_environment()
    summary = {"environment": environment, "cases": []}
    out = ROOT / "reports/smoke"
    out.mkdir(parents=True, exist_ok=True)
    for dataset in ("crack", "pothole"):
        original = yaml.safe_load((ROOT / f"configs/seg/{dataset}.yaml").read_text(encoding="utf-8"))
        tiny = {"path": str(ROOT), "names": original["names"], "nc": original["nc"]}
        for split in ("train", "val"):
            source = Path(original[split])
            manifest = out / f"{dataset}_{split}.txt"
            manifest.write_text("\n".join(source.read_text(encoding="utf-8").splitlines()[:4]) + "\n", encoding="utf-8")
            tiny[split] = str(manifest)
        config = out / f"{dataset}.yaml"
        config.write_text(yaml.safe_dump(tiny), encoding="utf-8")
        for size in ("n", "s", "m"):
            start = time.perf_counter()
            trainer = SegmentationTrainer(overrides={
                "model": str(ROOT / f"models/pretrained/yolov8{size}-seg.pt"),
                "data": str(config), "epochs": 1, "batch": 1, "imgsz": 640,
                "device": 0, "workers": 0, "amp": False, "plots": False,
                "save": False, "cache": False, "val": False,
                "project": str(out / "batches"), "name": f"{dataset}_{size}", "exist_ok": True,
                "optimizer": "AdamW", "lr0": .001, "verbose": False,
            })
            trainer._setup_train()
            batch = trainer.preprocess_batch(next(iter(trainer.train_loader)))
            trainer.model.train()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, components = trainer.model(batch)
                loss = loss.sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite SEG loss: {dataset}/{size}")
            loss.backward()
            if not any(p.grad is not None and torch.isfinite(p.grad).all() for p in trainer.model.parameters()):
                raise RuntimeError("No finite model gradients")
            torch.cuda.synchronize(0)
            summary["cases"].append({"dataset": dataset, "model": f"yolov8{size}-seg", "loss": float(loss.detach()),
                                     "loss_components": components.detach().cpu().tolist(), "seconds": time.perf_counter() - start,
                                     "mask_shape": list(batch["masks"].shape), "status": "passed"})
            print(f"SMOKE PASSED {dataset}/{size}: loss={float(loss.detach()):.4f}", flush=True)
            del loss, components, batch, trainer
            gc.collect()
            torch.cuda.empty_cache()
    (out / "gpu_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
