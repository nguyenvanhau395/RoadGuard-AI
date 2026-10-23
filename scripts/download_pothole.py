"""Download the authorized Roboflow Pothole v2 YOLO segmentation export.

Credentials are used only in memory and are never logged or copied to files.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "major-vl1h9/pothole-bwzav"
VERSION = 2


def credentials(use_project_notebooks: bool):
    if os.environ.get("ROBOFLOW_API_KEY"):
        yield os.environ["ROBOFLOW_API_KEY"]
    if use_project_notebooks:
        found = set()
        for path in sorted((ROOT / "archive/notebooks").glob("*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell in notebook.get("cells", []):
                source = "".join(cell.get("source", []))
                for key in re.findall(r'''Roboflow\(\s*api_key\s*=\s*["']([^"']+)''', source):
                    if key not in found and key not in {"API_KEY", "YOUR_API_KEY"}:
                        found.add(key)
                        yield key


def get_json(path: str, key: str):
    query = urllib.parse.urlencode({"api_key": key})
    request = urllib.request.Request(f"https://api.roboflow.com/{path}?{query}", headers={"User-Agent": "road-damage-vision/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-existing-project-key", action="store_true")
    args = parser.parse_args()
    destination = ROOT / "data/raw/pothole-v2"
    receipt = ROOT / "reports/data_quality/pothole_download.json"
    if (destination / "data.yaml").exists() and receipt.exists():
        print("Pothole v2 is already downloaded; dataset verification can proceed.", flush=True)
        return
    key = None
    metadata = None
    for candidate in credentials(args.use_existing_project_key):
        try:
            metadata = get_json(f"{PROJECT}/{VERSION}", candidate)
            key = candidate
            break
        except urllib.error.HTTPError as error:
            print(f"Credential access returned HTTP {error.code}.", flush=True)
    if key is None:
        raise SystemExit("Set ROBOFLOW_API_KEY with access to the public Pothole dataset.")
    export = get_json(f"{PROJECT}/{VERSION}/yolov8", key)
    for _ in range(30):
        package = export.get("export", {})
        link = package.get("link") or export.get("link")
        if link:
            break
        print("Waiting for Roboflow YOLOv8 export.", flush=True)
        time.sleep(10)
        export = get_json(f"{PROJECT}/{VERSION}/yolov8", key)
    else:
        raise SystemExit("Roboflow export is not yet available; rerun the downloader later.")
    archive = ROOT / "data/archives/pothole-v2-yolov8.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(".zip.partial")
    started = time.monotonic()
    if not archive.exists() or not zipfile.is_zipfile(archive):
        with urllib.request.urlopen(link, timeout=120) as response, partial.open("wb") as output:
            downloaded = 0
            last_print = time.monotonic()
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                if time.monotonic() - last_print >= 15:
                    print(f"Downloaded {downloaded / 1024**2:.1f} MiB.", flush=True)
                    last_print = time.monotonic()
        partial.replace(archive)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise SystemExit("Unsafe archive path detected; extraction stopped.")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise SystemExit("Unexpected symbolic link in dataset archive.")
        package.extractall(destination)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    version = metadata.get("version", {})
    receipt.write_text(json.dumps({
        "source": f"https://universe.roboflow.com/{PROJECT}/dataset/{VERSION}",
        "project": PROJECT, "version": VERSION, "format": "yolov8",
        "published_splits": version.get("splits"),
        "archive_bytes": archive.stat().st_size,
        "download_extract_seconds": time.monotonic() - started,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "archive": str(archive.relative_to(ROOT)), "dataset": str(destination.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Dataset ready at {destination.relative_to(ROOT)}.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as error:
        raise SystemExit(f"Roboflow/download request failed: HTTP {error.code}; credentials omitted.")
    except urllib.error.URLError:
        raise SystemExit("Dataset download network failure; credentials omitted.")
