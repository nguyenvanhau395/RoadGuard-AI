"""Basic, non-destructive audits of YOLO instance segmentation exports.

Raw images and labels are never rewritten. Missing or malformed labels exclude
their image from the generated manifest; an existing empty label is valid
background. The generated YAML uses absolute paths and should be regenerated
after moving the project.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageOps

from .paths import dataset_root, project_root

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
AUDIT_VERSION = 1


def validate_label(text: str, num_classes: int) -> list[dict]:
    """Parse a YOLO polygon label, raising ValueError for any invalid row."""
    polygons = []
    for line_number, line in enumerate(text.splitlines(), 1):
        values = line.split()
        if not values:
            continue
        try:
            class_id = int(values[0])
        except ValueError as exc:
            raise ValueError(f"line {line_number}: class ID must be an integer") from exc
        if not 0 <= class_id < num_classes:
            raise ValueError(f"line {line_number}: unknown class ID {class_id}")
        if len(values) < 7 or (len(values) - 1) % 2:
            raise ValueError(f"line {line_number}: polygon needs at least three x/y vertices")
        try:
            coordinates = [float(value) for value in values[1:]]
        except ValueError as exc:
            raise ValueError(f"line {line_number}: coordinates must be numeric") from exc
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in coordinates):
            raise ValueError(f"line {line_number}: coordinates must be finite and within [0, 1]")
        vertices = list(zip(coordinates[::2], coordinates[1::2]))
        if len(set(vertices)) < 3:
            raise ValueError(f"line {line_number}: polygon needs three distinct vertices")
        polygons.append({"class_id": class_id, "vertices": vertices})
    return polygons


def _load_classes(raw: Path) -> dict[int, str]:
    config_path = raw / "data.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset export has no data.yaml: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    names = config.get("names") if isinstance(config, dict) else None
    if isinstance(names, list):
        names = dict(enumerate(names))
    elif isinstance(names, dict):
        try:
            names = {int(key): value for key, value in names.items()}
        except (ValueError, TypeError) as exc:
            raise ValueError("data.yaml names must use integer class IDs") from exc
    else:
        raise ValueError("data.yaml needs a names list or mapping")
    if not names or sorted(names) != list(range(len(names))):
        raise ValueError("Class IDs must be contiguous, starting at zero")
    if any(not isinstance(name, str) or not name.strip() for name in names.values()):
        raise ValueError("Class names must be non-empty strings")
    if "nc" in config and int(config["nc"]) != len(names):
        raise ValueError("data.yaml nc disagrees with names")
    return dict(sorted(names.items()))


def _split_directory(raw: Path, split: str) -> Path:
    candidates = [raw / name for name in (("valid", "val") if split == "val" else (split,))]
    available = [path for path in candidates if (path / "images").is_dir()]
    if len(available) != 1:
        raise ValueError(f"Expected exactly one images directory for {split}: {candidates}")
    return available[0]


def _fingerprint(raw: Path, split_dirs: dict[str, Path]) -> str:
    """Invalidate an audit when roots, file contents metadata, or config change."""
    digest = hashlib.sha256()
    digest.update(f"{AUDIT_VERSION}|{raw.resolve()}".encode("utf-8"))
    digest.update((raw / "data.yaml").read_bytes())
    for split, directory in split_dirs.items():
        digest.update(split.encode("utf-8"))
        for subdir in ("images", "labels"):
            for path in sorted((directory / subdir).glob("*")):
                if path.is_file():
                    stat = path.stat()
                    digest.update(f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8"))
    return digest.hexdigest()


def _preview(images: list[Path], label_dir: Path, names: dict[int, str], destination: Path) -> None:
    """Write a deterministic 3-by-3 contact sheet, without changing raw data."""
    tile_width, tile_height = 288, 310
    canvas = Image.new("RGB", (tile_width * 3, tile_height * 3), "#151b24")
    draw = ImageDraw.Draw(canvas)
    selected = [images[round(index * (len(images) - 1) / min(8, len(images) - 1))]
                for index in range(min(9, len(images)))] if len(images) > 1 else images[:1]
    palette = [(72, 205, 165), (247, 172, 80), (99, 163, 255), (245, 115, 152)]
    for index, path in enumerate(selected):
        with Image.open(path) as source:
            picture = source.convert("RGB")
        overlay = Image.new("RGBA", picture.size)
        overlay_draw = ImageDraw.Draw(overlay)
        polygons = validate_label((label_dir / f"{path.stem}.txt").read_text(encoding="utf-8-sig"), len(names))
        for polygon in polygons:
            color = palette[polygon["class_id"] % len(palette)]
            points = [(x * (picture.width - 1), y * (picture.height - 1)) for x, y in polygon["vertices"]]
            overlay_draw.polygon(points, fill=(*color, 60), outline=(*color, 255), width=2)
        picture = Image.alpha_composite(picture.convert("RGBA"), overlay).convert("RGB")
        tile = ImageOps.contain(picture, (tile_width - 10, tile_height - 40))
        x, y = (index % 3) * tile_width, (index // 3) * tile_height
        canvas.paste(tile, (x + (tile_width - tile.width) // 2, y + 5))
        draw.text((x + 5, y + tile_height - 31), path.name[:36], fill="white")
        draw.text((x + 5, y + tile_height - 17), f"{len(polygons)} polygons", fill="#aebace")
    canvas.save(destination, quality=90)


def prepare_dataset(
    dataset: str,
    *,
    root: str | Path | None = None,
    force: bool = False,
    previews: bool = True,
) -> dict:
    """Audit all three splits and write YOLO YAML, manifests, and audit JSON.

    A cached audit is reused only while raw file metadata and paths match and all
    generated files still exist. Set force=True to decode every image again.
    At least one usable image is required in every split. Invalid data remains
    untouched, with each excluded image and its reason recorded in audit.json.
    """
    project = project_root(root)
    raw = dataset_root(dataset, project)
    names = _load_classes(raw)
    split_dirs = {split: _split_directory(raw, split) for split in ("train", "val", "test")}
    fingerprint = _fingerprint(raw, split_dirs)
    report_dir = project / "reports" / "data_quality" / dataset
    audit_path = report_dir / "audit.json"
    manifest_dir = project / "data" / "manifests" / dataset
    yaml_path = project / "configs" / "seg" / f"{dataset}.yaml"
    if not force and audit_path.exists():
        try:
            cached = json.loads(audit_path.read_text(encoding="utf-8"))
            files = [yaml_path, *[manifest_dir / f"{split}.txt" for split in split_dirs]]
            if previews:
                files.extend(report_dir / f"preview_{split}.jpg" for split in split_dirs)
            if cached.get("fingerprint") == fingerprint and all(path.is_file() for path in files):
                cached["class_names"] = {int(key): value for key, value in cached["class_names"].items()}
                return cached
        except (ValueError, KeyError, OSError):
            pass

    summary = {
        "dataset": dataset,
        "raw_path": str(raw),
        "yaml_path": str(yaml_path),
        "audit_path": str(audit_path),
        "fingerprint": fingerprint,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "class_names": names,
        "manifest_paths": {},
        "splits": {},
        "issues": [],
        "excluded": [],
        "preview_paths": {},
    }
    manifests = {}
    global_stems = defaultdict(list)
    for split, directory in split_dirs.items():
        image_dir, label_dir = directory / "images", directory / "labels"
        images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        stem_counts = Counter(path.stem.casefold() for path in images)
        stats = {
            "total_images": len(images), "usable_images": 0, "excluded_images": 0,
            "blank_labels": 0, "polygon_count": 0, "class_counts": dict.fromkeys(names, 0),
            "image_shapes": {}, "missing_labels": 0, "invalid_labels": 0, "unreadable_images": 0,
        }
        usable = []
        shape_counts = Counter()
        for image_path in images:
            relative = str(image_path.relative_to(raw))
            global_stems[image_path.stem.casefold()].append({"split": split, "path": relative})
            label_path = label_dir / f"{image_path.stem}.txt"
            problems = []
            if stem_counts[image_path.stem.casefold()] > 1:
                problems.append(("duplicate_stem", "Ambiguous image/label pairing within split"))
            try:
                with Image.open(image_path) as picture:
                    picture.load()
                    width, height = picture.size
                    if width < 1 or height < 1:
                        raise ValueError("Image has no pixels")
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
                stats["unreadable_images"] += 1
                problems.append(("unreadable_image", str(exc)))
            polygons = []
            if not label_path.is_file():
                stats["missing_labels"] += 1
                problems.append(("missing_label", "Missing labels are excluded, not inferred to be background"))
            else:
                try:
                    polygons = validate_label(label_path.read_text(encoding="utf-8-sig"), len(names))
                except (OSError, UnicodeError, ValueError) as exc:
                    stats["invalid_labels"] += 1
                    problems.append(("invalid_label", str(exc)))
            if problems:
                summary["excluded"].append({"split": split, "image": relative, "reasons": [code for code, _ in problems]})
                for code, detail in problems:
                    summary["issues"].append({"split": split, "image": relative, "code": code, "detail": detail})
                continue
            usable.append(image_path.resolve())
            shape_counts[f"{width}x{height}"] += 1
            stats["blank_labels"] += not polygons
            stats["polygon_count"] += len(polygons)
            for polygon in polygons:
                stats["class_counts"][polygon["class_id"]] += 1
        known_stems = set(stem_counts)
        for label_path in sorted(label_dir.glob("*.txt")):
            if label_path.stem.casefold() not in known_stems:
                summary["issues"].append({"split": split, "label": str(label_path.relative_to(raw)), "code": "orphan_label", "detail": "Label has no matching image"})
        stats["usable_images"] = len(usable)
        stats["excluded_images"] = len(images) - len(usable)
        stats["image_shapes"] = dict(shape_counts)
        summary["splits"][split] = stats
        manifests[split] = usable
    for stem, locations in global_stems.items():
        if len({entry["split"] for entry in locations}) > 1:
            summary["issues"].append({"code": "cross_split_duplicate_stem", "stem": stem, "locations": locations, "detail": "Possible split overlap; content was not deduplicated"})

    report_dir.mkdir(parents=True, exist_ok=True)
    empty_splits = [split for split, images in manifests.items() if not images]
    if empty_splits:
        summary["status"] = "failed"
        audit_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        raise ValueError(f"No usable images in {empty_splits}; see {audit_path}")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    for split, images in manifests.items():
        manifest_path = manifest_dir / f"{split}.txt"
        manifest_path.write_text("\n".join(path.as_posix() for path in images) + "\n", encoding="utf-8")
        summary["manifest_paths"][split] = str(manifest_path)
        if previews:
            preview_path = report_dir / f"preview_{split}.jpg"
            _preview(images, split_dirs[split] / "labels", names, preview_path)
            summary["preview_paths"][split] = str(preview_path)
    config = {
        "path": project.as_posix(),
        **{split: Path(path).as_posix() for split, path in summary["manifest_paths"].items()},
        "nc": len(names), "names": names,
    }
    yaml_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    summary["status"] = "ready"
    audit_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
