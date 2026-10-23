"""Synthetic fixtures verify filtering without modifying any real dataset."""

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from road_damage.dataset import prepare_dataset, validate_label


VALID_POLYGON = "0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n"


class LabelValidationTests(unittest.TestCase):
    def test_valid_polygons_preserve_class_ids_and_allow_blank_background(self):
        self.assertEqual(validate_label(" \n\n", 2), [])
        polygons = validate_label(VALID_POLYGON + VALID_POLYGON.replace("0 ", "1 ", 1), 2)
        self.assertEqual([polygon["class_id"] for polygon in polygons], [0, 1])

    def test_detection_rows_and_invalid_polygons_are_rejected(self):
        for text in (
            "0 0.5 0.5 0.2 0.2", "0 0 0 1 0 1", "2 0 0 1 0 1 1",
            "0 nan 0 1 0 1 1", "0 0 0 1.1 0 1 1", "0 0 0 0 0 1 1",
            "0.5 0 0 1 0 1 1",
        ):
            with self.subTest(label=text), self.assertRaises(ValueError):
                validate_label(text, 1)


class DatasetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "data" / "raw" / "crack-4"
        self.raw.mkdir(parents=True)
        (self.raw / "data.yaml").write_text("names: [crack]\nnc: 1\npath: D:/stale/location\n", encoding="utf-8")
        for split in ("train", "valid", "test"):
            (self.raw / split / "images").mkdir(parents=True)
            (self.raw / split / "labels").mkdir()
            self.add_sample(split, f"{split}_positive", VALID_POLYGON)

    def tearDown(self):
        self.temp.cleanup()

    def add_sample(self, split, name, label, extension=".png"):
        Image.new("RGB", (12, 8), "gray").save(self.raw / split / "images" / f"{name}{extension}")
        if label is not None:
            (self.raw / split / "labels" / f"{name}.txt").write_text(label, encoding="utf-8")

    def test_manifest_keeps_explicit_background_and_excludes_bad_pairs(self):
        self.add_sample("train", "background", "")
        self.add_sample("train", "missing", None)
        self.add_sample("train", "invalid", "0 0 0 1 0")
        self.add_sample("train", "corrupt", VALID_POLYGON)
        corrupt = self.raw / "train" / "images" / "corrupt.png"
        corrupt.write_bytes(b"not an image")
        result = prepare_dataset("crack", root=self.root, previews=False)
        stats = result["splits"]["train"]
        self.assertEqual(stats["total_images"], 5)
        self.assertEqual(stats["usable_images"], 2)
        self.assertEqual(stats["blank_labels"], 1)
        self.assertEqual(stats["excluded_images"], 3)
        self.assertEqual(stats["polygon_count"], 1)
        paths = Path(result["manifest_paths"]["train"]).read_text().splitlines()
        self.assertEqual({Path(path).stem for path in paths}, {"train_positive", "background"})
        self.assertTrue(all(Path(path).is_absolute() for path in paths))
        self.assertEqual(corrupt.read_bytes(), b"not an image")
        self.assertFalse((self.raw / "train" / "labels" / "missing.txt").exists())
        config = yaml.safe_load(Path(result["yaml_path"]).read_text())
        self.assertTrue(Path(config["val"]).is_absolute())
        self.assertEqual(config["names"], {0: "crack"})
        self.assertEqual(json.loads(Path(result["audit_path"]).read_text())["status"], "ready")

    def test_duplicate_stems_are_excluded_instead_of_guessing_label_pairing(self):
        self.add_sample("train", "ambiguous", VALID_POLYGON, ".png")
        self.add_sample("train", "ambiguous", VALID_POLYGON, ".jpg")
        result = prepare_dataset("crack", root=self.root, previews=False)
        self.assertEqual(result["splits"]["train"]["usable_images"], 1)
        self.assertEqual(result["splits"]["train"]["excluded_images"], 2)
        self.assertEqual(sum(issue["code"] == "duplicate_stem" for issue in result["issues"]), 2)

    def test_reaudit_after_raw_label_changes_and_keep_mapping(self):
        first = prepare_dataset("crack", root=self.root, previews=False)
        target = self.raw / "valid" / "labels" / "valid_positive.txt"
        target.write_text("", encoding="utf-8")
        second = prepare_dataset("crack", root=self.root, previews=False)
        self.assertNotEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(second["splits"]["val"]["blank_labels"], 1)
        cached = prepare_dataset("crack", root=self.root, previews=False)
        self.assertEqual(cached["class_names"], {0: "crack"})
        self.assertEqual(cached["audited_at_utc"], second["audited_at_utc"])

    def test_empty_usable_split_fails_with_audit_evidence(self):
        (self.raw / "test" / "labels" / "test_positive.txt").unlink()
        with self.assertRaisesRegex(ValueError, "No usable images"):
            prepare_dataset("crack", root=self.root, previews=False)
        audit = json.loads((self.root / "reports" / "data_quality" / "crack" / "audit.json").read_text())
        self.assertEqual(audit["status"], "failed")
        self.assertFalse((self.root / "configs" / "seg" / "crack.yaml").exists())


if __name__ == "__main__":
    unittest.main()
