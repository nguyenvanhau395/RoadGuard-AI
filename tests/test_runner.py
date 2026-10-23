import json
from pathlib import Path
import tempfile
import unittest

from road_damage.runner import gpu_lease
from road_damage.training import training_config


class QueueTests(unittest.TestCase):
    def test_live_lease_prevents_a_second_gpu_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with gpu_lease(root):
                with self.assertRaisesRegex(RuntimeError, "Another project"):
                    with gpu_lease(root):
                        self.fail("Second process must not acquire the GPU")
            self.assertFalse((root / "runs/seg/.gpu.lock").exists())

    def test_stale_lease_is_reclaimed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "runs/seg/.gpu.lock"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"pid": -1, "token": "stale"}), encoding="utf-8")
            with gpu_lease(root) as (_, record):
                self.assertNotEqual(record["token"], "stale")
            self.assertFalse(path.exists())

    def test_pilot_budget_is_exact_and_runs_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            configs = [training_config(directory, dataset, size) for dataset in ("crack", "pothole") for size in ("n", "s", "m")]
            self.assertEqual(len({(config["project"], config["name"]) for config in configs}), 6)
            self.assertTrue(all(config["epochs"] == 30 and config["patience"] == 0 for config in configs))
            self.assertTrue(all(config["task"] == "segment" and config["model"].endswith("-seg.pt") for config in configs))


if __name__ == "__main__":
    unittest.main()
