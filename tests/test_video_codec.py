import sys
import tempfile
import unittest
from pathlib import Path

import imageio_ffmpeg
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import test_interface as interface  # noqa: E402


class ByteRangeTests(unittest.TestCase):
    def test_standard_open_and_suffix_ranges(self):
        self.assertEqual(interface._parse_byte_range("bytes=0-99", 1000), (0, 99))
        self.assertEqual(interface._parse_byte_range("bytes=900-", 1000), (900, 999))
        self.assertEqual(interface._parse_byte_range("bytes=-50", 1000), (950, 999))
        self.assertIsNone(interface._parse_byte_range(None, 1000))

    def test_invalid_ranges_fail_explicitly(self):
        for value in ("items=0-9", "bytes=-0", "bytes=1000-", "bytes=20-10"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                interface._parse_byte_range(value, 1000)


class BrowserVideoWriterTests(unittest.TestCase):
    def test_writer_outputs_browser_safe_h264(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.mp4"
            writer = interface._BrowserVideoWriter(output, 12, 64, 48)
            for index in range(6):
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                frame[:, :, index % 3] = 40 + index * 30
                writer.write(frame)
            writer.close()

            reader = imageio_ffmpeg.read_frames(str(output), pix_fmt="rgb24")
            metadata = next(reader)
            reader.close()
            self.assertEqual(metadata["codec"], "h264")
            self.assertEqual(metadata["size"], (64, 48))
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
