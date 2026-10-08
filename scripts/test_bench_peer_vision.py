#!/usr/bin/env python3
"""bench_peer.py's vision cell sends a fresh image per request (S13's harness fix, docs/tasks/task-multimodal-support-2026-10.md):
the 2026-09-20 row sent one image's bytes for the warm-up and every timed run, so every engine's image cache could answer it.
Run: python3 -B scripts/test_bench_peer_vision.py"""
import hashlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_peer  # noqa: E402
from PIL import Image  # noqa: E402


class FreshImage(unittest.TestCase):
    def test_every_request_differs_in_bytes_not_size_or_content(self):
        base = Image.open(bench_peer.VISION_IMAGE_PATH).convert("RGB")
        imgs = [bench_peer.fresh_image(i) for i in range(4)]
        self.assertEqual(len({hashlib.sha256(b).digest() for b in imgs}), 4, "two requests share bytes: a cache could answer one")
        for b in imgs:
            im = Image.open(io.BytesIO(b)).convert("RGB")
            self.assertEqual(im.size, base.size)
            self.assertLessEqual(max(abs(x - y) for x, y in zip(im.tobytes(), base.tobytes())), 1)

    def test_repeatable(self):
        self.assertEqual(bench_peer.fresh_image(2), bench_peer.fresh_image(2))


if __name__ == "__main__":
    unittest.main()
