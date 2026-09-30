#!/usr/bin/env python3
"""bench_peer.py's decode-path record and the GPU-cell fallback check (docs/measurements/peer-sweep-2026-09-29.md).
Run: python3 -B scripts/test_bench_peer_decodepath.py"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_peer  # noqa: E402

FALLBACK = "cpu (int8int8) — requested metal → running on cpu: every resident metal projection quantizes activations"


class DecodePath(unittest.TestCase):
    def log(self, text):
        f = tempfile.NamedTemporaryFile("w", delete=False, suffix=".log")
        f.write(text)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_reads_the_last_decode_path_line(self):
        p = self.log("note: x\n  decode path: cpu (int4) - requested metal\n  decode path: metal-resident (int4)\nready\n")
        self.assertEqual(bench_peer.decode_path_since(p), "metal-resident (int4)")

    def test_offset_skips_an_earlier_cells_output(self):
        first = "  decode path: metal-resident (int4)\n"
        p = self.log(first + "  decode path: " + FALLBACK + "\n")
        self.assertEqual(bench_peer.decode_path_since(p, len(first)), FALLBACK)
        self.assertIsNone(bench_peer.decode_path_since(p, os.path.getsize(p)))

    def test_missing_log_is_none_not_an_error(self):
        self.assertIsNone(bench_peer.decode_path_since("/nonexistent/x.log"))

    def test_a_gpu_cell_on_the_cpu_is_refused(self):
        for be in ("metal", "cuda", "webgpu"):
            self.assertIn("CPU's", bench_peer.decode_path_mismatch(FALLBACK, be))

    def test_a_resident_path_and_a_cpu_cell_pass(self):
        self.assertIsNone(bench_peer.decode_path_mismatch("metal-resident (int4)", "metal"))
        self.assertIsNone(bench_peer.decode_path_mismatch("cuda-resident (int4)", "cuda"))
        self.assertIsNone(bench_peer.decode_path_mismatch(FALLBACK, "cpu"))   # a CPU cell may be on the CPU
        self.assertIsNone(bench_peer.decode_path_mismatch(None, "metal"))     # no line: -require-backend is the guard


if __name__ == "__main__":
    unittest.main()
