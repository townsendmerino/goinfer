#!/usr/bin/env python3
"""bench_peer.py's instant-gate preflight waits for a passing sample and still refuses on timeout (TE1 attempt 3's
instant sweep was refused by one sample at 10.8% against the 10% cap, right after the run script's own sample of the
same gate passed). Run: python3 -B scripts/test_bench_peer_preflight.py"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_peer  # noqa: E402  (import has no side effects: main() runs only under __main__)


class InstantPreflight(unittest.TestCase):
    def setUp(self):
        self.patches = [
            mock.patch.object(bench_peer, "IDLE_GATE", "instant"),
            mock.patch.object(bench_peer, "check_bench_disk", lambda *a, **k: None),
            mock.patch.object(bench_peer, "_gpu_compute_apps", lambda: []),
            mock.patch.object(bench_peer, "_gpu_state", lambda: None),
            mock.patch.object(bench_peer.time, "sleep", lambda s: None),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def test_waits_through_a_busy_sample_then_passes(self):
        samples = iter([(10.8, []), (12.0, []), (4.0, [])])
        calls = []

        def sample():
            calls.append(1)
            return next(samples)

        with mock.patch.object(bench_peer, "instant_idle_sample", sample), mock.patch.dict(os.environ, {"BENCH_IDLE_WAIT": "600"}):
            bench_peer.preflight()  # returns instead of sys.exit
        self.assertEqual(len(calls), 3, "the preflight should have sampled until the first passing reading")

    def test_still_refuses_after_the_wait(self):
        clock = iter(range(0, 10_000, 100))  # each call to time.time() advances 100 s
        with mock.patch.object(bench_peer, "instant_idle_sample", lambda: (50.0, [])), \
                mock.patch.object(bench_peer.time, "time", lambda: next(clock)), \
                mock.patch.dict(os.environ, {"BENCH_IDLE_WAIT": "300"}):
            with self.assertRaises(SystemExit) as cm:
                bench_peer.preflight()
        self.assertIn("REFUSED", str(cm.exception))
        self.assertIn("after waiting", str(cm.exception))

    def test_a_timed_workload_blocks_even_at_low_busy(self):
        clock = iter(range(0, 10_000, 100))
        with mock.patch.object(bench_peer, "instant_idle_sample", lambda: (1.0, [(123, "bench", 2.0)])), \
                mock.patch.object(bench_peer.time, "time", lambda: next(clock)), \
                mock.patch.dict(os.environ, {"BENCH_IDLE_WAIT": "200"}):
            with self.assertRaises(SystemExit) as cm:
                bench_peer.preflight()
        self.assertIn("timed workloads active", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
