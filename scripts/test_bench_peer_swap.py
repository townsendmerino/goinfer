#!/usr/bin/env python3
"""bench_peer.py's swap watch (peer-vetted 2026-10-07): any swap growth over BENCH_SWAP_VOID_MB voids the engine's arm, and
growth over BENCH_SWAP_KILL_MB kills the server's process group. Run: python3 -B scripts/test_bench_peer_swap.py"""
import os
import subprocess
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_peer  # noqa: E402


class SwapWatch(unittest.TestCase):
    def setUp(self):
        for k in ("BENCH_SWAP_VOID_MB", "BENCH_SWAP_KILL_MB", "BENCH_SWAPIN_VOID_MB"):
            self.addCleanup(os.environ.pop, k, None)

    def test_void_rule(self):
        w = bench_peer.SwapWatch()
        self.assertIsNone(w.void_reason(), "off unless BENCH_SWAP_VOID_MB is set")
        os.environ["BENCH_SWAP_VOID_MB"] = "0"
        w.peak = 0.0
        self.assertIsNone(w.void_reason(), "no growth is not a void")
        w.peak = 0.5
        self.assertIn("void", w.void_reason())

    def test_swapin_rule(self):
        # nobara's rule (2026-10-08): swap growth from idle pages pushed out is not a void; swap-ins over the limit are.
        os.environ["BENCH_SWAP_VOID_MB"] = "256"
        os.environ["BENCH_SWAPIN_VOID_MB"] = "100"
        w = bench_peer.SwapWatch()
        w.peak, w.swapin = 46.1, 0.0
        self.assertIsNone(w.void_reason(), "46 MB of growth and no swap-ins is not a void under the Linux rule")
        w.swapin = 120.0
        self.assertIn("swap-ins", w.void_reason())
        w.swapin, w.peak = 0.0, 300.0
        self.assertIn("grew", w.void_reason())

    def test_kill_stops_the_server_group(self):
        os.environ["BENCH_SWAP_KILL_MB"] = "100"
        proc = subprocess.Popen(["sleep", "60"], preexec_fn=os.setsid)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        w = bench_peer.SwapWatch()
        w.base = 0.0
        real = bench_peer.swap_used_mb
        bench_peer.swap_used_mb = lambda: 250.0   # a load that grew swap by 250 MB
        self.addCleanup(setattr, bench_peer, "swap_used_mb", real)
        w.start(proc.pid)
        for _ in range(50):
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        stats = w.stop()
        self.assertIsNotNone(proc.poll(), "the server was not killed")
        self.assertIn("killed", stats["swap_killed"])
        self.assertIn("killed", w.void_reason())


if __name__ == "__main__":
    unittest.main()
