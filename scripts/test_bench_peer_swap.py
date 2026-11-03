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
        for k in ("BENCH_SWAP_VOID_MB", "BENCH_SWAP_KILL_MB", "BENCH_SWAPIN_VOID_MB", "BENCH_SWAPOUT_VOID_MB", "BENCH_SWAP_VOID_STOP"):
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

    def test_parse_vm_stat(self):
        text = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:      100.\n"
                "Swapins:      64.\nSwapouts:     128.\n")
        self.assertEqual(bench_peer.parse_vm_stat(text), (1.0, 2.0))
        self.assertEqual(bench_peer.parse_vm_stat("no header\nSwapins: 64.\n"), (None, None))
        self.assertEqual(bench_peer.parse_vm_stat("(page size of 4096 bytes)\n"), (None, None))

    def test_swapout_rule(self):
        # The never-swap gate: any swap-in or swap-out in the cell voids it.
        os.environ["BENCH_SWAPIN_VOID_MB"] = "0"
        os.environ["BENCH_SWAPOUT_VOID_MB"] = "0"
        w = bench_peer.SwapWatch()
        w.swapin, w.swapout = 0.0, 0.0
        self.assertIsNone(w.void_reason())
        w.swapout = 0.1
        self.assertIn("swap-outs", w.void_reason())
        w.swapin, w.swapout = 0.1, 0.0
        self.assertIn("swap-ins", w.void_reason())

    def test_stop_reads_both_counters(self):
        real = bench_peer.swap_io_mb
        self.addCleanup(setattr, bench_peer, "swap_io_mb", real)
        readings = iter([(10.0, 20.0), (11.5, 24.0)])
        bench_peer.swap_io_mb = lambda: next(readings)
        os.environ["BENCH_SWAPOUT_VOID_MB"] = "0"
        stats = bench_peer.SwapWatch().stop()
        self.assertEqual((stats["swapin_mb"], stats["swapout_mb"]), (1.5, 4.0))
        self.assertIn("swap-outs", stats["swap_void"])

    def cell(self, void):
        return ((None if void else [1.0], "v" if void else None, {"swap": {"swap_void": "v" if void else None}}), {}, [])

    def test_a_void_cell_reruns_and_lists_it(self):
        os.environ["BENCH_SWAP_VOID_STOP"] = "3"
        results = iter([self.cell(True), self.cell(False)])
        streak = [0]
        result, voids = bench_peer.run_with_swap_voids(lambda: next(results), streak)
        self.assertEqual((result[0][0], voids, streak), ([1.0], ["v"], [0]))

    def test_three_voids_in_a_row_stop_across_cells(self):
        os.environ["BENCH_SWAP_VOID_STOP"] = "3"
        streak = [2]
        with self.assertRaises(SystemExit):
            bench_peer.run_with_swap_voids(lambda: self.cell(True), streak)

    def test_unset_takes_one_attempt(self):
        calls = []
        def run():
            calls.append(1)
            return self.cell(True)
        result, voids = bench_peer.run_with_swap_voids(run, [0])
        self.assertEqual((len(calls), voids), (1, []))
        self.assertEqual(result[0][1], "v")

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
