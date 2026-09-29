#!/usr/bin/env python3
"""bench_peer.py's BENCH_ABBA plan transform (TE4, te4-seq-v2-2026-09-28.md §1.1). Run: python3 -B scripts/test_bench_peer_abba.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_peer  # noqa: E402  (import has no side effects: main() runs only under __main__)

PLAN = [("A", "goinfer", "cuda", "0.5B", 128, "greedy"), ("A", "goinfer_old", "cuda", "0.5B", 128, "greedy"),
        ("A", "goinfer", "cuda", "1.5B", 128, "greedy"), ("A", "goinfer_old", "cuda", "1.5B", 128, "greedy"),
        ("E", "goinfer", "vision", "vision", 0, "greedy")]


class ABBA(unittest.TestCase):
    def test_off_keeps_the_plan_and_adds_no_block(self):
        self.assertEqual(bench_peer.abba_plan(PLAN, 0), [p + (None,) for p in PLAN])

    def test_quads_mirror_and_group(self):
        out = bench_peer.abba_plan(PLAN, 2)
        first = [(e, b) for (_, e, _, mk, _, _, b) in out if mk == "0.5B"]
        self.assertEqual(first, [("goinfer", 0), ("goinfer_old", 0), ("goinfer_old", 1), ("goinfer", 1),
                                 ("goinfer", 2), ("goinfer_old", 2), ("goinfer_old", 3), ("goinfer", 3)])
        # A group finishes all its quads before the next starts.
        models = [p[3] for p in out if p[0] == "A"]
        self.assertEqual(models, ["0.5B"] * 8 + ["1.5B"] * 8)

    def test_keys_are_unique_so_resume_skips_exactly_the_done_cells(self):
        out = bench_peer.abba_plan(PLAN, 3)
        self.assertEqual(len(out), len(set(out)))

    def test_every_block_holds_every_arm_once(self):
        out = bench_peer.abba_plan(PLAN, 2)
        for mk in ("0.5B", "1.5B"):
            for b in range(4):
                engs = sorted(p[1] for p in out if p[3] == mk and p[6] == b)
                self.assertEqual(engs, ["goinfer", "goinfer_old"])

    def test_unpaired_phases_pass_through(self):
        out = bench_peer.abba_plan(PLAN, 2)
        self.assertIn(("E", "goinfer", "vision", "vision", 0, "greedy", None), out)


if __name__ == "__main__":
    unittest.main()
