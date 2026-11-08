#!/usr/bin/env python3
"""scripts/void_rule.py, the one definition of the heavy-load void (Francis, 2026-11-08): a cell may start only with available memory at
least its fit need (no margin) and the load at or under half the cores, and is void if system swap grows at all. Run:
python3 -B scripts/test_void_rule.py"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import void_rule  # noqa: E402

SNAP = {"load1": 1.0, "ncores": 8, "available_mb": 5000, "reclaimable_mb": 100, "swap_mb": 900.0}


def old_rule(snap, need_mb, cap=None):
    """The rule before 2026-11-08, for the plants: the reclaimable reading against the need plus 2 GB."""
    cap = snap["ncores"] / 2 if cap is None else cap
    if snap["load1"] > cap:
        return "load"
    if need_mb and snap["reclaimable_mb"] < need_mb + 2048:
        return "memory"
    return None


class Start(unittest.TestCase):
    def test_available_at_the_need_starts_and_under_it_waits(self):
        self.assertIsNone(void_rule.start_void(dict(SNAP, available_mb=4000), 4000), "exactly the need starts: there is no margin")
        self.assertIn("below the fit need", void_rule.start_void(dict(SNAP, available_mb=3999), 4000))

    def test_the_reclaimable_reading_is_not_a_void_term(self):
        # reclaimable far under the need, available over it: the cell starts (the old rule would have held it)
        self.assertIsNone(void_rule.start_void(dict(SNAP, reclaimable_mb=10, available_mb=4500), 4000))

    def test_load_cap_is_half_the_cores_and_exactly_half_starts(self):
        self.assertIsNone(void_rule.start_void(dict(SNAP, load1=4.0), 0))
        self.assertIn("above half of 8 cores", void_rule.start_void(dict(SNAP, load1=4.01), 0))
        self.assertIn("above half of 16 cores", void_rule.start_void(dict(SNAP, load1=8.5, ncores=16), 0))
        self.assertIsNone(void_rule.start_void(dict(SNAP, load1=8.0, ncores=16), 0))
        self.assertIn("above the cap", void_rule.start_void(dict(SNAP, load1=1.5), 0, cap=1.0), "an explicit cap")

    def test_a_cell_without_a_model_load_has_no_memory_clause_and_unknown_does_not_void(self):
        self.assertIsNone(void_rule.start_void(dict(SNAP, available_mb=1), 0))
        self.assertIsNone(void_rule.start_void(dict(SNAP, available_mb=None), 4000))
        self.assertIsNone(void_rule.start_void(dict(SNAP, load1=None), 4000))

    def test_plant_the_old_rule_holds_a_cell_this_rule_starts(self):
        # A start the new rule allows (available over the need, reclaimable under need + 2 GB) is held by the old rule: restoring the
        # +2 GB term would make a cell that should start wait, and this test is what notices it.
        snap = dict(SNAP, available_mb=4100, reclaimable_mb=4100)
        self.assertIsNone(void_rule.start_void(snap, 4000))
        self.assertIsNotNone(old_rule(snap, 4000), "the plant: the old rule holds this start")
        with mock.patch.object(void_rule, "start_void", old_rule):
            with self.assertRaises(AssertionError):
                self.assertIsNone(void_rule.start_void(snap, 4000))


class UnknownCores(unittest.TestCase):
    def test_an_unreadable_core_count_skips_the_load_term(self):
        self.assertIsNone(void_rule.start_void(dict(SNAP, ncores=None, load1=99.0), 0), "no cap of 0.5 from an unknown core count")
        self.assertIn("above the cap 1", void_rule.start_void(dict(SNAP, ncores=None, load1=99.0), 0, cap=1.0), "an explicit cap still applies")


class Readings(unittest.TestCase):
    """snapshot() and swap_used_mb() with stubbed sources."""

    def test_snapshot_has_the_four_readings(self):
        with mock.patch.object(void_rule.hostmem, "available_mb", lambda: 6000), \
                mock.patch.object(void_rule.hostmem, "reclaimable_mb", lambda: 4000), \
                mock.patch.object(void_rule, "swap_used_mb", lambda: 321.5), \
                mock.patch.object(void_rule.os, "getloadavg", lambda: (2.25, 1.0, 1.0)), \
                mock.patch.object(void_rule.os, "cpu_count", lambda: 10):
            snap = void_rule.snapshot()
        self.assertEqual((snap["load1"], snap["ncores"], snap["available_mb"], snap["reclaimable_mb"], snap["swap_mb"]),
                         (2.25, 10, 6000, 4000, 321.5))

    def test_swap_on_darwin_is_sysctls_used_figure(self):
        out = "total = 2048.00M  used = 1404.50M  free = 643.50M  (encrypted)\n"
        with mock.patch.object(void_rule.sys, "platform", "darwin"), \
                mock.patch.object(void_rule.subprocess, "run", lambda *a, **k: mock.Mock(stdout=out)):
            self.assertEqual(void_rule.swap_used_mb(), 1404.5)
        with mock.patch.object(void_rule.sys, "platform", "darwin"), \
                mock.patch.object(void_rule.subprocess, "run", lambda *a, **k: mock.Mock(stdout="garbage")):
            self.assertIsNone(void_rule.swap_used_mb())

    def test_swap_on_linux_is_total_less_free_from_meminfo(self):
        meminfo = "MemTotal: 1 kB\nSwapTotal: 8388608 kB\nSwapFree: 8126464 kB\n"
        with mock.patch.object(void_rule.sys, "platform", "linux"), mock.patch("builtins.open", mock.mock_open(read_data=meminfo)):
            self.assertEqual(void_rule.swap_used_mb(), 256.0)


class Swap(unittest.TestCase):
    def test_any_growth_voids_and_none_does_not(self):
        self.assertIsNone(void_rule.swap_void(SNAP, dict(SNAP)))
        self.assertIsNone(void_rule.swap_void(SNAP, dict(SNAP, swap_mb=800.0)), "swap that fell is not growth")
        self.assertIn("grew 0.1 MB", void_rule.swap_void(SNAP, dict(SNAP, swap_mb=900.1)))

    def test_an_unreadable_swap_does_not_void(self):
        self.assertIsNone(void_rule.swap_void(SNAP, dict(SNAP, swap_mb=None)))
        self.assertIsNone(void_rule.swap_void(dict(SNAP, swap_mb=None), SNAP))


class Cli(unittest.TestCase):
    def run_cli(self, argv, snap):
        with mock.patch.object(void_rule, "snapshot", lambda: dict(snap)):
            with mock.patch("builtins.print"):
                return void_rule.main(argv)

    def test_start_exits_75_when_void_and_writes_all_four_readings(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "s.json")
            self.assertEqual(self.run_cli(["start", "--need-mb", "4000", "--json-out", out], dict(SNAP, available_mb=3000)), 75)
            rec = json.load(open(out))
            for k in ("load1", "available_mb", "reclaimable_mb", "swap_mb"):
                self.assertIn(k, rec)
            self.assertIn("below the fit need", rec["void"])
            self.assertEqual(self.run_cli(["start", "--need-mb", "4000"], SNAP), 0)

    def test_the_reason_is_lower_case_so_a_retried_cell_does_not_trip_nights_marker_scan(self):
        import io
        from contextlib import redirect_stderr
        err = io.StringIO()
        with mock.patch.object(void_rule, "snapshot", lambda: dict(SNAP, available_mb=3000)), redirect_stderr(err), \
                mock.patch("sys.stdout", io.StringIO()):
            void_rule.main(["start", "--need-mb", "4000"])
        self.assertIn("void: available memory", err.getvalue())
        self.assertNotIn("VOID", err.getvalue())

    def test_end_exits_76_when_swap_grew(self):
        with tempfile.TemporaryDirectory() as d:
            st = os.path.join(d, "s.json")
            json.dump(SNAP, open(st, "w"))
            self.assertEqual(self.run_cli(["end", "--start-json", st], dict(SNAP, swap_mb=950.0)), 76)
            self.assertEqual(self.run_cli(["end", "--start-json", st], SNAP), 0)


if __name__ == "__main__":
    unittest.main()
