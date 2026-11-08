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

    def test_a_cell_without_a_model_load_has_no_memory_clause(self):
        self.assertIsNone(void_rule.start_void(dict(SNAP, available_mb=1), 0))
        self.assertIsNone(void_rule.start_void(dict(SNAP, available_mb=None), 0), "no model load: the memory reading is not needed")

    def test_an_unreadable_reading_fails_closed_with_its_reason(self):
        # a guard that cannot read what it guards must not pass the cell (CLAUDE.md, "a guard that inverts under its own condition")
        self.assertIn("available memory unreadable", void_rule.start_void(dict(SNAP, available_mb=None), 4000))
        self.assertIn("load average unreadable", void_rule.start_void(dict(SNAP, load1=None), 0))


class UnknownCores(unittest.TestCase):
    def test_an_unreadable_core_count_skips_the_load_term(self):
        self.assertIsNone(void_rule.start_void(dict(SNAP, ncores=None, load1=99.0), 0), "no cap of 0.5 from an unknown core count")
        self.assertIn("above the cap 1", void_rule.start_void(dict(SNAP, ncores=None, load1=99.0), 0, cap=1.0), "an explicit cap still applies")


class SnapshotCores(unittest.TestCase):
    def snap(self, cpu_count, getloadavg=lambda: (99.0, 1.0, 1.0)):
        with mock.patch.object(void_rule.hostmem, "available_mb", lambda: 6000), \
                mock.patch.object(void_rule.hostmem, "reclaimable_mb", lambda: 4000), \
                mock.patch.object(void_rule, "swap_used_mb", lambda: 1.0), \
                mock.patch.object(void_rule.os, "getloadavg", getloadavg), \
                mock.patch.object(void_rule.os, "cpu_count", lambda: cpu_count):
            return void_rule.snapshot()

    def test_an_unreadable_core_count_stays_unknown_through_snapshot_and_skips_the_load_term(self):
        snap = self.snap(None)
        self.assertIsNone(snap["ncores"], "snapshot must not coerce an unknown core count to 1 (a cap of 0.5)")
        self.assertIsNone(void_rule.start_void(snap, 0))

    def test_an_unreadable_load_is_none_and_voids_the_start(self):
        def boom():
            raise OSError("no load average")
        snap = self.snap(8, boom)
        self.assertIsNone(snap["load1"])
        self.assertIn("load average unreadable", void_rule.start_void(snap, 0))

    def test_the_cli_says_when_it_skips_the_load_term(self):
        import io
        from contextlib import redirect_stderr
        err = io.StringIO()
        with mock.patch.object(void_rule, "snapshot", lambda: dict(SNAP, ncores=None, load1=99.0)), redirect_stderr(err), \
                mock.patch("sys.stdout", io.StringIO()):
            self.assertEqual(void_rule.main(["start", "--need-mb", "0"]), 0)
        self.assertIn("core count unreadable", err.getvalue())


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


class SwapVerdict(unittest.TestCase):
    """swap_verdict is the one swap bar bench_peer's SwapWatch and the shell runners share: any growth voids unless an explicit bar
    replaces it, and nobara's swap-in/out bars (peer-vetted 2026-10-08) replace the growth bar."""

    def env(self, **kw):
        return dict(kw)

    def test_any_growth_voids_with_nothing_set(self):
        self.assertIsNone(void_rule.swap_verdict(0.0, None, None, {}))
        self.assertIn("grew +0.5 MB", void_rule.swap_verdict(0.5, None, None, {}))

    def test_an_explicit_growth_bar_replaces_the_zero(self):
        self.assertIsNone(void_rule.swap_verdict(200.0, None, None, {"BENCH_SWAP_VOID_MB": "256"}))
        self.assertIn("grew", void_rule.swap_verdict(300.0, None, None, {"BENCH_SWAP_VOID_MB": "256"}))

    def test_nobaras_swap_in_bar_replaces_the_default_and_growth_alone_is_not_a_void(self):
        env = {"BENCH_SWAPIN_VOID_MB": "0"}
        self.assertIsNone(void_rule.swap_verdict(46.0, 0.0, 0.0, env))
        self.assertIn("swap-ins", void_rule.swap_verdict(46.0, 0.3, 0.0, env))
        self.assertIn("swap-in count unreadable", void_rule.swap_verdict(46.0, None, None, env), "the bar in force cannot be read: void")

    def test_the_swap_out_bar(self):
        env = {"BENCH_SWAPOUT_VOID_MB": "0"}
        self.assertIn("swap-outs", void_rule.swap_verdict(0.0, 0.0, 2.0, env))


class SwapSamplerPeak(unittest.TestCase):
    """The in-cell check samples swap while the cell runs and votes on the PEAK, so a spike that is freed again before the cell ends
    still voids it (Francis's rule: swap growing at all during the cell)."""

    def test_a_spike_that_settles_voids_though_the_end_equals_the_start(self):
        readings = iter([100.0, 100.0, 180.0, 100.0, 100.0])
        sampler = void_rule.SwapSampler(read=lambda: next(readings), poll_s=0)
        sampler.sample()           # the baseline, 100
        for _ in range(4):
            sampler.sample()
        res = sampler.result()
        self.assertEqual((res["base_mb"], res["peak_growth_mb"]), (100.0, 80.0))
        self.assertIn("grew +80.0 MB", void_rule.swap_verdict(res["peak_growth_mb"], None, None, {}))

    def test_flat_or_falling_swap_has_no_growth(self):
        readings = iter([100.0, 90.0, 100.0])
        sampler = void_rule.SwapSampler(read=lambda: next(readings), poll_s=0)
        for _ in range(3):
            sampler.sample()
        self.assertEqual(sampler.result()["peak_growth_mb"], 0.0)

    def test_an_unreadable_swap_has_no_result(self):
        sampler = void_rule.SwapSampler(read=lambda: None, poll_s=0)
        sampler.sample()
        self.assertEqual(sampler.result(), {"base_mb": None, "peak_mb": None, "peak_growth_mb": None, "unreadable": True})

    def test_a_reading_that_fails_mid_cell_marks_the_result_unreadable(self):
        readings = iter([100.0, None, 100.0])
        sampler = void_rule.SwapSampler(read=lambda: next(readings), poll_s=0)
        for _ in range(3):
            sampler.sample()
        self.assertTrue(sampler.result()["unreadable"])

    def test_a_reader_that_raises_marks_the_result_unreadable(self):
        def boom():
            raise RuntimeError("sysctl failed")
        sampler = void_rule.SwapSampler(read=boom, poll_s=0, base=100.0)
        sampler.sample()
        self.assertTrue(sampler.result()["unreadable"])

    def test_the_end_verdict_voids_on_an_unreadable_peak_file(self):
        with tempfile.TemporaryDirectory() as d:
            st, pk = os.path.join(d, "s.json"), os.path.join(d, "p.json")
            json.dump(SNAP, open(st, "w"))
            json.dump({"base_mb": 900.0, "peak_mb": 900.0, "peak_growth_mb": 0.0, "unreadable": True}, open(pk, "w"))
            with mock.patch.object(void_rule, "snapshot", lambda: dict(SNAP)), mock.patch("builtins.print"):
                self.assertEqual(void_rule.main(["end", "--start-json", st, "--peak-json", pk]), 76)

    def test_observe_folds_a_reading_in_without_reading_swap(self):
        sampler = void_rule.SwapSampler(read=lambda: 1 / 0, poll_s=0, base=100.0)
        sampler.observe(160.0)
        sampler.observe(120.0)
        self.assertEqual(sampler.result(), {"base_mb": 100.0, "peak_mb": 160.0, "peak_growth_mb": 60.0, "unreadable": False})

    def test_a_given_baseline_is_the_one_growth_is_measured_from(self):
        readings = iter([130.0, 120.0])
        sampler = void_rule.SwapSampler(read=lambda: next(readings), poll_s=0, base=100.0)
        sampler.sample()
        sampler.sample()
        self.assertEqual(sampler.result(), {"base_mb": 100.0, "peak_mb": 130.0, "peak_growth_mb": 30.0, "unreadable": False})


class SwapIO(unittest.TestCase):
    """swap_io_mb() with stubbed sources, and swap_void's swap-in/out deltas."""

    VM_STAT = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nSwapins:                         64.\nSwapouts:                       128.\n"

    def test_darwin_reads_vm_stat_at_its_page_size(self):
        with mock.patch.object(void_rule.sys, "platform", "darwin"), \
                mock.patch.object(void_rule.subprocess, "run", lambda *a, **k: mock.Mock(stdout=self.VM_STAT)):
            self.assertEqual(void_rule.swap_io_mb(), (1.0, 2.0))

    def test_linux_reads_pswpin_and_pswpout_at_4k_pages(self):
        data = "pgfault 1\npswpin 256\npswpout 512\n"
        with mock.patch.object(void_rule.sys, "platform", "linux"), mock.patch("builtins.open", mock.mock_open(read_data=data)):
            self.assertEqual(void_rule.swap_io_mb(), (1.0, 2.0))

    def test_an_unreadable_source_is_none_none(self):
        with mock.patch.object(void_rule.sys, "platform", "darwin"), \
                mock.patch.object(void_rule.subprocess, "run", mock.Mock(side_effect=OSError("no vm_stat"))):
            self.assertEqual(void_rule.swap_io_mb(), (None, None))

    def test_swap_void_takes_the_in_and_out_deltas(self):
        env = {"BENCH_SWAPIN_VOID_MB": "0", "BENCH_SWAPOUT_VOID_MB": "1"}
        a = dict(SNAP, swapin_mb=10.0, swapout_mb=20.0)
        self.assertIsNone(void_rule.swap_void(a, dict(a), env=env), "no swap-ins or swap-outs in the cell")
        self.assertIn("swap-ins of 0.50 MB", void_rule.swap_void(a, dict(a, swapin_mb=10.5), env=env))
        self.assertIsNone(void_rule.swap_void(a, dict(a, swapout_mb=20.5), env=env), "under the swap-out bar of 1 MB")
        self.assertIn("swap-outs of 2.00 MB", void_rule.swap_void(a, dict(a, swapout_mb=22.0), env=env))
        self.assertIn("swap-in count unreadable", void_rule.swap_void(a, dict(a, swapin_mb=None), env=env))


class WatchCli(unittest.TestCase):
    """`void_rule.py watch`: the baseline is taken, and the ready file written, before the runner proceeds; a stop that arrives at once
    still leaves a result."""

    def test_the_ready_file_comes_after_the_baseline_and_the_result_after_the_stop(self):
        import threading
        with tempfile.TemporaryDirectory() as d:
            ready, out = os.path.join(d, "ready"), os.path.join(d, "peak.json")
            readings = iter([100.0, 100.0, 140.0, 100.0, 100.0, 100.0, 100.0, 100.0])
            sampler = void_rule.SwapSampler(read=lambda: next(readings, 100.0), poll_s=0.01)
            stop = threading.Event()
            t = threading.Thread(target=void_rule.watch, args=(out, ready, sampler, stop), daemon=True)
            self.addCleanup(stop.set)  # a failed assertion must not leave the watcher waiting forever
            t.start()
            for _ in range(500):
                if os.path.exists(ready):
                    break
                threading.Event().wait(0.01)
            self.assertTrue(os.path.exists(ready), "watch never signalled that its baseline was taken")
            self.assertEqual(json.load(open(ready))["base_mb"], 100.0)
            threading.Event().wait(0.1)
            stop.set()
            t.join(5)
            res = json.load(open(out))
            self.assertEqual((res["base_mb"], res["peak_mb"], res["peak_growth_mb"]), (100.0, 140.0, 40.0))

    def test_a_stop_before_the_first_sample_still_has_a_baseline(self):
        import threading
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "peak.json")
            sampler = void_rule.SwapSampler(read=lambda: 77.0, poll_s=0.01)
            stop = threading.Event()
            stop.set()
            void_rule.watch(out, None, sampler, stop)
            self.assertEqual(json.load(open(out))["base_mb"], 77.0)


class AtomicFiles(unittest.TestCase):
    def test_a_json_file_appears_whole_or_not_at_all(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.json")
            seen = []
            real_replace = os.replace

            def spy(src, dst):
                seen.append((os.path.exists(dst), json.load(open(src))))  # the temp file is complete before it is renamed in
                return real_replace(src, dst)
            with mock.patch.object(void_rule.os, "replace", spy):
                void_rule.write_json_atomic(p, {"a": 1})
            self.assertEqual(seen, [(False, {"a": 1})])
            self.assertEqual(json.load(open(p)), {"a": 1})
            self.assertEqual(os.listdir(d), ["x.json"], "no temp file is left behind")

    def test_wait_ready_waits_for_a_whole_file_and_times_out_otherwise(self):
        import threading
        with tempfile.TemporaryDirectory() as d:
            r = os.path.join(d, "ready")
            threading.Timer(0.1, lambda: void_rule.write_json_atomic(r, {"base_mb": 5.0})).start()
            self.assertEqual(void_rule.wait_ready(r, timeout_s=5, poll_s=0.01), {"base_mb": 5.0})
            self.assertIsNone(void_rule.wait_ready(os.path.join(d, "never"), timeout_s=0.1, poll_s=0.01))
            partial = os.path.join(d, "partial")
            open(partial, "w").write('{"base_')
            self.assertIsNone(void_rule.wait_ready(partial, timeout_s=0.1, poll_s=0.01), "a half-written file is not ready")

    def test_the_wait_command_exits_77_when_the_file_never_appears(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(void_rule.main(["wait", "--ready-file", os.path.join(d, "never"), "--timeout-s", "0.1"]), 77)

    def test_watch_joins_its_thread_before_it_reads_the_result(self):
        import threading
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "peak.json")
            sampler = void_rule.SwapSampler(read=lambda: 5.0, poll_s=0.001)
            stop = threading.Event()
            stop.set()
            void_rule.watch(out, None, sampler, stop)
            self.assertIsNotNone(sampler._thread, "watch must hand its thread to the sampler so stop() joins it")
            self.assertFalse(sampler._thread.is_alive())


class Allowance(unittest.TestCase):
    """BENCH_ALLOW_NO_FREE_MEM=1 (the documented override for a box with no memory reading) skips the memory term, and says so."""

    def test_start_void_can_skip_an_unreadable_memory_term_only_when_allowed(self):
        self.assertIn("available memory unreadable", void_rule.start_void(dict(SNAP, available_mb=None), 4000))
        self.assertIsNone(void_rule.start_void(dict(SNAP, available_mb=None), 4000, allow_unreadable_memory=True))
        self.assertIn("above half", void_rule.start_void(dict(SNAP, available_mb=None, load1=9.0), 4000, allow_unreadable_memory=True),
                      "the load term still applies")

    def test_the_cli_honours_the_env_override_and_logs_it(self):
        import io
        from contextlib import redirect_stderr
        err = io.StringIO()
        with mock.patch.dict(os.environ, {"BENCH_ALLOW_NO_FREE_MEM": "1"}), \
                mock.patch.object(void_rule, "snapshot", lambda: dict(SNAP, available_mb=None)), redirect_stderr(err), \
                mock.patch("sys.stdout", io.StringIO()):
            self.assertEqual(void_rule.main(["start", "--need-mb", "4000"]), 0)
        self.assertIn("BENCH_ALLOW_NO_FREE_MEM=1", err.getvalue())


class Swap(unittest.TestCase):
    def test_any_growth_voids_and_none_does_not(self):
        self.assertIsNone(void_rule.swap_void(SNAP, dict(SNAP), env={}))
        self.assertIsNone(void_rule.swap_void(SNAP, dict(SNAP, swap_mb=800.0), env={}), "swap that fell is not growth")
        self.assertIn("grew +0.1 MB", void_rule.swap_void(SNAP, dict(SNAP, swap_mb=900.1), env={}))

    def test_an_unreadable_swap_voids_with_its_reason(self):
        self.assertIn("swap unreadable", void_rule.swap_void(SNAP, dict(SNAP, swap_mb=None), env={}))
        self.assertIn("swap unreadable", void_rule.swap_void(dict(SNAP, swap_mb=None), SNAP, env={}))
        # a sampled peak does not stand in for an end reading that failed: the end is part of the cell's record
        self.assertIn("swap unreadable", void_rule.swap_void(SNAP, dict(SNAP, swap_mb=None), 900.0, {}))

    def test_a_peak_that_was_asked_for_and_is_unreadable_voids(self):
        self.assertIn("swap unreadable", void_rule.swap_void(SNAP, dict(SNAP), None, {}, peak_required=True))
        self.assertIsNone(void_rule.swap_void(SNAP, dict(SNAP), 900.0, {}, peak_required=True))
        self.assertIsNone(void_rule.swap_void(SNAP, dict(SNAP), None, {}), "no watcher was asked for: the end against the start decides")


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

    def test_end_compares_the_sampled_peak_with_the_cells_start_reading(self):
        with tempfile.TemporaryDirectory() as d:
            st, pk = os.path.join(d, "s.json"), os.path.join(d, "p.json")
            json.dump(SNAP, open(st, "w"))  # the cell started with 900 MB of swap
            json.dump({"base_mb": 900.0, "peak_mb": 960.0, "peak_growth_mb": 60.0}, open(pk, "w"))
            self.assertEqual(self.run_cli(["end", "--start-json", st, "--peak-json", pk], SNAP), 76, "the end equals the start; the peak does not")
            # the watcher started earlier and saw 100 MB, so its own growth is 800; against the cell's start (900) the peak is no growth
            json.dump({"base_mb": 100.0, "peak_mb": 900.0, "peak_growth_mb": 800.0}, open(pk, "w"))
            self.assertEqual(self.run_cli(["end", "--start-json", st, "--peak-json", pk], SNAP), 0,
                             "the peak is read against the cell's start baseline, not the watcher's")

    def test_start_records_a_file_size_need_as_a_lower_bound(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "s.json")
            self.run_cli(["start", "--need-mb", "10", "--need-source", "file size", "--json-out", out], SNAP)
            self.assertTrue(json.load(open(out))["need_is_lower_bound"])
            self.run_cli(["start", "--need-mb", "10", "--need-source", "fit banner", "--json-out", out], SNAP)
            self.assertFalse(json.load(open(out))["need_is_lower_bound"])

    def test_end_exits_76_when_swap_grew(self):
        with tempfile.TemporaryDirectory() as d:
            st = os.path.join(d, "s.json")
            json.dump(SNAP, open(st, "w"))
            self.assertEqual(self.run_cli(["end", "--start-json", st], dict(SNAP, swap_mb=950.0)), 76)
            self.assertEqual(self.run_cli(["end", "--start-json", st], SNAP), 0)


if __name__ == "__main__":
    unittest.main()
