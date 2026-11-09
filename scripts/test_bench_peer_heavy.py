#!/usr/bin/env python3
"""bench_peer.py's heavy-load void (owner rule 2026-10-19; CLAUDE.md "Run budget"): a cell whose start finds this box's load
above half its cores, or its available memory (the fit guard's reading) under the fit need, is void, listed, and re-run in place,
never averaged; swap that grows during a cell voids it (void_rule.py, the one definition; Francis, 2026-11-08).
Driven through main()'s cell loop with a fake sampler, and checked against a plant (the rule removed: the heavy start's run is
averaged). Run: python3 -B scripts/test_bench_peer_heavy.py"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_peer  # noqa: E402

CLEAN = {"loadavg": [1.0, 1.0, 1.0], "available_mb": 9000, "ncores": 8, "gpu": None}
LOADED = {"loadavg": [4.5, 3.0, 2.0], "available_mb": 9000, "ncores": 8, "gpu": None}
SHORT = {"loadavg": [1.0, 1.0, 1.0], "available_mb": 900, "ncores": 8, "gpu": None}  # under the 1 GB fit need the MainLoop cell declares
RULE = bench_peer.heavy_void  # the fake run's own judge of a heavy start, untouched by the plant


def patch(tc, name, value):
    old = getattr(bench_peer, name)
    setattr(bench_peer, name, value)
    tc.addCleanup(setattr, bench_peer, name, old)


class Rule(unittest.TestCase):
    def test_load_above_half_the_cores(self):
        self.assertIsNone(bench_peer.heavy_void(dict(CLEAN, loadavg=[4.0, 0, 0]), None), "exactly half is not above half")
        self.assertIn("above half of 8 cores", bench_peer.heavy_void(LOADED, None))
        self.assertIn("above half of 16 cores", bench_peer.heavy_void(dict(LOADED, loadavg=[8.5, 0, 0], ncores=16), None))

    def test_available_memory_under_the_need(self):
        self.assertIsNone(bench_peer.heavy_void(dict(CLEAN, available_mb=1000), 1000), "exactly the need starts: no margin")
        self.assertIn("below the fit need", bench_peer.heavy_void(dict(CLEAN, available_mb=999), 1000))
        self.assertIsNone(bench_peer.heavy_void(SHORT, None), "no model load: the memory clause does not apply")

    def test_an_unreadable_core_count_skips_the_load_term_and_says_so(self):
        import contextlib
        import io
        self.addCleanup(setattr, bench_peer, "_LOAD_SKIP_SAID", bench_peer._LOAD_SKIP_SAID)
        bench_peer._LOAD_SKIP_SAID = False
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertIsNone(bench_peer.heavy_void(dict(CLEAN, loadavg=[40.0, 0, 0], ncores=None), None), "no cap of 0.5 from an unknown core count")
            bench_peer.heavy_void(dict(CLEAN, ncores=None), None)
        self.assertEqual(out.getvalue().count("core count unreadable"), 1, "said once")

    def test_machine_state_and_end_state_carry_the_four_readings(self):
        import hostmem
        for obj, name, value in [(hostmem, "available_mb", lambda *a, **k: 7000), (hostmem, "reclaimable_mb", lambda *a, **k: 3000),
                                 (bench_peer, "swap_used_mb", lambda: 111.5), (bench_peer, "_loadavg", lambda: [1.5, 1.0, 1.0]),
                                 (bench_peer, "_gpu_state", lambda: None), (bench_peer, "thermal_state", lambda: None)]:
            old = getattr(obj, name)
            setattr(obj, name, value)
            self.addCleanup(setattr, obj, name, old)
        m, e = bench_peer.machine_state(), bench_peer.end_state()
        self.assertEqual((m["available_mb"], m["reclaimable_mb"], m["swap_mb"], m["loadavg"][0]), (7000, 3000, 111.5, 1.5))
        self.assertEqual((e["available_mb"], e["reclaimable_mb"], e["swap_mb"], e["loadavg"][0]), (7000, 3000, 111.5, 1.5))
        self.assertNotIn("free_mem_mb", m, "the key holds the available reading and says so")

    def test_a_file_size_need_is_recorded_as_a_lower_bound(self):
        self.addCleanup(os.environ.pop, "BENCH_FIT_NEED_GB", None)
        self.addCleanup(bench_peer._FIT_NEED.clear)
        self.assertTrue(bench_peer.need_is_lower_bound("file size"))
        self.assertFalse(bench_peer.need_is_lower_bound("fit banner"))
        self.assertFalse(bench_peer.need_is_lower_bound("BENCH_FIT_NEED_GB"))
        self.assertFalse(bench_peer.need_is_lower_bound(None))

    def test_an_unreadable_memory_or_load_reading_fails_closed(self):
        self.assertIn("available memory unreadable", bench_peer.heavy_void(dict(CLEAN, available_mb=None), 1000))
        self.assertIn("load average unreadable", bench_peer.heavy_void(dict(CLEAN, loadavg=None), None))
        self.assertIsNone(bench_peer.heavy_void(dict(CLEAN, available_mb=None), None), "no model load: no memory clause")

    def test_the_documented_no_free_memory_override_skips_the_memory_term_and_says_so(self):
        import contextlib
        import io
        self.addCleanup(setattr, bench_peer, "_FREE_MEM_OVERRIDE_USED", bench_peer._FREE_MEM_OVERRIDE_USED)
        self.addCleanup(setattr, bench_peer, "_MEM_SKIP_SAID", bench_peer._MEM_SKIP_SAID)
        bench_peer._MEM_SKIP_SAID = False
        state = dict(CLEAN, available_mb=None)
        bench_peer._FREE_MEM_OVERRIDE_USED = False
        self.assertIn("available memory unreadable", bench_peer.heavy_void(state, 1000), "without the override the guard fails closed")
        bench_peer._FREE_MEM_OVERRIDE_USED = True
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertIsNone(bench_peer.heavy_void(state, 1000))
            bench_peer.heavy_void(state, 1000)
            self.assertIn("above half", bench_peer.heavy_void(dict(state, loadavg=[9.0, 0, 0]), 1000), "the load term still applies")
        self.assertEqual(out.getvalue().count("BENCH_ALLOW_NO_FREE_MEM=1"), 1, "logged once")

    def test_the_cell_end_check_voids_what_the_cell_left_behind(self):
        start = {"swap_mb": 100.0, "swapin_mb": 1.0, "swapout_mb": 1.0, "ncores": 8}
        end = {"loadavg": [1.0, 1, 1], "available_mb": 5000, "reclaimable_mb": 1, "swap_mb": 100.0, "swapin_mb": 1.0, "swapout_mb": 1.0}
        self.assertIsNone(bench_peer.cell_end_void(start, end, 1000))
        self.assertIn("grew +20.0 MB", bench_peer.cell_end_void(start, dict(end, swap_mb=120.0), 1000))
        self.assertIn("swap unreadable", bench_peer.cell_end_void(start, dict(end, swap_mb=None), 1000))
        self.assertIn("swap unreadable", bench_peer.cell_end_void(dict(start, swap_mb=None), end, 1000))
        self.assertIn("load average unreadable at the cell's end", bench_peer.cell_end_void(start, dict(end, loadavg=None), 1000))
        self.assertIn("available memory unreadable at the cell's end", bench_peer.cell_end_void(start, dict(end, available_mb=None), 1000))
        self.assertIsNone(bench_peer.cell_end_void(start, dict(end, available_mb=None), None), "no model load: no memory clause")

    def test_the_cell_end_check_takes_the_watchers_peak_against_the_cells_own_start(self):
        start = {"swap_mb": 100.0, "swapin_mb": 1.0, "swapout_mb": 1.0}
        end = {"loadavg": [1.0, 1, 1], "available_mb": 5000, "swap_mb": 100.0, "swapin_mb": 1.0, "swapout_mb": 1.0}
        self.assertIn("grew +35.0 MB", bench_peer.cell_end_void(start, end, 1000, peak_mb=135.0),
                      "swap that rose between machine_state and SwapWatch's baseline and receded is still a spike over the cell's start")
        self.assertIsNone(bench_peer.cell_end_void(start, end, 1000, peak_mb=100.0))

    def test_apply_cell_end_check_uses_the_raw_peak_so_float_drift_is_not_growth(self):
        old = bench_peer.swap_used_mb
        for name, value in [("swap_used_mb", lambda: 100.6), ("swap_io_mb", lambda: (None, None)), ("free_mem_mb", lambda: 5000),
                            ("_reclaimable_mb", lambda: 1), ("_loadavg", lambda: [1.0, 1.0, 1.0])]:
            o = getattr(bench_peer, name)
            setattr(bench_peer, name, value)
            self.addCleanup(setattr, bench_peer, name, o)
        # swap read 100.6 at machine_state, dipped to 100.4 at SwapWatch's baseline, returned to 100.6: zero growth over the cell's start,
        # but 100.4 + round(0.2, 1) is 100.60000000000001
        counts = {"swap": {"swap_base_mb": 100.4, "swap_growth_mb": 0.2, "swap_peak_mb": 100.6, "swap_void": None}}
        res = (([1.0], None, counts), {"swap_mb": 100.6, "swapin_mb": None, "swapout_mb": None}, [])
        self.assertIsNone(bench_peer.apply_cell_end_check(res, 0)[0][2]["swap"]["swap_void"])
        # a peak above the cell's start voids it, from the same raw figure
        counts = {"swap": {"swap_base_mb": 100.4, "swap_growth_mb": 20.2, "swap_peak_mb": 120.6, "swap_void": None}}
        res = (([1.0], None, counts), {"swap_mb": 100.6, "swapin_mb": None, "swapout_mb": None}, [])
        self.assertIn("grew +20.0 MB", bench_peer.apply_cell_end_check(res, 0)[0][2]["swap"]["swap_void"])

    def test_swapwatch_reports_its_raw_peak(self):
        old = bench_peer.swap_used_mb
        self.addCleanup(setattr, bench_peer, "swap_used_mb", old)
        reads = iter([100.4, 100.6, 100.6])
        bench_peer.swap_used_mb = lambda: next(reads, 100.6)
        w = bench_peer.SwapWatch()
        w.poll_once()
        res = w.stop()
        # exact: 100.4 + round(0.2, 1) is 100.60000000000001, which a rebuilt peak would report
        self.assertEqual(res["swap_peak_mb"], 100.6)

    def test_swapwatch_reports_no_numeric_peak_when_a_reading_failed(self):
        old = bench_peer.swap_used_mb
        self.addCleanup(setattr, bench_peer, "swap_used_mb", old)
        reads = iter([100.0, 130.0, None])
        bench_peer.swap_used_mb = lambda: next(reads, None)
        w = bench_peer.SwapWatch()
        w.poll_once()   # 130
        res = w.stop()  # the last read fails
        self.assertIsNone(res["swap_peak_mb"])
        self.assertIsNone(res["swap_growth_mb"])
        self.assertIn("swap unreadable", res["swap_void"])

    def test_cell_end_void_is_void_rules_end_void(self):
        start = {"swap_mb": 100.0, "swapin_mb": None, "swapout_mb": None, "ncores": None}
        end = {"swap_mb": 100.0, "swapin_mb": None, "swapout_mb": None, "available_mb": 5000, "loadavg": None}
        self.assertIsNone(bench_peer.cell_end_void(start, end, 1000), "no core count at the start: no load term")
        start = dict(start, ncores=8)
        self.assertIn("load average unreadable", bench_peer.cell_end_void(start, end, 1000))
        self.assertIn("load average unreadable", bench_peer.cell_end_void(start, dict(end, loadavg=[float("nan")] * 3), 1000))
        self.assertIn("swap unreadable", bench_peer.cell_end_void(start, dict(end, loadavg=[1.0] * 3, swap_mb=float("inf")), 1000))
        self.assertIn("swap unreadable", bench_peer.cell_end_void(start, dict(end, loadavg=[1.0] * 3), 1000, peak_mb=float("nan")))

    def test_the_end_check_says_when_the_override_skips_memory(self):
        start = {"swap_mb": 100.0, "ncores": 8}
        end = {"swap_mb": 100.0, "available_mb": None, "loadavg": [1.0] * 3}
        out = io.StringIO()
        self.addCleanup(setattr, bench_peer, "_END_MEM_SKIP_SAID", bench_peer._END_MEM_SKIP_SAID)
        bench_peer._END_MEM_SKIP_SAID = False
        with mock.patch.dict(os.environ, {"BENCH_ALLOW_NO_FREE_MEM": "1"}), contextlib.redirect_stdout(out):
            self.assertIsNone(bench_peer.cell_end_void(start, end, 1000))
        self.assertIn("BENCH_ALLOW_NO_FREE_MEM", out.getvalue())

    def test_the_end_check_honours_the_env_override_not_a_latch(self):
        start = {"swap_mb": 100.0}
        end = {"loadavg": [1.0, 1, 1], "available_mb": None, "swap_mb": 100.0}
        self.addCleanup(setattr, bench_peer, "_FREE_MEM_OVERRIDE_USED", bench_peer._FREE_MEM_OVERRIDE_USED)
        bench_peer._FREE_MEM_OVERRIDE_USED = False  # the reading was fine at the start, so the latch never set
        with mock.patch.dict(os.environ, {"BENCH_ALLOW_NO_FREE_MEM": "1"}):
            self.assertIsNone(bench_peer.cell_end_void(start, end, 1000), "the environment setting is the override")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BENCH_ALLOW_NO_FREE_MEM", None)
            self.assertIn("available memory unreadable", bench_peer.cell_end_void(start, end, 1000))

    def test_the_end_check_marks_the_cell_void_in_its_record_so_it_reruns(self):
        for name, value in [("swap_used_mb", lambda: 100.0), ("swap_io_mb", lambda: (None, None)), ("free_mem_mb", lambda: 5000),
                            ("_reclaimable_mb", lambda: 1), ("_loadavg", lambda: [1.0, 1.0, 1.0])]:
            old = getattr(bench_peer, name)
            setattr(bench_peer, name, value)
            self.addCleanup(setattr, bench_peer, name, old)
        counts = {}
        res = (([1.0], None, counts), {"swap_mb": 50.0, "swapin_mb": None, "swapout_mb": None}, [])
        out = bench_peer.apply_cell_end_check(res, 0)
        self.assertIn("grew +50.0 MB", out[0][2]["swap"]["swap_void"])
        self.assertIn("end", out[1])

    def test_provenance_says_a_file_size_need_is_a_lower_bound(self):
        self.assertIn("lower bound", bench_peer.provenance()["fit_need"])

    def test_fit_need_sources(self):
        self.addCleanup(os.environ.pop, "BENCH_FIT_NEED_GB", None)
        self.addCleanup(bench_peer._FIT_NEED.clear)
        bench_peer.learn_fit_need("m", "fit: 32768-token cap needs 1.1 GB KV (weights 3.9 GB, budget 38.5 GB, 37.4 GB left)")
        self.assertEqual(bench_peer.fit_need("m"), (5120, "fit banner"))
        os.environ["BENCH_FIT_NEED_GB"] = "other=2,m=5.6"
        self.assertEqual(bench_peer.fit_need("m"), (5734, "BENCH_FIT_NEED_GB"))
        os.environ["BENCH_FIT_NEED_GB"] = "3"
        self.assertEqual(bench_peer.fit_need("anything"), (3072, "BENCH_FIT_NEED_GB"))

    def test_free_memory_is_goinfers_reading(self):
        # The void rule reads the fit guard's own available figure (cmd/hostmem via scripts/hostmem.py); the reclaimable one is only recorded.
        import hostmem
        saved, saved_r = hostmem.available_mb, hostmem.reclaimable_mb
        self.addCleanup(setattr, hostmem, "available_mb", saved)
        self.addCleanup(setattr, hostmem, "reclaimable_mb", saved_r)
        hostmem.available_mb = lambda *a, **k: 12345
        hostmem.reclaimable_mb = lambda *a, **k: 1
        self.assertEqual(bench_peer.free_mem_mb(), 12345)

    def test_refuses_when_every_start_is_void(self):
        with self.assertRaises(SystemExit):
            bench_peer.start_clean(lambda: LOADED, None, lambda: ([1.0], None, None), wait=lambda s: None, limit_s=40)


class MainLoop(unittest.TestCase):
    """One cell through main(): the first two starts are heavy (load, then memory), the third clean. The fake run returns 50 tok/s
    when the box is heavy at its start and 100 when it is clean, so a run taken from a heavy start shows in the cell's mean."""

    def run_main(self):
        starts = [LOADED, SHORT, CLEAN]
        now = {"state": None}

        def sample():
            now["state"] = dict(starts.pop(0) if starts else CLEAN)
            return now["state"]

        def fake_cell(engine, mk, depth, cfg, backend):
            heavy = RULE(now["state"], 1024) is not None
            return ([50.0, 50.0] if heavy else [100.0, 100.0]), None, {}

        for name, value in [("machine_state", sample), ("run_cell", fake_cell), ("preflight", lambda: None),
                            ("provenance", lambda: {"kind": "provenance"}), ("gate_cell_idle", lambda: {"kind": "test"}),
                            ("plan_models", lambda: ["m"]), ("plan_engines", lambda: ["goinfer"]),
                            ("plan_backends", lambda: ["cpu"]), ("plan_depths", lambda: []), ("plan_configs", lambda: []),
                            ("plan_embed_lengths", lambda: []), ("plan_vision", lambda: False),
                            ("quant_mismatch", lambda *a: ""), ("goinfer_quant", lambda mk: ("int4", None)),
                            ("prompt_tokens", lambda d, mk: 128), ("prompt_format", lambda d, mk: "test"),
                            ("cell_ctx", lambda: 2048),
                            # the fake model's phase-A prompt, which main()'s missing-prompt check asks for before any cell runs
                            ("_PROMPTS", dict(bench_peer._PROMPTS, **{"m:128": {"text": "test", "tokens": 128}}))]:
            patch(self, name, value)
        os.environ["BENCH_FIT_NEED_GB"] = "1"
        os.environ["BENCH_VOID_WAIT"] = "0"
        self.addCleanup(os.environ.pop, "BENCH_FIT_NEED_GB", None)
        self.addCleanup(os.environ.pop, "BENCH_VOID_WAIT", None)
        d = tempfile.mkdtemp()
        out = os.path.join(d, "r.json")
        patch(self, "SERVE_LOG_DIR", d)
        old_argv = sys.argv
        sys.argv = ["bench_peer.py", out]
        self.addCleanup(setattr, sys, "argv", old_argv)
        bench_peer.main()
        return [r for r in json.load(open(out)) if r.get("phase")][0]

    def test_a_heavy_start_is_listed_and_the_cell_reruns_clean(self):
        rec = self.run_main()
        self.assertEqual(rec["mean"], 100.0, "the cell's rates came from a heavy start")
        self.assertEqual([v["void"].split()[0] for v in rec["void_starts"]], ["load", "available"])
        self.assertEqual(rec["machine"]["available_mb"], 9000)
        for k in ("loadavg", "available_mb", "reclaimable_mb", "swap_mb"):
            self.assertIn(k, rec["machine"]["end"], "the cell records the four readings at its end too")
        self.assertEqual((rec["machine"]["fit_need_mb"], rec["machine"]["fit_need_source"]), (1024, "BENCH_FIT_NEED_GB"))

    def test_plant_without_the_rule_a_heavy_cell_is_averaged(self):
        patch(self, "heavy_void", lambda state, need: None)
        rec = self.run_main()
        self.assertEqual(rec["mean"], 50.0, "with the rule removed the heavy start must be run and averaged")
        self.assertNotIn("void_starts", rec)


if __name__ == "__main__":
    unittest.main()
