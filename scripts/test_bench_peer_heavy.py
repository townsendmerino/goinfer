#!/usr/bin/env python3
"""bench_peer.py's heavy-load void (owner rule 2026-10-19; CLAUDE.md "Run budget"): a cell whose start finds this box's load
above half its cores, or its free memory under the fit need plus 2 GB, is void, listed, and re-run in place, never averaged.
Driven through main()'s cell loop with a fake sampler, and checked against a plant (the rule removed: the heavy start's run is
averaged). Run: python3 -B scripts/test_bench_peer_heavy.py"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_peer  # noqa: E402

CLEAN = {"loadavg": [1.0, 1.0, 1.0], "free_mem_mb": 9000, "ncores": 8, "gpu": None}
LOADED = {"loadavg": [4.5, 3.0, 2.0], "free_mem_mb": 9000, "ncores": 8, "gpu": None}
SHORT = {"loadavg": [1.0, 1.0, 1.0], "free_mem_mb": 2500, "ncores": 8, "gpu": None}
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

    def test_free_memory_under_need_plus_2gb(self):
        self.assertIsNone(bench_peer.heavy_void(dict(CLEAN, free_mem_mb=1000 + 2048), 1000))
        self.assertIn("below the fit need", bench_peer.heavy_void(dict(CLEAN, free_mem_mb=1000 + 2047), 1000))
        self.assertIsNone(bench_peer.heavy_void(SHORT, None), "no model load: the memory clause does not apply")

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
        # B41: the void rule reads the number cmd/hostmem prints (scripts/hostmem.py), the fit guard's own; no formula here.
        import hostmem
        saved = hostmem.available_mb
        self.addCleanup(setattr, hostmem, "available_mb", saved)
        hostmem.available_mb = lambda *a, **k: 12345
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
                            ("cell_ctx", lambda: 2048)]:
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
        self.assertEqual([v["void"].split()[0] for v in rec["void_starts"]], ["load", "free"])
        self.assertEqual(rec["machine"]["free_mem_mb"], 9000)
        self.assertEqual((rec["machine"]["fit_need_mb"], rec["machine"]["fit_need_source"]), (1024, "BENCH_FIT_NEED_GB"))

    def test_plant_without_the_rule_a_heavy_cell_is_averaged(self):
        patch(self, "heavy_void", lambda state, need: None)
        rec = self.run_main()
        self.assertEqual(rec["mean"], 50.0, "with the rule removed the heavy start must be run and averaged")
        self.assertNotIn("void_starts", rec)


if __name__ == "__main__":
    unittest.main()
