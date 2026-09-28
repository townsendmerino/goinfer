#!/usr/bin/env python3
"""Tests for power.py (TE3, docs/tasks/task-test-efficiency-2026-09.md): the N a gate needs, from the noise registry.

The cases that must be able to go red:
  - textbook sample sizes (the normal formula exactly, and the exact-t N against G*Power's published values);
  - the D6a amendment's +/-1.5-point SE (2852eaa1), reproduced from 400 noul rows near p 0.9;
  - the "cannot resolve" branches (margin inside the floor; N past the feasible cap);
  - a registry lookup miss, which must refuse rather than fall back to a guess;
  - the committed registry itself: every entry's sd recomputes from its recorded values.
Run directly:
    python3 scripts/test_power.py
"""
import io
import json
import math
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import power  # noqa: E402


def run(argv):
    out = io.StringIO()
    err = io.StringIO()
    saved = sys.stderr
    sys.stderr = err
    try:
        rc = power.main(argv, out=out)
    finally:
        sys.stderr = saved
    return rc, out.getvalue(), err.getvalue()


class Distributions(unittest.TestCase):
    def test_normal_quantiles(self):
        self.assertAlmostEqual(power.z(0.975), 1.959964, places=6)
        self.assertAlmostEqual(power.z(0.8), 0.841621, places=6)

    def test_t_quantiles_match_tables(self):
        # Student's t table: t_{0.975, df}
        for df, want in ((1, 12.7062), (2, 4.3027), (5, 2.5706), (10, 2.2281), (30, 2.0423)):
            self.assertAlmostEqual(power.t_ppf(0.975, df), want, places=4)
        self.assertAlmostEqual(power.t_ppf(0.8, 10), 0.8791, places=4)

    def test_noncentral_t_reduces_to_central(self):
        for df in (1, 3, 10, 50):
            self.assertAlmostEqual(power.nct_sf(2.0, df, 0.0), 1 - power.t_cdf(2.0, df), places=7)


class TextbookSampleSizes(unittest.TestCase):
    """Two-sided alpha 0.05, power 0.8, sd 1, so the margin is Cohen's d."""

    def test_normal_formula(self):
        # n = (z_0.975 + z_0.8)^2 / d^2 = 7.8489 / d^2 per paired/one-sample test, twice that per arm for two samples
        k = (1.959964 + 0.841621) ** 2
        self.assertAlmostEqual(power.n_normal(1.0, 1.0, 0.05, 0.8, True, False), k, places=3)
        self.assertAlmostEqual(power.n_normal(1.0, 0.5, 0.05, 0.8, True, False), 31.395, places=2)
        self.assertAlmostEqual(power.n_normal(1.0, 0.5, 0.05, 0.8, False, False), 62.791, places=2)
        # one-sided: (z_0.95 + z_0.8)^2 = 6.1826
        self.assertAlmostEqual(power.n_normal(1.0, 1.0, 0.05, 0.8, True, True), 6.1826, places=3)

    def test_exact_t_against_gpower(self):
        # G*Power 3.1: one-sample (paired) t, d=0.5 -> 34 (actual power 0.8078); d=0.8 -> 15; d=0.2 -> 199.
        # Two independent means, d=0.5 -> 64 per group (0.8015); d=0.8 -> 26 (0.8075); d=0.2 -> 394 (Cohen 1988's
        # normal-approximation table prints 393 for the last).
        for d, paired, want in ((0.5, True, 34), (0.8, True, 15), (0.2, True, 199),
                                (0.5, False, 64), (0.8, False, 26), (0.2, False, 394)):
            n, _ = power.solve_n(1.0, d, 0.05, 0.8, paired=paired)
            self.assertEqual(n, want, "d=%s paired=%s" % (d, paired))
        self.assertAlmostEqual(power.power_at(34, 1.0, 0.5, 0.05, True, False), 0.8078, places=3)
        self.assertAlmostEqual(power.power_at(64, 1.0, 0.5, 0.05, False, False), 0.8015, places=3)

    def test_n_scales_with_sd_over_margin(self):
        # the gate form: sd 0.01 against a 3% bar at an expected ratio of 1.0 is d = ln(1/0.97)/0.01 = 3.05
        rc, out, _ = run(["served-decode", "x/y/z/d1", "--bar", "0.97", "--paired", "--sd", "0.01"])
        self.assertEqual(rc, 0, out)
        n, _ = power.solve_n(0.01, math.log(1 / 0.97), paired=True)
        self.assertIn("N = %d paired blocks" % n, out)


class Evaluation(unittest.TestCase):
    def test_d6a_amendment_se_reproduced(self):
        # 2852eaa1: "estimated here from 400 rows. Its standard error near p ~ 0.9 is about +/-1.5 points"
        self.assertAlmostEqual(power.binomial_se(0.9, 400), 0.015, places=6)
        self.assertEqual(power.binomial_n(0.9, 0.015, None), 400)
        rc, out, _ = run(["binomial", "--p", "0.9", "--half-width", "0.015"])
        self.assertEqual(rc, 0)
        self.assertIn("N = 400 for it to be one SE", out)
        self.assertIn("N = 1537 for it to be the 95% half-width", out)
        rc, out, _ = run(["binomial", "--p", "0.9", "--n", "400"])
        self.assertIn("SE = 0.0150 (1.50 points)", out)
        self.assertIn("half-width = 0.0294 (2.94 points)", out)

    def test_d6a_kind_weighted_se(self):
        # the kind-weighted top-1 over the OOD split's proportions 9,767 / 3,219 / 72 at the amended 400 / 400 / 72 rows
        w = [(9767, 400, 0.9), (3219, 400, 0.9), (72, 72, 0.9)]
        se = power.stratified_se(w)
        want = math.sqrt(sum((a / 13058.0) ** 2 * 0.09 / n for a, n, _ in w))
        self.assertAlmostEqual(se, want, places=12)
        self.assertAlmostEqual(se, 0.01181, places=5)
        rc, out, _ = run(["binomial", "--p", "0.9", "--stratum", "9767:400", "--stratum", "3219:400", "--stratum", "72:72"])
        self.assertEqual(rc, 0)
        self.assertIn("SE of the weighted proportion = 0.0118", out)

    def test_ece_floor_formula(self):
        # half-normal mean: E|N(0, s^2)| = s sqrt(2/pi); B equal bins of N/B rows at accuracy p
        p, b, n = 0.9, 15, 872
        want = math.sqrt(2 / math.pi) * math.sqrt(p * (1 - p) * b / n)
        self.assertAlmostEqual(power.ece_floor(p, b, n), want, places=12)
        # and the inverse gives an N at or just past that floor
        nf = power.ece_n(p, b, 0.02)
        self.assertLessEqual(power.ece_floor(p, b, nf), 0.02)
        self.assertGreater(power.ece_floor(p, b, nf - 1), 0.02)
        lo, hi = power.ece_sd_bounds(p, n)
        self.assertLess(lo, hi)
        self.assertAlmostEqual(hi, math.sqrt(p * (1 - p) / n), places=12)

    def test_ece_floor_monte_carlo(self):
        # a perfectly calibrated model: every row's confidence is its bin's accuracy; the plug-in ECE's mean over
        # replicates should sit near the formula (equal bins, so the formula is the exact large-n limit)
        import random
        rng = random.Random(20260928)
        p, b, n, reps = 0.9, 4, 800, 400
        per = n // b
        tot = 0.0
        for _ in range(reps):
            e = 0.0
            for _ in range(b):
                hits = sum(1 for _ in range(per) if rng.random() < p)
                e += (per / n) * abs(hits / per - p)
            tot += e
        mc = tot / reps
        self.assertAlmostEqual(mc, power.ece_floor(p, b, n), delta=0.1 * power.ece_floor(p, b, n))


class CannotResolve(unittest.TestCase):
    def test_margin_inside_the_floor(self):
        # an expected 1.00 against a 0.97 bar is a 0.0305 margin; a 0.035 floor swallows it
        rc, out, _ = run(["served-decode", "x/y/z/d1", "--bar", "0.97", "--paired", "--sd", "0.01", "--floor", "0.035"])
        self.assertEqual(rc, power.EXIT_UNRESOLVABLE)
        self.assertIn("CANNOT RESOLVE at any N", out)

    def test_past_the_feasible_cap(self):
        rc, out, _ = run(["served-decode", "x/y/z/d1", "--bar", "0.99", "--paired", "--sd", "0.05", "--max-n", "12"])
        self.assertEqual(rc, power.EXIT_UNRESOLVABLE)
        self.assertIn("CANNOT RESOLVE at a feasible N (max N 12", out)
        self.assertIn("Change the instrument", out)

    def test_worst_case_charges_the_floor_twice(self):
        base = ["served-decode", "x/y/z/d1", "--bar", "0.97", "--paired", "--sd", "0.01", "--floor", "0.016"]
        rc1, out1, _ = run(base)
        rc2, out2, _ = run(base + ["--worst-case"])
        self.assertEqual(rc1, 0, out1)
        self.assertEqual(rc2, power.EXIT_UNRESOLVABLE, out2)


class Registry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "reg.json")
        reg = {
            "instruments": {"served-decode": {"max_n": 20}, "inproc-kernel": {"max_n": 60}},
            "entries": [
                {"id": "aa-pooled", "instrument": "served-decode",
                 "cell": {"machine": "nobara", "backend": "cuda", "model": "*", "depth": "*"},
                 "provenance": "direct-aa", "sd": 0.008, "sd_kind": "paired_log_ratio", "floor": 0.0, "n": 20,
                 "statistic": "RMS", "unit": "block", "estimator": "rms_log_about_0",
                 "values": [1.008, 0.992], "source": [{"path": "x.md", "line": 1}]},
                {"id": "inf-7b", "instrument": "served-decode",
                 "cell": {"machine": "nobara", "backend": "cuda", "model": "7B", "depth": "d128"},
                 "provenance": "model-inference", "sd": 0.004, "sd_kind": "arm_log_level", "floor": 0.0, "n": 14,
                 "statistic": "components", "unit": "cell", "source": [{"commit": "abc"}]},
                {"id": "bits", "instrument": "inproc-kernel",
                 "cell": {"machine": "mac", "backend": "metal", "model": "*", "depth": "*"},
                 "provenance": "floor-only", "sd": None, "statistic": "bit-identical", "source": []},
            ],
        }
        with open(self.path, "w") as f:
            json.dump(reg, f)

    def tearDown(self):
        self.tmp.cleanup()

    def test_lookup_miss_refuses(self):
        rc, out, err = run(["--registry", self.path, "served-decode", "mac/cpu/7B/d128", "--bar", "0.97", "--paired"])
        self.assertEqual(rc, power.EXIT_USAGE)
        self.assertIn("registry miss", err)
        self.assertIn("pass --sd", err)
        self.assertEqual(out, "")

    def test_floor_only_entry_is_not_used_for_n(self):
        rc, _, err = run(["--registry", self.path, "inproc-kernel", "mac/metal/1.5B/d128", "--bar", "0.97", "--paired"])
        self.assertEqual(rc, power.EXIT_USAGE)
        self.assertIn("floor-only entries cover it: bits", err)

    def test_unknown_instrument(self):
        rc, _, err = run(["--registry", self.path, "served-nothing", "a/b/c/d", "--bar", "0.97"])
        self.assertEqual(rc, power.EXIT_USAGE)
        self.assertIn("unknown instrument", err)

    def test_measured_entry_beats_a_more_specific_inference(self):
        reg = power.load_registry(self.path)
        e, others = power.lookup(reg, "served-decode", power.parse_cell("nobara/cuda/7B/d128"))
        self.assertEqual(e["id"], "aa-pooled")
        self.assertEqual([o["id"] for o in others], ["inf-7b"])

    def test_arm_level_sd_doubles_in_variance_when_paired(self):
        reg = power.load_registry(self.path)
        e = [x for x in reg["entries"] if x["id"] == "inf-7b"][0]
        sd, _ = power.entry_sd(e, paired=True)
        self.assertAlmostEqual(sd, 0.004 * math.sqrt(2), places=12)

    def test_paired_entry_refused_for_unpaired_design(self):
        rc, _, err = run(["--registry", self.path, "served-decode", "nobara/cuda/0.5B/d128", "--bar", "0.97"])
        self.assertEqual(rc, power.EXIT_USAGE)
        self.assertIn("paired statistic", err)


class CommittedRegistry(unittest.TestCase):
    """The committed docs/measurements/noise-registry.json: well-formed, and every recomputable sd recomputes."""

    @classmethod
    def setUpClass(cls):
        cls.reg = power.load_registry()

    def test_every_entry_is_well_formed(self):
        ids = set()
        for e in self.reg["entries"]:
            self.assertNotIn(e["id"], ids, "duplicate id")
            ids.add(e["id"])
            self.assertIn(e["instrument"], self.reg["instruments"], e["id"])
            self.assertIn(e["provenance"], power.PROVENANCE_RANK, e["id"])
            self.assertTrue(e.get("source"), "%s has no source" % e["id"])
            self.assertTrue(e.get("statistic"), e["id"])
            for k in ("machine", "backend", "model", "depth"):
                self.assertIn(k, e["cell"], e["id"])
            if e.get("sd") is not None:
                self.assertIn(e.get("sd_kind"), ("paired_log_ratio", "arm_log_level"), e["id"])
                self.assertGreater(e["sd"], 0, e["id"])

    def test_every_sd_recomputes_from_its_values(self):
        n = 0
        for e in self.reg["entries"]:
            r = power.recompute_sd(e)
            if r is None:
                continue
            n += 1
            self.assertAlmostEqual(r, e["sd"], delta=5e-5, msg=e["id"])
        self.assertGreater(n, 5, "too few recomputable entries: the registry lost its values")

    def test_verify_command_passes(self):
        rc, out, _ = run(["verify"])
        self.assertEqual(rc, 0, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
