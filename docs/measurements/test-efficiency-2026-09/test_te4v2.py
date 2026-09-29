#!/usr/bin/env python3
"""Hand-built cases for te4v2_replay.py's rule (TE4-SEQ-v2, te4-seq-v2-2026-09-28.md §1). Run: python3 -B test_te4v2.py"""
import math
import unittest

import te4v2_replay as v2


class Rule(unittest.TestCase):
    def test_large_clear_win_stops_at_the_second_quad(self):
        ds = [math.log(1.20)] * 8  # 20% ahead, no spread
        self.assertEqual(v2.sequential(ds, 4, 0.0081, 6, 0.0, 1.03, 0.97), ("PASS", 2))

    def test_clear_loss_stops_fail(self):
        ds = [math.log(0.80) + 0.001 * (-1) ** i for i in range(8)]
        self.assertEqual(v2.sequential(ds, 4, 0.0081, 6, 0.0, 1.03, 0.97), ("FAIL", 2))

    def test_never_stops_on_the_first_quad(self):
        # A win the first quad alone would clear: the rule still needs q = 2, so it cannot report q = 1.
        ds = [math.log(1.50)] * 8
        self.assertGreaterEqual(v2.sequential(ds, 4, 0.0081, 6, 0.0, 1.03, 0.97)[1], 2)

    def test_the_prior_blocks_a_tight_early_read_from_a_noisy_instrument(self):
        # Four identical block readings 4% ahead: the data sd is 0, but the Mac CPU prior (0.0333) keeps the early
        # interval wide, so it cannot stop at q = 2 against a 1.03 bar.
        ds = [math.log(1.04)] * 8
        v, q = v2.sequential(ds, 4, 0.0333, 6, 0.0, 1.03, 0.97)
        self.assertGreater(q, 2)

    def test_a_shift_between_quads_widens_the_interval(self):
        # Quad 1 reads +8%, quad 2 reads -2%: the spread keeps the rule from stopping PASS at q = 2.
        ds = [math.log(1.08)] * 2 + [math.log(0.98)] * 2 + [0.0] * 4
        v, q = v2.sequential(ds, 4, 0.0081, 6, 0.0, 1.03, 0.97)
        self.assertFalse(v == "PASS" and q == 2)

    def test_at_the_cap_the_fixed_n_verdict_applies(self):
        ds = [0.004, -0.003, 0.002, -0.001, 0.003, -0.002, 0.001, 0.0]  # an A/A
        v, q = v2.sequential(ds, 4, 0.0081, 6, 0.0, 1.03, 0.97)
        self.assertEqual(v2.fixed_n(ds, 0.0, 1.03, 0.97), "AMBIGUOUS")
        self.assertEqual((v, q), ("AMBIGUOUS", 4))

    def test_floor_is_charged(self):
        ds = [math.log(1.035)] * 8
        self.assertEqual(v2.fixed_n(ds, 0.0, 1.03, 0.97), "PASS")
        self.assertEqual(v2.fixed_n(ds, 0.0144, 1.03, 0.97), "AMBIGUOUS")

    def test_prior_df(self):
        self.assertEqual(v2.prior_df(20), 6)
        self.assertEqual(v2.prior_df(3), 2)
        self.assertEqual(v2.prior_df(1), 1)

    def test_quad_orders(self):
        self.assertEqual(len(v2.quad_orders(4)), 24)
        o6 = v2.quad_orders(6)
        self.assertEqual(len(o6), v2.PERM_MAX)
        self.assertEqual(len({tuple(o) for o in o6}), v2.PERM_MAX)


class Replay(unittest.TestCase):
    def test_abba_blocks_pairs_by_block(self):
        recs = [{"engine": e, "backend": "cuda", "model": "0.5B", "depth": 128, "config": "greedy", "block": b,
                 "runs": [100.0 + b], "secs": 10.0, "counts": {"completion_rates": [100.0 + b + (e == "ollama")]}}
                for b in range(2) for e in ("goinfer", "ollama")]
        got = v2.abba_blocks(recs, "cuda", "0.5B", "goinfer", "ollama")
        self.assertEqual(sorted(got), [0, 1])
        self.assertAlmostEqual(got[1][1], 102.0)


if __name__ == "__main__":
    unittest.main()
