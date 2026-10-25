#!/usr/bin/env python3
"""Tests for grade_g2.py: python3 docs/measurements/ks-f2-2026-10/test_grade_g2.py"""
import contextlib, io, json, os, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grade_g2  # noqa: E402


def cell(rows):
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", prefix="g2-test-", delete=False)
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.close()
    return f.name


def pairs(**over):
    rows = []
    for i in range(10):
        for arm, ms in (("A", 100.0), ("B", 90.0)):
            r = {"arm": arm, "tag": f"pair{i}", "decode_ms_per_token": ms, "load": 1.0, "vmswap_peak_mib": 0,
                 "avail_mib": 50000, "vmhwm_mib": 20000, "need_mib": 20000}
            r.update(over)
            rows.append(r)
    return rows


def run(rows):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = grade_g2.grade([cell(rows)])
    return code, out.getvalue()


class GradeG2(unittest.TestCase):
    def test_clean_cell_ships(self):
        code, out = run(pairs())
        self.assertEqual(code, 0, out)

    def test_need_zero_falls_back_to_the_measured_peak(self):
        # need_mib 0 is missing, not small: the cell's own peak (20000) stands in, and 50000 >= 20000 + 2048 passes.
        code, out = run(pairs(need_mib=0))
        self.assertEqual(code, 0, out)
        self.assertIn("not positive", out)

    def test_need_zero_still_voids_a_short_arm(self):
        # With need_mib 0 the old check weakened to avail < 2048; now 10000 MiB available against the 20000 peak voids every arm.
        code, out = run(pairs(need_mib=0, avail_mib=10000))
        self.assertEqual(code, 1, out)
        self.assertIn("VOID", out)
        self.assertIn("INCOMPLETE", out)

    def test_no_need_and_no_peak_voids(self):
        code, out = run(pairs(need_mib=0, vmhwm_mib=0))
        self.assertEqual(code, 1, out)
        self.assertIn("no need recorded", out)

    def test_swap_voids(self):
        code, out = run(pairs(vmswap_peak_mib=5))
        self.assertEqual(code, 1, out)
        self.assertIn("swap peak", out)


if __name__ == "__main__":
    unittest.main()
