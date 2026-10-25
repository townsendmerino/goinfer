"""BENCH_PHASES narrows bench_peer's phases A and B; unset, both run, as before."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_peer  # noqa: E402


class PlanPhases(unittest.TestCase):
    def setUp(self):
        self.saved = os.environ.pop("BENCH_PHASES", None)

    def tearDown(self):
        os.environ.pop("BENCH_PHASES", None)
        if self.saved is not None:
            os.environ["BENCH_PHASES"] = self.saved

    def test_unset_runs_both(self):
        self.assertEqual(bench_peer.plan_phases(), {"A", "B"})

    def test_b_only(self):
        os.environ["BENCH_PHASES"] = "b"
        self.assertEqual(bench_peer.plan_phases(), {"B"})

    def test_no_phase_refuses(self):
        os.environ["BENCH_PHASES"] = ","
        with self.assertRaises(SystemExit):
            bench_peer.plan_phases()

    def test_unknown_phase_refuses(self):
        os.environ["BENCH_PHASES"] = "B,Z"
        with self.assertRaises(SystemExit):
            bench_peer.plan_phases()


if __name__ == "__main__":
    unittest.main()
