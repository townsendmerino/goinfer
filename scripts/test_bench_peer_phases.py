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

    def test_unknown_phase_refuses(self):
        os.environ["BENCH_PHASES"] = "B,Z"
        with self.assertRaises(SystemExit):
            bench_peer.plan_phases()


class PromptSets(unittest.TestCase):
    def test_a_set_is_its_texts_and_a_single_entry_one_text(self):
        saved = dict(bench_peer._PROMPTS)
        try:
            bench_peer._PROMPTS["X:1"] = {"texts": ["a", "b"], "tokens": [1, 2]}
            bench_peer._PROMPTS["X:2"] = {"text": "c", "tokens": 1}
            self.assertEqual(bench_peer.prompt_texts(1, "X"), ["a", "b"])
            self.assertEqual(bench_peer.prompt_texts(2, "X"), ["c"])
        finally:
            bench_peer._PROMPTS.clear()
            bench_peer._PROMPTS.update(saved)


if __name__ == "__main__":
    unittest.main()
