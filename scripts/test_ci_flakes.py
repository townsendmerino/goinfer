#!/usr/bin/env python3
"""ci_flakes.py's parsing and records over a small fixture: failing tests from a go test -json stream and from a job
log (timestamps, `--- FAIL:` lines inside the log's embedded JSON, the tail before the first error when no test is
named), and the per-failure record (re-run passed, solo or not). No network. Run: python3 -B scripts/test_ci_flakes.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ci_flakes  # noqa: E402

TEST_JSON = "\n".join([
    '{"Action":"run","Package":"github.com/x/goinfer/chat","Test":"TestA"}',
    '{"Action":"output","Package":"github.com/x/goinfer/chat","Test":"TestA","Output":"--- FAIL: TestA (0.01s)\\n"}',
    '{"Action":"fail","Package":"github.com/x/goinfer/chat","Test":"TestA/sub_1"}',
    '{"Action":"fail","Package":"github.com/x/goinfer/chat","Test":"TestA"}',
    '{"Action":"fail","Package":"github.com/x/goinfer/chat"}',
    '{"Action":"pass","Package":"github.com/x/goinfer/decoder","Test":"TestB"}',
    '{"Action":"fail","Package":"github.com/x/goinfer/internal/serveapp"}',
    'not json at all',
])

LOG_WITH_FAILS = "\n".join([
    "2026-11-19T03:50:47.6011637Z ##[group]Run go test ./...",
    '2026-11-19T04:04:58.8168616Z {"Action":"output","Test":"TestServe_x","Output":"--- FAIL: TestServe_x (3.20s)\\n"}',
    "2026-11-19T04:04:58.8168616Z     --- FAIL: TestServe_y/case_2 (0.00s)",
    "2026-11-19T04:04:58.5539461Z ##[error]Process completed with exit code 1.",
])

LOG_NO_TEST = "\n".join(["2026-11-19T03:50:47.6011637Z ##[group]Run staticcheck ./..."] +
                        [f"2026-11-19T03:50:48.0000000Z line {i}" for i in range(30)] +
                        ["2026-11-19T03:50:49.0000000Z decoder/x.go:12:2: unused (U1000)",
                         "2026-11-19T03:50:49.1000000Z ##[error]Process completed with exit code 1.",
                         "2026-11-19T03:50:49.2000000Z Post job cleanup."])


class Parsing(unittest.TestCase):
    def test_test_json(self):
        self.assertEqual(ci_flakes.failed_tests_from_test_json(TEST_JSON),
                         ["TestA", "TestA/sub_1", "github.com/x/goinfer/internal/serveapp (package)"])

    def test_log_fail_lines(self):
        self.assertEqual(ci_flakes.failed_tests_from_log(LOG_WITH_FAILS), (["TestServe_x", "TestServe_y/case_2"], ""))

    def test_log_tail_when_no_test_is_named(self):
        tests, tail = ci_flakes.failed_tests_from_log(LOG_NO_TEST)
        self.assertEqual(tests, [])
        self.assertIn("decoder/x.go:12:2: unused (U1000)", tail)
        self.assertTrue(tail.endswith("##[error]Process completed with exit code 1."))
        self.assertNotIn("2026-11-19T", tail)
        self.assertNotIn("Post job cleanup", tail)
        self.assertNotIn("line 0\n", tail)
        self.assertLessEqual(len(tail.splitlines()), ci_flakes.TAIL_LINES + 1)


def job(name, conclusion, jid=1):
    return {"id": jid, "name": name, "conclusion": conclusion,
            "steps": [{"name": "test", "conclusion": conclusion}]}


class Records(unittest.TestCase):
    def setUp(self):
        # run 1: root-windows red then green on re-run (attempt 2); run 2, same sha 'bbb': lint red alone;
        # run 3, sha 'ccc': root-darwin and gpu-darwin both red, not re-run.
        self.runs = [
            {"id": 1, "head_sha": "a" * 40, "event": "pull_request", "head_branch": "x", "run_attempt": 2},
            {"id": 2, "head_sha": "b" * 40, "event": "push", "head_branch": "main", "run_attempt": 1},
            {"id": 3, "head_sha": "c" * 40, "event": "push", "head_branch": "main", "run_attempt": 1},
        ]
        self.jobs = {
            (1, 1): [job("root-windows", "failure", 11), job("lint", "success", 12)],
            (1, 2): [job("root-windows", "success", 13), job("lint", "success", 12)],
            (2, 1): [job("lint", "failure", 21), job("root-windows", "success", 22)],
            (3, 1): [job("root-darwin", "failure", 31), job("gpu-darwin", "failure", 32), job("lint", "success", 33)],
        }
        self.recs = ci_flakes.build_records(self.runs, self.jobs, lambda run, j: ([f"Test{j['id']}"], ""))

    def test_one_record_per_failed_attempt(self):
        self.assertEqual([(r["run"], r["attempt"], r["lane"]) for r in self.recs],
                         [(1, 1, "root-windows"), (2, 1, "lint"), (3, 1, "gpu-darwin"), (3, 1, "root-darwin")])

    def test_rerun_passed(self):
        by = {(r["run"], r["lane"]): r for r in self.recs}
        self.assertIs(by[(1, "root-windows")]["rerun_passed"], True)
        self.assertIsNone(by[(2, "lint")]["rerun_passed"])

    def test_other_lanes_on_the_same_sha(self):
        by = {(r["run"], r["lane"]): r for r in self.recs}
        self.assertEqual(by[(1, "root-windows")]["other_lanes_failed_on_sha"], [])
        self.assertEqual(by[(3, "root-darwin")]["other_lanes_failed_on_sha"], ["gpu-darwin"])

    def test_record_fields(self):
        r = self.recs[0]
        self.assertEqual((r["sha"], r["event"], r["failed_step"], r["tests"]), ("a" * 40, "pull_request", "test", ["Test11"]))

    def test_markdown(self):
        md = ci_flakes.markdown(self.recs, "2026-11-18T00:00:00Z")
        self.assertIn("| root-windows | 1 (1) | aaaaaaaaa | pull_request | Test11 | yes | none (solo) |", md)
        self.assertIn("4 failed job attempts, 2 solo", md)
        self.assertIn("No failed ci job", ci_flakes.markdown([], "x"))


if __name__ == "__main__":
    unittest.main()
