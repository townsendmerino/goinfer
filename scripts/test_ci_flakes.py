#!/usr/bin/env python3
"""ci_flakes.py's parsing and records over a small fixture: failing tests from a go test -json stream and from a job
log (timestamps, `--- FAIL:` lines inside the log's embedded JSON, the tail before the first error when no test is
named), the per-failure record (re-run passed, solo or not), and how a job's log and a lane's kept test.json are fetched
(gh is faked). No network. Run: python3 -B scripts/test_ci_flakes.py"""
import os
import subprocess
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

    def test_log_colour_codes_are_dropped(self):
        log = "\n".join(["2026-11-19T03:50:47.6011637Z \x1b[36;1mgo vet ./...\x1b[0m",
                         "2026-11-19T03:50:48.0000000Z \x1b[31mdecoder/x.go:12:2: unkeyed fields\x1b[0m",
                         "2026-11-19T03:50:49.1000000Z ##[error]Process completed with exit code 1."])
        tests, tail = ci_flakes.failed_tests_from_log(log)
        self.assertEqual(tests, [])
        self.assertEqual(tail, "go vet ./...\ndecoder/x.go:12:2: unkeyed fields\n##[error]Process completed with exit code 1.")

    def test_log_tail_when_no_test_is_named(self):
        tests, tail = ci_flakes.failed_tests_from_log(LOG_NO_TEST)
        self.assertEqual(tests, [])
        self.assertIn("decoder/x.go:12:2: unused (U1000)", tail)
        self.assertTrue(tail.endswith("##[error]Process completed with exit code 1."))
        self.assertNotIn("2026-11-19T", tail)
        self.assertNotIn("Post job cleanup", tail)
        self.assertNotIn("line 0\n", tail)
        self.assertLessEqual(len(tail.splitlines()), ci_flakes.TAIL_LINES + 1)


class FakeGh:
    """Stands in for subprocess.run: records each gh command and answers it from `answer(args) -> (rc, stdout, stderr)`."""

    def __init__(self, answer):
        self.calls, self.answer = [], answer

    def __call__(self, args, **kw):
        self.calls.append(args)
        rc, out, err = self.answer(args)
        return subprocess.CompletedProcess(args, rc, out, err)


REFUSAL = "the response contains terminal escape sequences; pass --allow-escape-sequences to output it anyway"


class Fetching(unittest.TestCase):
    def test_a_job_log_is_asked_for_with_its_escape_sequences(self):
        gh = FakeGh(lambda a: (0, "the log", "") if "--allow-escape-sequences" in a else (1, "", REFUSAL))
        self.assertEqual(ci_flakes.job_log("o/r", 7, gh), ("the log", ""))
        self.assertEqual(gh.calls, [["gh", "api", "--allow-escape-sequences", "repos/o/r/actions/jobs/7/logs"]])

    def test_a_gh_without_the_flag_is_asked_again_without_it(self):
        gh = FakeGh(lambda a: (1, "", "unknown flag: --allow-escape-sequences\n\nUsage: ...") if "--allow-escape-sequences" in a else (0, "the log", ""))
        self.assertEqual(ci_flakes.job_log("o/r", 7, gh), ("the log", ""))
        self.assertEqual(gh.calls[1], ["gh", "api", "repos/o/r/actions/jobs/7/logs"])

    def test_any_other_failure_is_reported_once(self):
        gh = FakeGh(lambda a: (1, "", "HTTP 410: log expired\n"))
        self.assertEqual(ci_flakes.job_log("o/r", 7, gh), (None, "HTTP 410: log expired"))
        self.assertEqual(len(gh.calls), 1)

    def test_the_log_names_the_tests_when_the_lane_keeps_no_test_json(self):
        gh = FakeGh(lambda a: (0, LOG_WITH_FAILS, ""))
        self.assertEqual(ci_flakes._tests_for("o/r", gh)({"id": 1}, {"id": 7, "name": "root-darwin"}), (["TestServe_x", "TestServe_y/case_2"], ""))
        self.assertEqual([c[:3] for c in gh.calls], [["gh", "api", "--allow-escape-sequences"]], "no artifact is asked for")

    def test_each_lane_that_keeps_its_test_json_is_asked_for_its_own(self):
        for lane, name in (("root-windows", "root-windows-test-json"), ("root-windows-arm64", "root-windows-arm64-test-json")):
            gh = FakeGh(lambda a: (1, "", "no artifact") if a[1:3] == ["run", "download"] else (0, LOG_WITH_FAILS, ""))
            tests, _ = ci_flakes._tests_for("o/r", gh)({"id": 1}, {"id": 7, "name": lane})
            self.assertEqual(gh.calls[0][:7], ["gh", "run", "download", "1", "-R", "o/r", "-n"])
            self.assertEqual(gh.calls[0][7], name)
            self.assertEqual(tests, ["TestServe_x", "TestServe_y/case_2"], "without the artifact, the log")
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".github", "workflows", "ci.yml")) as f:
            ci = f.read()
        for name in ci_flakes.ARTIFACTS.values():
            self.assertIn(f"name: {name}\n", ci, "the lane uploads it under this name")

    def test_an_unfetchable_log_says_why(self):
        gh = FakeGh(lambda a: (1, "", "HTTP 404"))
        self.assertEqual(ci_flakes._tests_for("o/r", gh)({"id": 1}, {"id": 7, "name": "lint"}), ([], "(log unavailable: HTTP 404)"))


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
