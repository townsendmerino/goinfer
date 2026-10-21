#!/usr/bin/env python3
"""The cuda-device-suite issue logic (scripts/cuda_device_suite_issue.py) against a fake gh: red opens an issue titled with the short sha or comments on the open one,
green closes the open one, an unreadable verdict touches nothing; the plant that reads every verdict as green opens nothing for a red run.
Run: python3 -B scripts/test_cuda_device_suite_issue.py"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cuda_device_suite_issue as cdi  # noqa: E402

RED = ["cuda-device-suite: 90 passed, 1 failed, 0 device skips, 3 other skips, 1 failed package(s)", "FAILED TESTS (1):",
       "  github.com/townsendmerino/goinfer/cuda.TestBroken", "VERDICT: RED"]
GREEN = ["cuda-device-suite: 91 passed, 0 failed, 0 device skips, 3 other skips, 0 failed package(s)", "VERDICT: GREEN"]

FAKE = """#!/usr/bin/env python3
import json, os, sys
a = sys.argv[1:]
with open(os.environ["FAKE_GH_LOG"], "a") as f:
    f.write(json.dumps(a) + "\\n")
if a[:2] == ["issue", "list"]:
    print(os.environ.get("FAKE_GH_OPEN", "[]"))
if a[:2] == ["issue", "create"] or a[:2] == ["issue", "comment"]:
    body = a[a.index("--body-file") + 1]
    with open(os.environ["FAKE_GH_LOG"] + ".body", "a") as f:
        f.write(open(body).read())
"""


class Fake:
    def __init__(self, test, open_issues=()):
        self.d = tempfile.mkdtemp()
        self.gh = os.path.join(self.d, "gh")
        self.log = os.path.join(self.d, "calls")
        open(self.gh, "w").write(FAKE)
        os.chmod(self.gh, 0o755)
        os.environ["FAKE_GH_LOG"] = self.log
        os.environ["FAKE_GH_OPEN"] = json.dumps([{"number": n, "title": "cuda-device-suite red on abc0001"} for n in open_issues])

    def calls(self):
        return [json.loads(l) for l in open(self.log)] if os.path.exists(self.log) else []

    def mutating(self):
        return [c for c in self.calls() if c[:2] != ["issue", "list"]]

    def body(self):
        return open(self.log + ".body").read()

    def run(self, lines, sha="def5678"):
        vl = os.path.join(self.d, "verdict.log")
        open(vl, "w").write("\n".join(lines) + "\n")
        return subprocess.run([sys.executable, os.path.join(HERE, "cuda_device_suite_issue.py"), "--verdict-log", vl, "--sha", sha, "--gh", self.gh],
                              capture_output=True, text=True)


class Issue(unittest.TestCase):
    def test_redWithNoOpenIssueOpensOneTitledWithTheSha(self):
        f = Fake(self)
        p = f.run(["go: some output /home/francis/x"] + RED)
        self.assertEqual(p.returncode, 0, p.stdout)
        create = [c for c in f.mutating() if c[:2] == ["issue", "create"]]
        self.assertEqual(len(create), 1, f.calls())
        self.assertEqual(create[0][create[0].index("--title") + 1], "cuda-device-suite red on def5678")
        self.assertIn("cuda.TestBroken", f.body())
        self.assertNotIn("/home/francis", f.body())
        self.assertIn("VERDICT: RED", f.body())

    def test_redWithAnOpenIssueCommentsOnTheNewestAndOpensNothing(self):
        f = Fake(self, open_issues=(41, 44))
        self.assertEqual(f.run(RED).returncode, 0)
        m = f.mutating()
        self.assertEqual([c[:3] for c in m], [["issue", "comment", "44"]], f.calls())
        self.assertIn("cuda.TestBroken", f.body())

    def test_greenClosesTheOpenIssueWithAComment(self):
        f = Fake(self, open_issues=(44,))
        self.assertEqual(f.run(GREEN).returncode, 0)
        m = f.mutating()
        self.assertEqual(len(m), 1, f.calls())
        self.assertEqual(m[0][:3], ["issue", "close", "44"])
        self.assertIn("--comment", m[0])
        self.assertIn("def5678", m[0][m[0].index("--comment") + 1])

    def test_greenWithNothingOpenDoesNothing(self):
        f = Fake(self)
        self.assertEqual(f.run(GREEN).returncode, 0)
        self.assertEqual(f.mutating(), [])

    def test_aLogWithNoVerdictTouchesNothingAndExitsTwo(self):
        f = Fake(self, open_issues=(44,))
        p = f.run(["FATAL: cannot check out origin/main"])
        self.assertEqual(p.returncode, 2, p.stdout)
        self.assertEqual(f.calls(), [])

    def test_aFailingGhIsReported(self):
        f = Fake(self)
        open(f.gh, "w").write("#!/bin/sh\necho boom >&2\nexit 1\n")
        self.assertEqual(f.run(RED).returncode, 1)

    def test_theVerdictIsTheLastLineNotAnEarlierOne(self):
        f = Fake(self, open_issues=(44,))
        self.assertEqual(f.run(RED + ["a rerun follows"] + GREEN).returncode, 0)
        self.assertEqual(f.mutating()[0][:3], ["issue", "close", "44"])


class Plant(unittest.TestCase):
    def test_readingEveryVerdictAsGreenOpensNothingForARedRun(self):
        saved = cdi.parse_verdict
        cdi.parse_verdict = lambda lines: "GREEN"
        try:
            f = Fake(self)
            os.environ["FAKE_GH_LOG"] = f.log
            vl = os.path.join(f.d, "v.log")
            open(vl, "w").write("\n".join(RED) + "\n")
            cdi.main(["--verdict-log", vl, "--sha", "def5678", "--gh", f.gh])
            self.assertEqual([c for c in f.mutating() if c[:2] == ["issue", "create"]], [], "the plant must open no issue, or the red test cannot see the defect it plants")
        finally:
            cdi.parse_verdict = saved


if __name__ == "__main__":
    unittest.main()
