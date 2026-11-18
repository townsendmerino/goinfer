#!/usr/bin/env python3
"""ci_test_summary.py prints the pass/skip/fail counts of top-level tests and a package's skip count beside its line, from a fake `go test -json` file.
Run: python3 -B scripts/test_ci_test_summary.py"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = "github.com/townsendmerino/goinfer/decoder"


def ev(action, test=None, **kw):
    d = {"Action": action, "Package": PKG, **kw}
    if test:
        d["Test"] = test
    return json.dumps(d)


def run(events):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write("\n".join(events) + "\n")
    p = subprocess.run([sys.executable, os.path.join(HERE, "ci_test_summary.py"), f.name], capture_output=True, text=True)
    return p.returncode, p.stdout


class Counts(unittest.TestCase):
    EVENTS = [ev("pass", "TestA", Elapsed=0.1), ev("pass", "TestB", Elapsed=0.1), ev("skip", "TestC", Elapsed=0.0), ev("skip", "TestD", Elapsed=0.0),
              ev("skip", "TestD/sub", Elapsed=0.0), ev("fail", "TestE", Elapsed=0.2), ev("fail", Elapsed=1.0)]

    def test_countsTopLevelTestsByResult(self):
        _, out = run(self.EVENTS)
        self.assertIn("== TESTS: 2 passed, 2 skipped, 1 failed (top-level) ==", out)

    def test_aPackageLineCarriesItsSkips(self):
        _, out = run(self.EVENTS)
        self.assertRegex(out, r"FAIL\s+1\.0s\s+decoder \[2 skipped\]")

    def test_anAllSkippedRunIsVisibleNotJustOk(self):
        _, out = run([ev("skip", "TestA", Elapsed=0.0), ev("skip", "TestB", Elapsed=0.0), ev("pass", Elapsed=0.1)])
        self.assertIn("== TESTS: 0 passed, 2 skipped, 0 failed (top-level) ==", out)


if __name__ == "__main__":
    unittest.main()
