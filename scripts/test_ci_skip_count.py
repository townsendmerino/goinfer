#!/usr/bin/env python3
"""scripts/ci_skip_count.sh counts top-level pass/skip/fail from a fake `go test -json` file and refuses an empty capture.
Run: python3 -B scripts/test_ci_skip_count.py"""
import json
import os
import subprocess
import tempfile
import unittest

SH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ci_skip_count.sh")


def ev(action, test=None, pkg="p"):
    d = {"Time": "2026-11-18T00:00:00Z", "Action": action, "Package": pkg}
    if test:
        d["Test"] = test
    d["Elapsed"] = 0.1
    return json.dumps(d, separators=(",", ":"))  # as go test -json writes it


def run(events):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write("\n".join(events) + "\n")
    p = subprocess.run(["bash", SH, f.name], capture_output=True, text=True)
    return p.returncode, p.stdout


class Count(unittest.TestCase):
    def test_countsTopLevelOnly(self):
        rc, out = run([ev("pass", "A"), ev("pass", "B"), ev("skip", "C"), ev("skip", "C/sub"), ev("fail", "D"), ev("fail")])
        self.assertEqual(rc, 0)
        self.assertIn("== TESTS: 2 passed, 1 skipped, 1 failed (top-level) ==", out)

    def test_anAllSkippedRunShowsItsSkips(self):
        _, out = run([ev("skip", "A"), ev("skip", "B"), ev("pass")])
        self.assertIn("0 passed, 2 skipped, 0 failed", out)

    def test_anEmptyCaptureIsRefused(self):
        rc, _ = run([ev("pass")])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
