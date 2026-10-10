#!/usr/bin/env python3
"""Tests for comment_census.py (CC6): python3 scripts/test_comment_census.py"""
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import comment_census as cc

OLD = '''package p

// Old keeps M-35 as a label and says 12 ms once.
func Old() {}

const kernel = `
// 2026-09-12 inside a raw string is kernel text, not a comment
`
'''


def git(d, *a):
    return subprocess.run(["git", *a], cwd=d, capture_output=True, text=True, check=True).stdout.strip()


class Markers(unittest.TestCase):
    def kinds(self, s):
        return sorted({k for k, _ in cc.markers(s)})

    def test_each_kind(self):
        self.assertEqual(self.kinds("// found 2026-09-12 by bisecting"), ["date"])
        self.assertEqual(self.kinds("// see 3358e6ba for the fix"), ["commit id"])
        self.assertEqual(self.kinds("// M-35 option (b) was never done"), ["tracker id"])
        self.assertEqual(self.kinds("// measured 2.28x on an 8k prefill"), ["figure with a unit"])
        self.assertEqual(self.kinds("// the guard moved from forwardn.go:44"), ["file.go:NNN"])

    def test_not_history(self):
        for s in ("// returns an error, never nil", "// uses UTF-8 and SHA-256", "// the 1234567 case", "// a defaced file", "// timeout in seconds",
                  "// sees docs/measurements/x-2026-09-23.md for the numbers", "// (M-08 on docs/code-notes/gpu.md#runModelToModelW)"):
            self.assertEqual(self.kinds(s), [], s)


class Diff(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        git(self.d, "init", "-q")
        git(self.d, "config", "user.email", "t@example.com")
        git(self.d, "config", "user.name", "T")
        open(os.path.join(self.d, "p.go"), "w").write(OLD)
        git(self.d, "add", "p.go")
        git(self.d, "commit", "-q", "-m", "base")
        self.base = git(self.d, "rev-parse", "HEAD")
        self._cwd = os.getcwd()
        os.chdir(self.d)

    def tearDown(self):
        os.chdir(self._cwd)
        self.tmp.cleanup()

    def run_diff(self, new, *extra):
        open("p.go", "w").write(new)
        git(self.d, "commit", "-qam", "change")
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cc.main(["diff", "--base", self.base, "--head", "HEAD", *extra])
        return code, buf.getvalue()

    def test_added_history_comment_warns_and_exits_zero(self):
        code, out = self.run_diff(OLD + "\n// New: found 2026-10-01, 40 ms slower (N-79).\nfunc New() {}\n")
        self.assertEqual(code, 0, out)
        self.assertIn("WARNING", out)
        self.assertIn("p.go:", out)
        for kind in ("date", "figure with a unit", "tracker id"):
            self.assertIn(kind, out)

    def test_strict_exits_one(self):
        code, _ = self.run_diff(OLD + "\n// New: found 2026-10-01.\nfunc New() {}\n", "--strict")
        self.assertEqual(code, 1)

    def test_clean_change_is_quiet(self):
        code, out = self.run_diff(OLD + "\n// New reports the count; the caller must not retain the slice.\nfunc New() {}\n")
        self.assertEqual((code, "no history markers" in out), (0, True), out)

    def test_a_kept_label_and_a_rewrap_are_not_new_history(self):
        new = OLD.replace("// Old keeps M-35 as a label and says 12 ms once.", "// Old keeps M-35 as a label and\n// says 12 ms once.")
        code, out = self.run_diff(new)
        self.assertIn("no history markers", out, out)

    def test_raw_string_and_code_lines_are_never_reported(self):
        new = OLD.replace("// 2026-09-12 inside a raw string is kernel text", "// 2026-10-09 inside a raw string is kernel text") + "\nvar x = 40 // ms is not a marker kind with no unit digits\n"
        code, out = self.run_diff(new)
        self.assertNotIn("raw string", out)

    def test_directive_is_never_reported(self):
        code, out = self.run_diff(OLD + "\n//go:build ignore2026-09-12\nfunc D() {}\n")
        self.assertNotIn("go:build", out)

    def test_trailing_comment_is_reported(self):
        code, out = self.run_diff(OLD + "\nvar y = 1 // since 2026-10-01\n")
        self.assertIn("date", out)


class Report(unittest.TestCase):
    def test_report_counts_comment_blocks_and_ignores_raw_strings(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "p.go")
            open(p, "w").write(OLD)
            row = cc.census_file(p)
            self.assertEqual(row["comment"], 1)
            self.assertEqual(row["short"], 1)
            self.assertEqual(row["short_unmarked"], 0)  # the one block carries M-35
            self.assertEqual(row["doc"], 1)  # Old is an exported declaration with a doc comment


if __name__ == "__main__":
    unittest.main()
