#!/usr/bin/env python3
"""Tests for remap_gate_citations.py — pins V-26 (docs/review-2026-09-04.md).

No test runner is wired into CI for this yet; run directly:
    python3 scripts/test_remap_gate_citations.py
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "remap_gate_citations", os.path.join(HERE, "remap_gate_citations.py"))
remap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(remap)


class TestBuildMapper(unittest.TestCase):
    def test_pure_insertion_does_not_shift_the_line_immediately_before_it(self):
        # `@@ -10,0 +11,3 @@` -- 3 new lines inserted after old line 10; old line 10 itself
        # is untouched. A citation sitting AT old line 10 (the line immediately before the
        # insertion) must stay at 10, not shift by the insertion's size.
        mapline = remap.build_mapper([(10, 0, 11, 3)])
        self.assertEqual(mapline(10), 10, "the line immediately before a pure insertion moved")
        # Anything strictly after the insertion point shifts by the full delta (+3).
        self.assertEqual(mapline(11), 14)
        self.assertEqual(mapline(20), 23)
        # Anything strictly before it is untouched.
        self.assertEqual(mapline(5), 5)

    def test_replacement_hunk_still_shifts_correctly(self):
        # `@@ -10,2 +10,5 @@` -- 2 old lines replaced by 5 new ones: net +3.
        mapline = remap.build_mapper([(10, 2, 10, 5)])
        self.assertEqual(mapline(9), 9)          # before the hunk: untouched
        self.assertEqual(mapline(12), 15)        # after the hunk: shifted by the full delta
        # A citation landing INSIDE the replaced range has no fixed answer (its content
        # changed); the function returns o+delta-so-far without asserting more than that
        # it terminates and does not crash.
        self.assertEqual(mapline(10), 10)

    def test_multiple_hunks_accumulate(self):
        # Two insertions: +3 after line 10, +2 after line 30 (in OLD-file coordinates).
        mapline = remap.build_mapper([(10, 0, 11, 3), (30, 0, 34, 2)])
        self.assertEqual(mapline(5), 5)
        self.assertEqual(mapline(10), 10)   # immediately before the FIRST insertion
        self.assertEqual(mapline(20), 23)   # past the first, before the second
        self.assertEqual(mapline(30), 33)   # immediately before the SECOND insertion (+3 from the first only)
        self.assertEqual(mapline(35), 40)   # past both


class TestRemapPathMultiplePaths(unittest.TestCase):
    """V-26: remap_path used to be inlined in a loop that reassigned the SAME name (`base`)
    used for the git ref, so from the second path onward `git diff` was handed a file
    basename instead of a revision and remapped nothing. Drives the real function (not a
    reimplementation of its bug) against a real git repo with two files edited in one commit,
    over two paths, and asserts BOTH get remapped -- not just the first.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = self.tmp.name
        self._git("init", "-q")
        self._git("config", "user.email", "test@test")
        self._git("config", "user.name", "test")
        os.makedirs(os.path.join(self.repo, "docs"))
        os.makedirs(os.path.join(self.repo, "pkg"))

    def tearDown(self):
        self.tmp.cleanup()

    def _git(self, *args):
        subprocess.run(["git"] + list(args), cwd=self.repo, check=True,
                        capture_output=True, text=True)

    def _write(self, rel, lines):
        with open(os.path.join(self.repo, rel), "w") as f:
            f.write("\n".join(lines) + "\n")

    def test_two_paths_both_remap(self):
        # a.go and b.go each get one line inserted at the top, pushing everything down by 1.
        self._write("pkg/a.go", [f"line{i}" for i in range(1, 6)])
        self._write("pkg/b.go", [f"line{i}" for i in range(1, 6)])
        self._write("docs/x.md", ["cites `pkg/a.go:3` and `pkg/b.go:3`."])
        self._git("add", "-A")
        self._git("commit", "-q", "-m", "base")

        self._write("pkg/a.go", ["NEW"] + [f"line{i}" for i in range(1, 6)])
        self._write("pkg/b.go", ["NEW"] + [f"line{i}" for i in range(1, 6)])
        self._git("add", "-A")
        self._git("commit", "-q", "-m", "edit")

        cwd = os.getcwd()
        os.chdir(self.repo)
        try:
            total = remap.remap_path("HEAD~1", False, "pkg/a.go") + \
                remap.remap_path("HEAD~1", False, "pkg/b.go")
        finally:
            os.chdir(cwd)

        got = open(os.path.join(self.repo, "docs/x.md")).read()
        self.assertIn("pkg/a.go:4", got, f"a.go citation was not remapped: {got!r}")
        self.assertIn("pkg/b.go:4", got,
                      f"b.go citation was not remapped (V-26: second path silently no-ops): {got!r}")
        self.assertEqual(total, 2)


if __name__ == "__main__":
    unittest.main()
