#!/usr/bin/env python3
"""scripts/hostmem.py: which cmd/hostmem flags each reading passes, and the command line's dispatch."""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hostmem  # noqa: E402


class Flags(unittest.TestCase):
    def run_with(self, fn):
        seen = {}

        def fake(cmd, **kw):
            seen["cmd"] = cmd
            return mock.Mock(returncode=0, stdout="42\n")
        with mock.patch.object(hostmem.subprocess, "run", fake):
            return fn(), seen["cmd"]

    def test_available_passes_mb_only(self):
        mb, cmd = self.run_with(hostmem.available_mb)
        self.assertEqual((mb, cmd[-1:]), (42, ["-mb"]))

    def test_reclaimable_passes_reclaimable(self):
        mb, cmd = self.run_with(hostmem.reclaimable_mb)
        self.assertEqual(mb, 42)
        self.assertIn("-reclaimable", cmd)
        self.assertIn("-mb", cmd)


class CommandLine(unittest.TestCase):
    def dispatch(self, argv):
        with mock.patch.object(hostmem, "available_mb", lambda: 1), mock.patch.object(hostmem, "reclaimable_mb", lambda: 2):
            out = io.StringIO()
            with redirect_stdout(out):
                hostmem.main(argv)
            return out.getvalue().strip()

    def test_both_spellings_print_reclaimable(self):
        self.assertEqual(self.dispatch(["--reclaimable"]), "2")
        self.assertEqual(self.dispatch(["-reclaimable"]), "2")

    def test_no_flag_prints_available(self):
        self.assertEqual(self.dispatch([]), "1")

    def test_unknown_flag_is_an_error(self):
        with self.assertRaises(SystemExit) as e:
            self.dispatch(["-reclaim"])
        self.assertIn("unknown", str(e.exception.code))


if __name__ == "__main__":
    unittest.main()
