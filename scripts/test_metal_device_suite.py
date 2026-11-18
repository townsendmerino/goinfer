#!/usr/bin/env python3
"""The Metal day job's files: the LaunchAgent plist parses and runs the script that exists, and the script refuses off macOS before it touches anything.
Run: python3 -B scripts/test_metal_device_suite.py"""
import os
import plistlib
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "docs/measurements/metal-device-suite-2026-11-17")
SCRIPT = os.path.join(D, "run-metal-device-suite.sh")


class Files(unittest.TestCase):
    def test_plistRunsTheScriptAtADailyHourAndDoesNotRunAtLoad(self):
        p = plistlib.load(open(os.path.join(D, "com.goinfer.metal-device-suite.plist"), "rb"))
        self.assertEqual(p["Label"], "com.goinfer.metal-device-suite")
        cmd = p["ProgramArguments"][-1]
        self.assertIn("docs/measurements/metal-device-suite-2026-11-17/run-metal-device-suite.sh", cmd)
        self.assertTrue(os.path.exists(os.path.join(ROOT, "docs/measurements/metal-device-suite-2026-11-17/run-metal-device-suite.sh")))
        self.assertEqual(set(p["StartCalendarInterval"]), {"Hour", "Minute"})
        self.assertFalse(p["RunAtLoad"])

    def test_theScriptParses(self):
        self.assertEqual(subprocess.run(["bash", "-n", SCRIPT]).returncode, 0)

    def test_theScriptHoldsTheTimingLockAndOnlyMainFilesIssues(self):
        s = open(SCRIPT).read()
        self.assertIn("timing_lock.py", s)
        self.assertIn('[ "$REV" = "origin/main" ]', s)

    @unittest.skipIf(sys.platform == "darwin", "the refusal is for non-macOS boxes")
    def test_theScriptRefusesOffMacBeforeTouchingTheRepo(self):
        p = subprocess.run(["bash", SCRIPT], capture_output=True, text=True, env=dict(os.environ, HOME="/nonexistent"))
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertIn("not macOS", p.stdout)


if __name__ == "__main__":
    unittest.main()
