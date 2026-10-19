#!/usr/bin/env python3
"""queue-cuda-device-suite.sh (the 18:00 timer's command) adds the job once and skips when it is already queued; against a fake checkout whose night.py records its argv.
Run: python3 -B scripts/test_cuda_device_suite_queue.py"""
import os
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(os.path.dirname(HERE), "docs/measurements/cuda-device-suite-2026-10-18/queue-cuda-device-suite.sh")
SCRIPT = "docs/measurements/cuda-device-suite-2026-10-18/run-cuda-device-suite.sh"
NIGHT = """import os, sys
a = sys.argv[1:]
open(os.path.join(os.path.dirname(__file__), "calls"), "a").write(" ".join(a) + "\\n")
if a[0] == "list":
    print(open(os.path.join(os.path.dirname(__file__), "listing")).read(), end="")
"""


def fake_repo(listing, with_script=True):
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "scripts"))
    open(os.path.join(d, "scripts/night.py"), "w").write(NIGHT)
    open(os.path.join(d, "scripts/listing"), "w").write(listing)
    if with_script:
        os.makedirs(os.path.join(d, os.path.dirname(SCRIPT)))
        open(os.path.join(d, SCRIPT), "w").write("#!/bin/bash\n")
    return d


def run(d):
    p = subprocess.run(["bash", HELPER], capture_output=True, text=True, env=dict(os.environ, GOINFER_NIGHT_REPO=d))
    calls = open(os.path.join(d, "scripts/calls")).read().splitlines() if os.path.exists(os.path.join(d, "scripts/calls")) else []
    return p, calls


class Queue(unittest.TestCase):
    def test_addsTheJobWhenNothingIsQueued(self):
        p, calls = run(fake_repo("the night queue is empty\n"))
        self.assertEqual(p.returncode, 0, p.stderr)
        adds = [c for c in calls if c.startswith("add ")]
        self.assertEqual(len(adds), 1, calls)
        self.assertTrue(adds[0].startswith("add cuda-device-suite --est 10 "), adds[0])
        self.assertIn(SCRIPT, adds[0])

    def test_skipsWhenAlreadyQueued(self):
        p, calls = run(fake_repo("1  cuda-device-suite  est 10 min\n"))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("already queued", p.stdout)
        self.assertEqual([c for c in calls if c.startswith("add ")], [])

    def test_aQueueOfOtherJobsStillGetsIt(self):
        p, calls = run(fake_repo("1  gate-gpu-cuda  est 150\n2  b13-dark-gates  est 8\n"))
        self.assertEqual(len([c for c in calls if c.startswith("add ")]), 1, calls)

    def test_aBranchJobWithAPrefixOfTheNameDoesNotSuppressIt(self):
        p, calls = run(fake_repo("1  cuda-device-suite-kv3-final  est 10 min\n2  cuda-device-suite-swap-giw  est 10 min\n"))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(len([c for c in calls if c.startswith("add ")]), 1, calls)

    def test_aCheckoutWithoutTheScriptFailsLoudly(self):
        p, calls = run(fake_repo("the night queue is empty\n", with_script=False))
        self.assertNotEqual(p.returncode, 0)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
