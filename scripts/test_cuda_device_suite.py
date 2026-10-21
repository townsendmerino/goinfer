#!/usr/bin/env python3
"""The cuda-device-suite verdict (scripts/cuda_device_suite.py): a failing test is red, a device skip is red, and the plant that treats skips as passes goes green
wrongly. Fed fake `go test -json` streams through the runner's real command path. Run: python3 -B scripts/test_cuda_device_suite.py"""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cuda_device_suite as cds  # noqa: E402

PKG = "github.com/townsendmerino/goinfer/cuda"


def ev(action, test=None, output=None):
    d = {"Action": action, "Package": PKG}
    if test:
        d["Test"] = test
    if output is not None:
        d["Output"] = output
    return json.dumps(d)


def passing(name):
    return [ev("run", name), ev("pass", name)]


def skipping(name, why):
    return [ev("run", name), ev("output", name, f"    x_test.go:12: {why}\n"), ev("skip", name)]


def failing(name):
    return [ev("run", name), ev("output", name, "    x_test.go:9: boom\n"), ev("fail", name)]


def run(events, rc=0):
    """The runner on a fake `go test -json` that prints the events and exits rc. Returns (exit, output)."""
    with tempfile.TemporaryDirectory() as d:
        fake = os.path.join(d, "fake.py")
        with open(fake, "w") as f:
            f.write("import sys\nfor l in %r:\n    print(l)\nsys.exit(%d)\n" % (events, rc))
        p = subprocess.run([sys.executable, os.path.join(HERE, "cuda_device_suite.py"), "--", sys.executable, fake],
                           capture_output=True, text=True)
    return p.returncode, p.stdout


class Verdict(unittest.TestCase):
    def test_allPassIsGreen(self):
        rc, out = run(passing("TestA") + passing("TestB"))
        self.assertEqual(rc, 0, out)
        self.assertIn("VERDICT: GREEN", out)

    def test_oneFailingTestIsRedAndNamed(self):
        rc, out = run(passing("TestA") + failing("TestBroken") + [ev("fail")], rc=1)
        self.assertEqual(rc, 1, out)
        self.assertIn("VERDICT: RED", out)
        self.assertIn(f"{PKG}.TestBroken", out.split("FAILED TESTS")[1])
        self.assertEqual(out.strip().splitlines()[-1], "VERDICT: RED")

    def test_theFailingNamesAreInTheLast25Lines(self):
        # night.py's SUMMARY.md quotes the last 25 lines of a job that is not ok.
        rc, out = run(sum([passing(f"TestP{i}") for i in range(200)], []) + failing("TestBroken") + [ev("fail")], rc=1)
        self.assertIn("TestBroken", "\n".join(out.strip().splitlines()[-25:]))

    def test_deviceSkipIsRedEvenWhenGoTestExitsZero(self):
        for why in ("no device: cuda init failed", "no CUDA device: x", "no cuda device: x", "the model did not go resident on this device"):
            rc, out = run(passing("TestA") + skipping("TestNeedsGPU", why), rc=0)
            self.assertEqual(rc, 1, (why, out))
            self.assertIn(f"{PKG}.TestNeedsGPU", out.split("DEVICE SKIPS")[1])

    def test_aSkipForAnAssetIsNotAFailure(self):
        rc, out = run(passing("TestA") + skipping("TestHeavy", "set GOINFER_HEAVY_TESTS=1 (loads a small model for a device context)"))
        self.assertEqual(rc, 0, out)
        self.assertIn("1 other skips", out)

    def test_nothingPassedIsRed(self):
        rc, out = run(skipping("TestOnly", "set GOINFER_HEAVY_TESTS=1"))
        self.assertEqual(rc, 1, out)
        self.assertIn("nothing ran", out)

    def test_aBuildFailureIsRedWithItsLines(self):
        rc, out = run([ev("output", None, "build failed\n"), ev("fail")] + passing("TestA"), rc=1)
        self.assertEqual(rc, 1, out)
        self.assertIn("FAILED PACKAGES", out)

    def test_aNonZeroExitWithNoFailedTestIsRed(self):
        rc, out = run(passing("TestA"), rc=2)
        self.assertEqual(rc, 1, out)
        self.assertIn("exited 2", out)


class Plant(unittest.TestCase):
    """The check that this suite can go wrong: classify device skips as passes and the device-skip case reads GREEN."""

    def test_treatingSkipsAsPassesGoesGreenWrongly(self):
        events = [json.loads(l) for l in passing("TestA") + skipping("TestNeedsGPU", "no device: x")]
        real, _ = cds.judge(events, 0, [])
        self.assertTrue(real, "VOID: the real classifier did not flag the device skip")
        saved = cds.DEVICE_SKIP
        cds.DEVICE_SKIP = re.compile(r"(?!)")  # never matches: every skip is a pass
        try:
            planted, _ = cds.judge(events, 0, [])
        finally:
            cds.DEVICE_SKIP = saved
        self.assertFalse(planted, "the plant (skips as passes) must read green, or this test cannot see the defect it plants")


if __name__ == "__main__":
    unittest.main()
