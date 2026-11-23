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

PKG = "github.com/townsendmerino/goinfer/backend/cuda"


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


def run(events, rc=0, *extra):
    """The runner on a fake `go test -json` that prints the events and exits rc. Returns (exit, output)."""
    with tempfile.TemporaryDirectory() as d:
        fake = os.path.join(d, "fake.py")
        with open(fake, "w") as f:
            f.write("import sys\nfor l in %r:\n    print(l)\nsys.exit(%d)\n" % (events, rc))
        p = subprocess.run([sys.executable, os.path.join(HERE, "cuda_device_suite.py"), *extra, "--", sys.executable, fake],
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


class Metal(unittest.TestCase):
    """--backend metal: the Metal module's own skip wording is a device skip, and the summary line carries the backend."""

    def test_eachMetalDeviceSkipWordingIsRed(self):
        for why in ("no metal device: x", "no Metal device: x", "no GPU device: x", "device: open failed", "EnableResident: nope (no Metal device?)",
                    "not resident on Metal: tiny"):
            rc, out = run(passing("TestA") + skipping("TestNeedsGPU", why), 0, "--backend", "metal")
            self.assertEqual(rc, 1, why)
            self.assertIn("metal-device-suite: 1 passed, 0 failed, 1 device skips", out)
            self.assertIn("VERDICT: RED", out)

    def test_theCudaPatternDoesNotCatchMetalWordingAndViceVersa(self):
        rc, _ = run(passing("TestA") + skipping("TestNeedsGPU", "no metal device: x"), 0)
        self.assertEqual(rc, 0)
        rc, _ = run(passing("TestA") + skipping("TestNeedsGPU", "no CUDA device"), 0, "--backend", "metal")
        self.assertEqual(rc, 1)  # "no CUDA device" still matches the shared "no ... device" arm

    def test_assetAndMemoryGuardSkipsAreGreen(self):
        for why in ("GOINFER_HEAVY_TESTS not set", "model not found under ~/models",
                    "not run: the Metal memory guard declined the resident build on this machine right now (x)", "this device makes no residency set: x",
                    "Metal does not build this fixture resident"):
            rc, out = run(passing("TestA") + skipping("TestAsset", why), 0, "--backend", "metal")
            self.assertEqual(rc, 0, (why, out))

    def test_aFailingMetalTestIsRedAndNamed(self):
        rc, out = run(passing("TestA") + failing("TestBroken"), 1, "--backend", "metal")
        self.assertEqual(rc, 1)
        self.assertIn("TestBroken", "\n".join(out.splitlines()[-25:]))


class NewSkips(unittest.TestCase):
    """--state: a test that passed last run and skips now is NEW SKIPS; a reason that is not an asset or opt-in is red, and stays red until it passes."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.state = os.path.join(self.d, "state.json")

    def night(self, events, rc=0):
        return run(events, rc, "--state", self.state)

    def test_theFirstRunHasNothingToCompareWithAndWritesTheState(self):
        rc, out = self.night(passing("TestA") + passing("TestB"))
        self.assertEqual(rc, 0)
        self.assertNotIn("NEW SKIPS", out)
        self.assertEqual(sorted(json.load(open(self.state))["passed"]), [PKG + ".TestA", PKG + ".TestB"])

    def test_aTestThatPassedAndNowSkipsForAnUnexplainedReasonIsRed(self):
        self.night(passing("TestA") + passing("TestB"))
        rc, out = self.night(passing("TestA") + skipping("TestB", "resident path declined"))
        self.assertEqual(rc, 1, out)
        self.assertIn("NEW SKIPS, passed last time", out)
        self.assertIn(PKG + ".TestB", "\n".join(out.splitlines()[-25:]))

    def test_itStaysRedUntilTheTestPassesAgain(self):
        self.night(passing("TestA") + passing("TestB"))
        self.night(passing("TestA") + skipping("TestB", "resident path declined"))
        rc, _ = self.night(passing("TestA") + skipping("TestB", "resident path declined"))
        self.assertEqual(rc, 1)
        rc, _ = self.night(passing("TestA") + passing("TestB"))
        self.assertEqual(rc, 0)

    def test_anAssetOrOptInReasonIsCountedNotRed(self):
        for why in ("GOINFER_HEAVY_TESTS not set", "no checkpoint at ~/models/x", "model not found"):
            self.setUp()
            self.night(passing("TestA") + passing("TestB"))
            rc, out = self.night(passing("TestA") + skipping("TestB", why))
            self.assertEqual(rc, 0, (why, out))
            self.assertIn("NEW SKIPS, asset or opt-in wording", out)

    def test_aTestThatWasAlwaysSkippedIsNotNew(self):
        self.night(passing("TestA") + skipping("TestB", "resident path declined"))
        rc, out = self.night(passing("TestA") + skipping("TestB", "resident path declined"))
        self.assertEqual(rc, 0, out)

    def test_aFailedRunWithNoPassesLeavesTheStateAlone(self):
        self.night(passing("TestA"))
        self.night(failing("TestA"), 1)
        self.assertEqual(json.load(open(self.state))["passed"], [PKG + ".TestA"])

    def test_withoutAStateFileNothingChanges(self):
        rc, out = run(passing("TestA") + skipping("TestB", "resident path declined"))
        self.assertEqual(rc, 0)
        self.assertNotIn("NEW SKIPS", out)


class Tokens(unittest.TestCase):
    """internal/testgate's leading tokens decide before the wording patterns, on both backends: SKIP-DEVICE is a device skip
    whatever follows it, and SKIP-HEAVY / SKIP-ASSET are asset or opt-in skips even when the message mentions a device."""

    def judge(self, why, backend, prev=None):
        events = [json.loads(l) for l in passing("TestA") + skipping("TestB", why)]
        return cds.judge(events, 0, [], backend=backend, prev_passed=prev)

    def test_skipDeviceIsRedWhateverItsWording(self):
        for backend in ("cuda", "metal"):
            red, lines = self.judge("SKIP-DEVICE: primary ctx: CUDA_ERROR_INVALID_CONTEXT", backend)
            self.assertTrue(red, (backend, lines))
            self.assertIn("1 device skips", "\n".join(lines))

    def test_skipHeavyAndSkipAssetAreNotDeviceSkips(self):
        for backend in ("cuda", "metal"):
            for why in ("SKIP-HEAVY: no device needed until GOINFER_HEAVY_TESTS=1", "SKIP-ASSET: no cuda device tower at ~/models/x",
                        "SKIP-MEASURE: no device needed until GOINFER_MEASURE_TESTS=1"):
                red, lines = self.judge(why, backend)
                self.assertFalse(red, (backend, why, lines))

    def test_aTokenedNewSkipIsCountedNotRed(self):
        for why in ("SKIP-ASSET: tower.bin gone", "SKIP-HEAVY: opt in"):
            red, lines = self.judge(why, "cuda", prev={f"{PKG}.TestB"})
            self.assertFalse(red, (why, lines))
            self.assertIn("NEW SKIPS, asset or opt-in wording", "\n".join(lines))

    def test_theTokensAreTestgates(self):
        with open(os.path.join(os.path.dirname(HERE), "internal", "testgate", "testgate.go")) as f:
            src = f.read()
        for tok in (cds.TOKEN_DEVICE, cds.TOKEN_HEAVY, cds.TOKEN_ASSET, cds.TOKEN_MEASURE):
            self.assertIn(f'"{tok}"', src)


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
