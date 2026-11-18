#!/usr/bin/env python3
"""The weekly heavy night's launcher (queue-gate-gpu-weekly.sh) and job wrapper (run-gate-gpu-weekly.sh), against a fake checkout whose night.py and timing_lock.py print what a test sets:
it adds the job once at priority 80, skips when queued, defers (exit 75) at 18:00 when nobara is booked or its lock is foreign, queues at 20:30 regardless; the job exits 75 on a held lock.
Run: python3 -B scripts/test_gate_gpu_weekly_queue.py"""
import os
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = "docs/measurements/gate-gpu-weekly-2026-11-17/"
HELPER = os.path.join(ROOT, D + "queue-gate-gpu-weekly.sh")
JOB = os.path.join(ROOT, D + "run-gate-gpu-weekly.sh")
SCRIPT = D + "run-gate-gpu-weekly.sh"
NIGHT = """import os, sys
a = sys.argv[1:]
d = os.path.dirname(__file__)
open(os.path.join(d, "calls"), "a").write(" ".join(a) + "\\n")
if a[0] in ("list", "status"):
    print(open(os.path.join(d, a[0])).read(), end="")
"""
LOCK = """import os, sys
d = os.path.dirname(__file__)
a = sys.argv[1:]
if a[0] == "status":
    print(open(os.path.join(d, "lock")).read(), end="")
if a[0] == "check":
    sys.exit(int(open(os.path.join(d, "check_rc")).read()))
"""
FREE = "free (/home/x/.goinfer-timing.lock)\n"
IDLE = "idle, 0 queued: last pass none yet\n"


def fake_repo(listing="the night queue is empty\n", status=IDLE, lock=FREE, check_rc=0, with_script=True):
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "scripts"))
    for name, body in (("night.py", NIGHT), ("timing_lock.py", LOCK), ("list", listing), ("status", status), ("lock", lock), ("check_rc", str(check_rc))):
        open(os.path.join(d, "scripts", name), "w").write(body)
    if with_script:
        os.makedirs(os.path.join(d, os.path.dirname(SCRIPT)))
        open(os.path.join(d, SCRIPT), "w").write("#!/bin/bash\n")
    return d


def run(d, hour="18"):
    p = subprocess.run(["bash", HELPER], capture_output=True, text=True, env=dict(os.environ, GOINFER_NIGHT_REPO=d, GATE_WEEKLY_HOUR=hour))
    calls = open(os.path.join(d, "scripts/calls")).read().splitlines() if os.path.exists(os.path.join(d, "scripts/calls")) else []
    return p, [c for c in calls if c.startswith("add ")]


class Queue(unittest.TestCase):
    def test_addsTheJobAtPriority80AfterTheNightlySuite(self):
        p, adds = run(fake_repo())
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(len(adds), 1, adds)
        self.assertTrue(adds[0].startswith("add gate-gpu-weekly --est 150 --priority 80 --not-before 22:00 --retry-until 06:00 "), adds[0])
        self.assertIn(SCRIPT, adds[0])

    def test_theNightlySuiteBeingQueuedDoesNotDeferIt(self):
        p, adds = run(fake_repo("  50  cuda-device-suite                   10 min  22:00-22:10\n"))
        self.assertEqual((p.returncode, len(adds)), (0, 1), p.stdout)

    def test_skipsWhenAlreadyQueued(self):
        p, adds = run(fake_repo("  80  gate-gpu-weekly  150 min\n"))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("already queued", p.stdout)
        self.assertEqual(adds, [])

    def test_aBookedJobDefersTheFirstFiringWithExit75(self):
        p, adds = run(fake_repo("  50  c12-pilot  120 min  22:00-24:00\n"))
        self.assertEqual(p.returncode, 75, p.stdout)
        self.assertEqual(adds, [])

    def test_aRunningOtherJobDefers(self):
        p, adds = run(fake_repo(status="running c12-pilot (since 22:00)\n", lock="held by night.py\n"))
        self.assertEqual((p.returncode, adds), (75, []))

    def test_aForeignLockDefers(self):
        p, adds = run(fake_repo(lock="held by pid 123: tack pilot-serve\n"))
        self.assertEqual((p.returncode, adds), (75, []))

    def test_theSecondFiringQueuesEvenWhenBooked(self):
        p, adds = run(fake_repo("  50  c12-pilot  120 min  22:00-24:00\n", lock="held by pid 123: x\n"), hour="20")
        self.assertEqual((p.returncode, len(adds)), (0, 1), p.stdout)
        self.assertIn("--priority 80", adds[0])

    def test_aCheckoutWithoutTheScriptFailsLoudly(self):
        d = fake_repo(with_script=False)
        p, adds = run(d)
        self.assertNotEqual(p.returncode, 0)
        self.assertEqual(adds, [])


class Job(unittest.TestCase):
    def job(self, rc):
        d = tempfile.mkdtemp()
        # the wrapper resolves SRC from its own location, so copy it into a fake tree
        os.makedirs(os.path.join(d, D))
        os.makedirs(os.path.join(d, "scripts"))
        os.makedirs(os.path.join(d, "docs/measurements/gate-gpu-night-2026-10-06"))
        open(os.path.join(d, "scripts/timing_lock.py"), "w").write(LOCK)
        open(os.path.join(d, "scripts/check_rc"), "w").write(str(rc))
        open(os.path.join(d, "docs/measurements/gate-gpu-night-2026-10-06/run-gate-gpu.sh"), "w").write("echo gate ran\nexit 0\n")
        open(os.path.join(d, D + "run-gate-gpu-weekly.sh"), "w").write(open(JOB).read())
        return subprocess.run(["bash", os.path.join(d, D + "run-gate-gpu-weekly.sh")], capture_output=True, text=True)

    def test_aHeldLockExits75WithoutRunningTheGate(self):
        p = self.job(75)
        self.assertEqual(p.returncode, 75)
        self.assertNotIn("gate ran", p.stdout)

    def test_aFreeLockRunsTheGate(self):
        p = self.job(0)
        self.assertEqual((p.returncode, "gate ran" in p.stdout), (0, True), p.stdout + p.stderr)


if __name__ == "__main__":
    unittest.main()
