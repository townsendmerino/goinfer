#!/usr/bin/env python3
"""The day tools hold the timing lock shared while they run: touched.py, `go run ./cmd/gate quick` and refresh_parity_hashes.sh.
A timed run (an exclusive acquire) tried while one of them runs is refused, naming the tool, and the same tool run with
--ignore-timing-lock (no hold) lets it through: that run is the plant, the hold removed, and the probe must see the difference.

Each tool runs for real against a private lock file, with a `git` first on PATH that runs the probe at the tool's first diff or
merge-base (after its hold is taken) and then stops it. Stdlib only; about 15 s, most of it building cmd/gate.
    python3 scripts/test_day_loaders.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import timing_lock  # noqa: E402

PROBE = """import sys
sys.path.insert(0, %r)
import timing_lock as t
try:
    t.acquire(label="probe: a timed run").release()
    print("took")
except t.Held as e:
    print("refused: " + " ".join(str(e).split()))
""" % HERE

FAKE_GIT = """#!/bin/bash
# The probe runs outside the tool's hold (no inherited token), as another session's timed run would.
case "$1" in
merge-base|diff)
	env -u GOINFER_TIMING_LOCK %s -c "$(cat %s)" >> "$PROBE_OUT"
	exit 97 ;;
ls-files) exit 0 ;;
esac
exec %s "$@"
"""


class DayToolsHoldTheLockShared(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.tmp.name
        probe = os.path.join(d, "probe.py")
        with open(probe, "w") as f:
            f.write(PROBE)
        os.mkdir(os.path.join(d, "bin"))
        git = os.path.join(d, "bin", "git")
        with open(git, "w") as f:
            f.write(FAKE_GIT % (sys.executable, probe, shutil.which("git")))
        os.chmod(git, 0o755)
        cls.gate = os.path.join(d, "gate")
        r = subprocess.run(["go", "build", "-o", cls.gate, "./cmd/gate"], cwd=ROOT, capture_output=True, text=True,
                           env=dict(os.environ, GOWORK="off"))
        if r.returncode != 0:
            raise RuntimeError(f"building cmd/gate: {r.stderr}")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_tool(self, argv):
        """Runs the tool with the probe on PATH; returns the probe's first line (what a timed run saw while the tool ran)."""
        d = tempfile.mkdtemp(dir=self.tmp.name)
        out = os.path.join(d, "probe.out")
        env = dict(os.environ, PATH=os.path.join(self.tmp.name, "bin") + os.pathsep + os.environ["PATH"], PROBE_OUT=out,
                   GOINFER_TIMING_LOCK_FILE=os.path.join(d, "timing.lock"), GOWORK="off")
        env.pop(timing_lock.TOKEN_ENV, None)
        env.pop("TOUCHED_BASE", None)
        r = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
        self.assertTrue(os.path.exists(out), f"{argv}: the probe never ran (rc {r.returncode}): {r.stdout[-500:]} {r.stderr[-500:]}")
        with open(out) as f:
            return f.readline().strip()

    def check(self, argv, label, ignore_flag):
        seen = self.run_tool(argv)
        self.assertTrue(seen.startswith("refused: "), f"{label}: a timed run started while it ran: {seen}")
        self.assertIn("loader(s) hold the timing lock shared", seen)
        self.assertIn(f"label {label!r}", seen)
        planted = self.run_tool(argv[:2] + [ignore_flag] + argv[2:])  # PLANT: the same run with no hold
        self.assertEqual(planted, "took", f"{label} with no hold: the probe still saw a holder, so it cannot see the hold")

    def test_touched(self):
        self.check([sys.executable, os.path.join(HERE, "touched.py")], "touched.py", "--ignore-timing-lock")

    def test_refresh_parity_hashes(self):
        self.check(["bash", os.path.join(HERE, "refresh_parity_hashes.sh")], "refresh_parity_hashes.sh", "--ignore-timing-lock")

    def test_gate_quick(self):
        self.check([self.gate, "quick"], "gate quick", "-ignore-timing-lock")

    def test_touched_does_not_start_under_a_timed_run(self):
        lock = os.path.join(self.tmp.name, "held.lock")
        holder = subprocess.Popen([sys.executable, "-c", "import sys, time; sys.path.insert(0, %r); import timing_lock as t; "
                                   "l = t.acquire(label='bench'); print('ready', flush=True); time.sleep(60)" % HERE],
                                  env=dict(os.environ, GOINFER_TIMING_LOCK_FILE=lock), stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(holder.stdout.readline().strip(), "ready")
            env = dict(os.environ, GOINFER_TIMING_LOCK_FILE=lock)
            env.pop(timing_lock.TOKEN_ENV, None)
            r = subprocess.run([sys.executable, os.path.join(HERE, "touched.py")], cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 75, r.stderr)
            self.assertIn("'bench'", r.stderr)
        finally:
            holder.kill()
            holder.wait()
            holder.stdout.close()


if __name__ == "__main__":
    unittest.main()
