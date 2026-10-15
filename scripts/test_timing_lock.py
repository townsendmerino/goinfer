#!/usr/bin/env python3
"""Tests for timing_lock.py (TE9, docs/tasks/task-test-efficiency-2026-09.md): one timed run per box.

Every case runs real processes against a private lock file (GOINFER_TIMING_LOCK_FILE), never ~/.goinfer-timing.lock.
The refusal cases are the ones that must be able to go red: a contender that got the lock while a holder had it would
fail test_contender_is_refused_and_names_the_holder. Run directly:
    python3 scripts/test_timing_lock.py
"""
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import timing_lock  # noqa: E402

TL = os.path.join(HERE, "timing_lock.py")
HOLDER = (
    "import sys, time; sys.path.insert(0, %r); import timing_lock as t\n"
    "l = t.acquire(label='holder-under-test')\n"
    "print('ready', flush=True)\n"
    "time.sleep(float(sys.argv[1]))\n"
) % HERE


class TimingLockTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.lock = os.path.join(self.dir.name, "timing.lock")
        self.env = dict(os.environ, GOINFER_TIMING_LOCK_FILE=self.lock)
        self.env.pop(timing_lock.TOKEN_ENV, None)
        self._saved = {k: os.environ.get(k) for k in ("GOINFER_TIMING_LOCK_FILE", timing_lock.TOKEN_ENV)}
        os.environ["GOINFER_TIMING_LOCK_FILE"] = self.lock
        os.environ.pop(timing_lock.TOKEN_ENV, None)
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
                p.wait()
            p.stdout.close()
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.dir.cleanup()

    def holder(self, secs=30):
        p = subprocess.Popen([sys.executable, "-c", HOLDER, str(secs)], env=self.env, stdout=subprocess.PIPE, text=True)
        self.procs.append(p)
        self.assertEqual(p.stdout.readline().strip(), "ready")
        return p

    def test_free_lock_is_taken_and_released(self):
        lk = timing_lock.acquire(label="solo")
        self.assertFalse(lk.inherited)
        self.assertEqual(timing_lock.read_holder()["label"], "solo")
        lk.release()
        lk2 = timing_lock.acquire(label="again")  # released means takeable
        lk2.release()

    def test_contender_is_refused_and_names_the_holder(self):
        h = self.holder()
        with self.assertRaises(timing_lock.Held) as cm:
            timing_lock.acquire(label="contender")
        msg = str(cm.exception)
        self.assertIn(f"pid {h.pid}", msg)
        self.assertIn("holder-under-test", msg)
        # The CLI wrapper refuses with EX_TEMPFAIL and names the holder, and never runs the command.
        marker = os.path.join(self.dir.name, "ran")
        r = subprocess.run([sys.executable, TL, "run", "--", sys.executable, "-c", f"open({marker!r}, 'w')"],
                           env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 75, r.stderr)
        self.assertIn(f"pid {h.pid}", r.stderr)
        self.assertFalse(os.path.exists(marker), "a refused run must not execute its command")

    def test_a_foreign_token_does_not_bypass(self):
        self.holder()
        os.environ[timing_lock.TOKEN_ENV] = "not-the-holders-token"
        with self.assertRaises(timing_lock.Held):
            timing_lock.acquire(label="impostor")

    def test_children_of_the_holder_proceed(self):
        # The wrapper holds the lock; a nested wrapper and a library acquire inside it both proceed.
        inner = (f"import sys; sys.path.insert(0, {HERE!r}); import timing_lock as t; "
                 "l = t.acquire(label='child'); print('inherited' if l.inherited else 'own')")
        r = subprocess.run([sys.executable, TL, "run", "--label", "outer", "--",
                            sys.executable, TL, "run", "--", sys.executable, "-c", inner],
                           env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "inherited")

    def test_wait_takes_the_lock_when_the_holder_finishes(self):
        self.holder(secs=2)
        t0 = time.monotonic()
        lk = timing_lock.acquire(label="patient", wait_s=20)
        waited = time.monotonic() - t0
        lk.release()
        self.assertGreater(waited, 1.0, "it should have waited for the holder")
        self.assertLess(waited, 15.0)

    def test_a_killed_holder_frees_the_lock(self):
        h = self.holder()
        h.send_signal(signal.SIGKILL)
        h.wait()
        lk = timing_lock.acquire(label="after-kill")  # the kernel dropped the flock with the process
        lk.release()

    def test_status_reports_held_then_free(self):
        h = self.holder()
        r = subprocess.run([sys.executable, TL, "status"], env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn(f"pid {h.pid}", r.stdout)
        h.kill()
        h.wait()
        r = subprocess.run([sys.executable, TL, "status"], env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("free", r.stdout)

    def test_status_under_the_holder_is_free_and_under_another_is_held(self):
        # gate quick asks `status` before it runs; under its own night job it must not refuse its own parent.
        r = subprocess.run([sys.executable, TL, "run", "--label", "outer", "--", sys.executable, TL, "status"],
                           env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("inherited", r.stdout)
        self.holder()
        env = dict(self.env, **{timing_lock.TOKEN_ENV: "not-the-holders-token"})
        r = subprocess.run([sys.executable, TL, "status"], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 1, r.stdout)

    # --- day tools: `check`, touched.py, refresh_parity_hashes.sh and bench_compare.sh refuse beside a timed run ---

    def test_check_refuses_while_held_and_names_the_holder(self):
        h = self.holder()
        r = subprocess.run([sys.executable, TL, "check", "--tool", "probe"], env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 75, r.stderr)
        self.assertIn("REFUSED", r.stderr)
        self.assertIn(f"pid {h.pid}", r.stderr)

    def test_check_override_goes_ahead_with_a_warning(self):
        self.holder()
        r = subprocess.run([sys.executable, TL, "check", "--tool", "probe", "--ignore-timing-lock"], env=self.env,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("WARNING", r.stderr)

    def test_check_waits_for_the_holder(self):
        self.holder(secs=2)
        r = subprocess.run([sys.executable, TL, "check", "--tool", "probe", "--wait", "20"], env=self.env,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_check_is_free_under_its_own_holder_and_when_free(self):
        r = subprocess.run([sys.executable, TL, "check"], env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        r = subprocess.run([sys.executable, TL, "run", "--", sys.executable, TL, "check"], env=self.env,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def scratch_repo(self):
        """A git repo holding copies of touched.py and timing_lock.py and no .go file, so touched.py has nothing to build."""
        repo = os.path.join(self.dir.name, "repo")
        os.makedirs(os.path.join(repo, "scripts"))
        for f in ("touched.py", "timing_lock.py"):
            shutil.copy(os.path.join(HERE, f), os.path.join(repo, "scripts", f))
        for c in (["init", "-q"], ["add", "."], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"]):
            subprocess.run(["git", *c], cwd=repo, check=True, capture_output=True)
        return repo

    def test_touched_refuses_while_held_and_does_nothing(self):
        repo = self.scratch_repo()
        env = dict(self.env, TOUCHED_BASE="HEAD")
        h = self.holder()
        r = subprocess.run([sys.executable, "scripts/touched.py"], cwd=repo, env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 75, r.stdout + r.stderr)
        self.assertIn(f"pid {h.pid}", r.stderr)
        self.assertNotIn("touched:", r.stdout, "a refused run must not start its checks")
        r = subprocess.run([sys.executable, "scripts/touched.py", "--ignore-timing-lock"], cwd=repo, env=env,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("touched: no .go file", r.stdout)

    def test_touched_runs_when_free_and_under_its_own_holder(self):
        repo = self.scratch_repo()
        env = dict(self.env, TOUCHED_BASE="HEAD")
        for cmd in ([sys.executable, "scripts/touched.py"],
                    [sys.executable, TL, "run", "--", sys.executable, "scripts/touched.py"]):
            r = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("touched: no .go file", r.stdout)

    def test_refresh_and_bench_compare_refuse_while_held(self):
        h = self.holder()
        root = os.path.dirname(HERE)
        for script in ("refresh_parity_hashes.sh", "bench_compare.sh"):
            r = subprocess.run(["bash", os.path.join(HERE, script)], cwd=root, env=self.env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 75, f"{script}: {r.stdout}{r.stderr}")
            self.assertIn(f"pid {h.pid}", r.stderr, script)


if __name__ == "__main__":
    unittest.main(verbosity=2)
