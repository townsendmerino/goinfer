"""BENCH_PHASES narrows bench_peer's phases A and B; unset, both run, as before."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_peer  # noqa: E402


class PlanPhases(unittest.TestCase):
    def setUp(self):
        self.saved = os.environ.pop("BENCH_PHASES", None)

    def tearDown(self):
        os.environ.pop("BENCH_PHASES", None)
        if self.saved is not None:
            os.environ["BENCH_PHASES"] = self.saved

    def test_unset_runs_both(self):
        self.assertEqual(bench_peer.plan_phases(), {"A", "B"})

    def test_b_only(self):
        os.environ["BENCH_PHASES"] = "b"
        self.assertEqual(bench_peer.plan_phases(), {"B"})

    def test_no_phase_refuses(self):
        os.environ["BENCH_PHASES"] = ","
        with self.assertRaises(SystemExit):
            bench_peer.plan_phases()

    def test_unknown_phase_refuses(self):
        os.environ["BENCH_PHASES"] = "B,Z"
        with self.assertRaises(SystemExit):
            bench_peer.plan_phases()


class FreeMemCheck(unittest.TestCase):
    def setUp(self):
        self.saved_env = os.environ.pop("BENCH_ALLOW_NO_FREE_MEM", None)
        self.saved_fn = bench_peer.free_mem_mb

    def tearDown(self):
        bench_peer.free_mem_mb = self.saved_fn
        os.environ.pop("BENCH_ALLOW_NO_FREE_MEM", None)
        if self.saved_env is not None:
            os.environ["BENCH_ALLOW_NO_FREE_MEM"] = self.saved_env

    def test_provenance_records_the_override_only_when_used(self):
        os.environ["BENCH_ALLOW_NO_FREE_MEM"] = "1"
        bench_peer.free_mem_mb = lambda: 60000  # set, but a reading is there
        bench_peer.check_free_mem()
        self.assertFalse(bench_peer.provenance()["free_mem_override"])
        bench_peer.free_mem_mb = lambda: None
        bench_peer.check_free_mem()
        self.assertTrue(bench_peer.provenance()["free_mem_override"])
        bench_peer._FREE_MEM_OVERRIDE_USED = False

    def test_no_reading_refuses_without_the_override(self):
        bench_peer.free_mem_mb = lambda: None
        with self.assertRaises(SystemExit):
            bench_peer.check_free_mem()

    def test_the_override_runs_and_says_so(self):
        bench_peer.free_mem_mb = lambda: None
        os.environ["BENCH_ALLOW_NO_FREE_MEM"] = "1"
        self.assertTrue(bench_peer.check_free_mem())

    def test_a_reading_passes(self):
        bench_peer.free_mem_mb = lambda: 60000
        self.assertFalse(bench_peer.check_free_mem())


class TermExits(unittest.TestCase):
    def test_sigterm_becomes_systemexit_so_the_cells_finally_runs(self):
        import signal
        saved = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
        try:
            with self.assertRaises(SystemExit) as cm:
                bench_peer._term_exits(signal.SIGTERM, None)
            self.assertEqual(cm.exception.code, 128 + signal.SIGTERM)
            # a second signal during the teardown must not interrupt the finally that stops the server
            self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_IGN)
            self.assertEqual(signal.getsignal(signal.SIGINT), signal.SIG_IGN)
            with self.assertRaises(SystemExit) as cm:  # Ctrl-C takes the same path
                bench_peer._term_exits(signal.SIGINT, None)
            self.assertEqual(cm.exception.code, 128 + signal.SIGINT)
        finally:
            signal.signal(signal.SIGTERM, saved[0])
            signal.signal(signal.SIGINT, saved[1])


_TEARDOWN_CHILD = r"""
import os, signal, subprocess, sys, time
sys.path.insert(0, sys.argv[1])
import bench_peer, timing_lock
signal.signal(signal.SIGTERM, bench_peer._term_exits)
with timing_lock.hold("teardown test"):
    try:
        print("ready", flush=True)
        time.sleep(60)
    finally:
        print("teardown", flush=True)
        time.sleep(1.0)  # a cell's teardown: the test sends a second SIGTERM in here
        env = {k: v for k, v in os.environ.items() if k != timing_lock.TOKEN_ENV}
        rc = subprocess.run([sys.executable, os.path.join(sys.argv[1], "timing_lock.py"), "status"], env=env,
                            capture_output=True).returncode
        open(sys.argv[2], "w").write(str(rc))
"""


class TeardownBeforeUnlock(unittest.TestCase):
    def test_sigterm_runs_the_teardown_to_its_end_while_the_lock_is_held(self):
        import signal
        import subprocess
        import tempfile
        scripts = os.path.dirname(os.path.abspath(__file__))
        with tempfile.TemporaryDirectory() as d:
            lock, marker = os.path.join(d, "lock"), os.path.join(d, "marker")
            env = dict(os.environ, GOINFER_TIMING_LOCK_FILE=lock)
            env.pop("GOINFER_TIMING_LOCK", None)
            child = subprocess.Popen([sys.executable, "-c", _TEARDOWN_CHILD, scripts, marker], env=env,
                                     stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), "ready")
                child.send_signal(signal.SIGTERM)
                self.assertEqual(child.stdout.readline().strip(), "teardown")
                child.send_signal(signal.SIGTERM)  # a second signal must not cut the teardown short
                self.assertEqual(child.wait(timeout=30), 128 + signal.SIGTERM)
            finally:
                if child.poll() is None:
                    child.kill()
            with open(marker) as f:
                self.assertEqual(f.read(), "1", "the lock must still be held during the teardown (timing_lock status exits 1)")
            free = subprocess.run([sys.executable, os.path.join(scripts, "timing_lock.py"), "status"], env=env,
                                  capture_output=True)
            self.assertEqual(free.returncode, 0, "the lock must be free once the run has exited")


class ChildSetup(unittest.TestCase):
    def test_a_server_started_under_the_mask_still_stops_on_sigterm(self):
        import signal
        import subprocess
        saved = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # as _term_exits leaves it during a teardown
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            child = subprocess.Popen(["sleep", "30"], preexec_fn=bench_peer._child_setup)
        finally:
            signal.signal(signal.SIGTERM, saved[0])
            signal.signal(signal.SIGINT, saved[1])
        try:
            self.assertEqual(os.getpgid(child.pid), child.pid, "the child leads its own process group")
            child.send_signal(signal.SIGTERM)
            self.assertEqual(child.wait(timeout=10), -signal.SIGTERM)
        finally:
            if child.poll() is None:
                child.kill()


class SignalInstall(unittest.TestCase):
    def test_main_installs_the_handler_for_sigterm_and_sigint(self):
        src = open(bench_peer.__file__).read()
        main = src[src.index('if __name__ == "__main__":'):]
        for sig in ("SIGTERM", "SIGINT"):
            self.assertIn(f"signal.signal(signal.{sig}, _term_exits)", main)


class DepthBackend(unittest.TestCase):
    KEYS = ("BENCH_PHASES", "BENCH_DEPTH_BACKEND", "BENCH_BACKENDS", "BENCH_DEPTHS")

    def setUp(self):
        self.saved = {k: os.environ.pop(k, None) for k in self.KEYS}

    def tearDown(self):
        for k, v in self.saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_phase_a_only_never_reads_it(self):
        os.environ["BENCH_PHASES"] = "A"
        os.environ["BENCH_DEPTH_BACKEND"] = "no-such-backend"
        self.assertIsNone(bench_peer.depth_backend())

    def test_phase_b_refuses_an_unknown_backend(self):
        os.environ["BENCH_PHASES"] = "B"
        os.environ["BENCH_DEPTH_BACKEND"] = "no-such-backend"
        with self.assertRaises(SystemExit):
            bench_peer.depth_backend()

    def test_both_phases_refuse_a_backend_phase_a_did_not_select(self):
        os.environ["BENCH_PHASES"] = "A,B"
        os.environ["BENCH_BACKENDS"] = "cpu"
        with self.assertRaises(SystemExit):
            bench_peer.depth_backend()

    def test_phase_b_alone_refuses_a_backend_bench_backends_excludes(self):
        os.environ["BENCH_PHASES"] = "B"
        os.environ["BENCH_BACKENDS"] = "metal"  # and no BENCH_DEPTH_BACKEND: phase B would run on cuda
        with self.assertRaises(SystemExit):
            bench_peer.depth_backend()

    def test_phase_b_alone_without_bench_backends_takes_the_default(self):
        os.environ["BENCH_PHASES"] = "B"
        self.assertEqual(bench_peer.depth_backend(), "cuda")

    def test_phase_b_default_is_cuda(self):
        os.environ["BENCH_PHASES"] = "B"
        os.environ["BENCH_BACKENDS"] = "cuda"
        self.assertEqual(bench_peer.depth_backend(), "cuda")


class MissingPrompts(unittest.TestCase):
    def test_a_cell_without_a_prompt_refuses_and_a_full_plan_passes(self):
        saved = dict(bench_peer._PROMPTS)
        try:
            bench_peer._PROMPTS["X:512"] = {"text": "a", "tokens": 1}
            bench_peer.missing_prompts([("B", "goinfer", "cuda", "X", 512, "greedy")])
            with self.assertRaises(SystemExit):
                bench_peer.missing_prompts([("A", "goinfer", "cuda", "X", 128, "greedy")])
            bench_peer.missing_prompts([("E", "goinfer", "vision", "vision", 0, "greedy")])  # phases D and E take no prompt
        finally:
            bench_peer._PROMPTS.clear()
            bench_peer._PROMPTS.update(saved)


if __name__ == "__main__":
    unittest.main()
