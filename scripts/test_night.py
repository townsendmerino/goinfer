#!/usr/bin/env python3
"""Tests for night.py's runner: python3 scripts/test_night.py (stdlib only, seconds)."""
import contextlib
import datetime as dt
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_tmp = tempfile.mkdtemp(prefix="night-test-")
os.environ["NIGHT_DIR"] = os.path.join(_tmp, "night")  # read when night.py is imported
os.environ["GOINFER_TIMING_LOCK_FILE"] = os.path.join(_tmp, "timing.lock")
os.environ["NIGHT_HOOKS"] = os.path.join(_tmp, "night-hooks.toml")  # read when night.py is imported; absent unless a test writes it
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import night  # noqa: E402

_seq = [0]


def queue_job(name, marker):
    """Writes a queue file as `night.py add` does; the job touches marker when it runs."""
    night.QUEUE.mkdir(parents=True, exist_ok=True)
    _seq[0] += 1
    seq = 1_000_000 + _seq[0]
    f = night.QUEUE / f"{seq}-{name}.json"
    f.write_text(json.dumps({"name": name, "cmd": ["/bin/sh", "-c", f"echo ran >> {marker}"], "cwd": _tmp,
                             "est_min": 1.0, "timeout_min": 2.0, "priority": 50, "seq": seq}))
    return f


class Runner(night.Runner):
    """The real runner with no keep-awake process and an instant settle that runs the test's hook once: the window
    between the pick and the move where a drop or re-add lands."""
    hook = None

    def keep_awake(self):
        pass

    def settle(self):
        h, Runner.hook = Runner.hook, None
        if h:
            h()


class FreeMemoryReading(unittest.TestCase):
    def test_reads_goinfers_number(self):
        # B41: the runner's fit-need gate reads the number cmd/hostmem prints (scripts/hostmem.py), the fit guard's own.
        import hostmem
        saved = hostmem.available_mb
        self.addCleanup(setattr, hostmem, "available_mb", saved)
        hostmem.available_mb = lambda *a, **k: 6789
        self.assertEqual(night.free_mem_mb(), 6789)


class RunJobFileVanishes(unittest.TestCase):
    def setUp(self):
        for d in (night.QUEUE, night.RUNNING):
            if d.exists():
                for f in d.glob("*"):
                    f.unlink()
        self.marker = Path(_tmp) / f"ran-{self.id().rsplit('.', 1)[-1]}"
        self.marker.unlink(missing_ok=True)
        self.rundir = Path(tempfile.mkdtemp(dir=_tmp))

    def run_queue(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            Runner(night.now() + dt.timedelta(hours=1), self.rundir).run()
        return out.getvalue()

    def runs(self):
        return self.marker.read_text().count("ran") if self.marker.exists() else 0

    def test_dropped_while_settling_is_skipped(self):
        f = queue_job("dropme", self.marker)
        Runner.hook = f.unlink  # night.py drop, after the runner picked it
        log = self.run_queue()
        self.assertIn("dropme: skipped", log)
        self.assertIn("dropped, so it does not run", log)
        self.assertIn("queue empty", log)
        self.assertEqual(self.runs(), 0)
        self.assertEqual(list(night.RUNNING.glob("*.json")), [])

    def test_readded_while_settling_runs_once_from_its_new_entry(self):
        f = queue_job("readd", self.marker)

        def readd():  # a drop and a re-add: the same name in a new file
            f.unlink()
            queue_job("readd", self.marker)
        Runner.hook = readd
        log = self.run_queue()
        self.assertIn("readd: skipped", log)
        self.assertIn("re-added, so it runs from its new entry", log)
        self.assertIn("readd: ok", log)
        self.assertEqual(self.runs(), 1)
        self.assertEqual(list(night.QUEUE.glob("*.json")), [])

    def test_other_jobs_still_run(self):
        f = queue_job("first", self.marker)
        queue_job("second", self.marker)
        Runner.hook = f.unlink
        log = self.run_queue()
        self.assertIn("first: skipped", log)
        self.assertIn("second: ok", log)
        self.assertEqual(self.runs(), 1)



class SwapGuardPlanCheck(unittest.TestCase):
    """A run plan that pins a serve build carrying the per-process swap guard (its banner text is in the binary) and sets swap_guard_mb is
    refused at `add` unless the plan records Francis's approval and a date; a plan pinning an older build runs as before."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(dir=_tmp))
        for d in (night.QUEUE,):
            if d.exists():
                for f in d.glob("*"):
                    f.unlink()

    def binary(self, name, carries):
        p = self.dir / name
        p.write_bytes(b"\x7fELF" + b"x" * 5000 + (night.GUARD_MARK if carries else b"an older banner") + b"y" * 5000)
        return p

    def plan(self, name, serve, **extra):
        p = self.dir / name
        p.write_text(json.dumps({"serve": str(serve), "commit": "abc", "swap_guard_mb": 4096, **extra}))
        return p

    def add(self, cmd, name="swapcheck"):
        a = type("A", (), dict(name=name, cmd=cmd, est=5, force=False, allow_duplicate=True, cwd=str(self.dir), timeout=None,
                               priority=50, by="", doc="", note=""))()
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            night.cmd_add(a)

    def test_a_new_pin_with_the_key_and_no_approval_is_refused(self):
        plan = self.plan("plan-new.json", self.binary("serve-new", True))
        with self.assertRaises(SystemExit):
            self.add(["python3", "matrix.py", str(plan)])
        self.assertIn("Francis", night.check_plan(plan, self.dir))
        self.assertEqual(list(night.QUEUE.glob("*swapcheck*")), [])

    def test_an_old_pin_with_the_key_runs(self):
        plan = self.plan("plan-old.json", self.binary("serve-old", False))
        self.assertIsNone(night.check_plan(plan, self.dir))
        self.add(["python3", "matrix.py", str(plan)])
        self.assertEqual(len(list(night.QUEUE.glob("*swapcheck*"))), 1)

    def test_approval_needs_francis_and_a_date(self):
        b = self.binary("serve-new", True)
        ok = self.plan("p1.json", b, swap_guard_approved_by="Francis", swap_guard_approved_on="2026-10-17")
        self.assertIsNone(night.check_plan(ok, self.dir))
        for i, extra in enumerate([{"swap_guard_approved_by": "Francis"},                                  # no date
                                   {"swap_guard_approved_by": "claude", "swap_guard_approved_on": "2026-10-17"},  # not him
                                   {"swap_guard_approved_by": "Francis", "swap_guard_approved_on": "today"}]):   # not a date
            self.assertIsNotNone(night.check_plan(self.plan(f"bad{i}.json", b, **extra), self.dir), extra)

    def test_a_plan_without_the_key_is_not_touched(self):
        p = self.dir / "plan-nokey.json"
        p.write_text(json.dumps({"serve": str(self.binary("serve-new", True)), "commit": "abc"}))
        self.assertIsNone(night.check_plan(p, self.dir))

    def test_a_script_that_names_the_plan_is_followed(self):
        plan = self.plan("plan-new.json", self.binary("serve-new", True))
        script = self.dir / "run-x.sh"
        script.write_text(f"#!/bin/bash\nB={self.dir}\nPLAN=$B/{plan.name}\npython3 matrix.py \"$PLAN\"\n")
        self.assertEqual(night.plan_files(["bash", str(script)], self.dir), [plan])
        with self.assertRaises(SystemExit):
            self.add(["bash", str(script)], "viascript")

    def test_the_queued_form_env_var_assignment_is_followed(self):
        """`env B=<dir> bash run-night.sh` with the script's own `B=${B:?...}` and `PLAN=$B/...json`: the shape the real night scripts have."""
        plan = self.plan("plan-new.json", self.binary("serve-new", True))
        script = self.dir / "run-y.sh"
        script.write_text("#!/bin/bash\nB=${B:?the pinned directory}\nPLAN=$B/" + plan.name + "\n")
        self.assertEqual(night.plan_files(["env", f"B={self.dir}", "bash", str(script)], self.dir), [plan])
        with self.assertRaises(SystemExit):
            self.add(["env", f"B={self.dir}", "bash", str(script)], "viaenv")

    def test_plant_without_the_check_the_first_case_is_not_refused(self):
        """The check is what refuses the first case: with it dropped, the same add queues. If this test fails, the first case no longer proves anything."""
        plan = self.plan("plan-new.json", self.binary("serve-new", True))
        real = night.check_job_plans
        night.check_job_plans = lambda cmd, cwd: []
        try:
            self.add(["python3", "matrix.py", str(plan)], "planted")
        finally:
            night.check_job_plans = real
        self.assertEqual(len(list(night.QUEUE.glob("*planted*"))), 1)


class BeforeQueue(unittest.TestCase):
    """G6: the runner runs the box's before_queue command before its first job, and starts nothing when it fails."""

    def setUp(self):
        for d in (night.QUEUE, night.RUNNING):
            if d.exists():
                for f in d.glob("*"):
                    f.unlink()
        night.HOOKS.unlink(missing_ok=True)
        self.addCleanup(night.HOOKS.unlink, missing_ok=True)
        self.log = Path(_tmp) / f"order-{self.id().rsplit('.', 1)[-1]}"
        self.log.unlink(missing_ok=True)
        self.rundir = Path(tempfile.mkdtemp(dir=_tmp))

    def hooks(self, text):
        night.HOOKS.write_text(text)

    def run_queue(self):
        with contextlib.redirect_stdout(io.StringIO()):
            Runner(night.now() + dt.timedelta(hours=1), self.rundir).run()
        return (self.rundir / "SUMMARY.md").read_text()

    def order(self):
        return self.log.read_text().split() if self.log.exists() else []

    def test_the_stop_runs_before_the_first_job(self):
        self.hooks(f"before_queue = \"/bin/sh -c 'echo stop >> {self.log}'\"\n")
        queue_job("first", self.log)
        summary = self.run_queue()
        self.assertEqual(self.order(), ["stop", "ran"])
        self.assertIn("exit 0", summary)

    def test_a_failing_stop_starts_nothing(self):
        self.hooks("before_queue = \"/bin/sh -c 'echo cannot stop; exit 3'\"\n")
        f = queue_job("first", self.log)
        summary = self.run_queue()
        self.assertEqual(self.order(), [], "a job ran after the stop failed")
        self.assertTrue(f.exists(), "the job left the queue")
        self.assertIn("FAILED: exit 3", summary)
        self.assertIn("No job was started", summary)
        self.assertIn("cannot stop", summary)

    def test_no_hooks_file_runs_the_queue(self):
        queue_job("first", self.log)
        summary = self.run_queue()
        self.assertEqual(self.order(), ["ran"])
        self.assertIn("sets no before_queue", summary)

    def test_an_unreadable_hooks_file_starts_nothing(self):
        self.hooks("before_queue = [not toml\n")
        queue_job("first", self.log)
        summary = self.run_queue()
        self.assertEqual(self.order(), [])
        self.assertIn("not readable TOML", summary)

    def test_a_stop_that_hangs_starts_nothing(self):
        self.hooks("before_queue = \"/bin/sleep 5\"\n")
        queue_job("first", self.log)
        prev, night.HOOK_TIMEOUT_S = night.HOOK_TIMEOUT_S, 1
        try:
            summary = self.run_queue()
        finally:
            night.HOOK_TIMEOUT_S = prev
        self.assertEqual(self.order(), [])
        self.assertIn("timed out after 1 s", summary)

    def test_the_stop_runs_under_the_timing_lock(self):
        status = Path(_tmp) / "lock-during-stop"
        lock_py = Path(__file__).resolve().parent / "timing_lock.py"
        self.hooks(f"before_queue = \"/bin/sh -c '{sys.executable} {lock_py} status > {status}'\"\n")
        queue_job("first", self.log)
        self.run_queue()
        self.assertIn("night.py before_queue", status.read_text())


class QueueEnvironment(unittest.TestCase):
    """The box's hooks file sets the jobs' PATH once (path_prepend) and names tools that must be found on it (require_on_path):
    a missing tool refuses the start, with every job still queued and SUMMARY.md saying what was missing."""

    def setUp(self):
        for d in (night.QUEUE, night.RUNNING):
            if d.exists():
                for f in d.glob("*"):
                    f.unlink()
        self.dir = Path(tempfile.mkdtemp(dir=_tmp))
        self.hooks = self.dir / "night-hooks.toml"
        self._hooks = night.HOOKS
        night.HOOKS = self.hooks
        self.summary = night.RUNS / night.now().strftime("%Y-%m-%d") / "SUMMARY.md"
        self.summary.unlink(missing_ok=True)

    def tearDown(self):
        night.HOOKS = self._hooks

    def toml(self, body):
        self.hooks.write_text(body)

    def job(self, name="envjob", script="true"):
        night.QUEUE.mkdir(parents=True, exist_ok=True)
        _seq[0] += 1
        seq = 2_000_000 + _seq[0]
        f = night.QUEUE / f"{seq}-{name}.json"
        f.write_text(json.dumps({"name": name, "cmd": ["/bin/sh", "-c", script], "cwd": str(self.dir), "est_min": 1.0,
                                 "timeout_min": 2.0, "priority": 50, "seq": seq}))
        return f

    def start(self, dry_run):
        a = type("A", (), dict(deadline=night.DEFAULT_DEADLINE, dry_run=dry_run, frm=None))()
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            try:
                night.cmd_start(a)
            except SystemExit as e:
                return out.getvalue(), e.code
        return out.getvalue(), None

    def run_queue(self):
        rundir = Path(tempfile.mkdtemp(dir=_tmp))
        with contextlib.redirect_stdout(io.StringIO()):
            Runner(night.now() + dt.timedelta(hours=1), rundir).run()
        return rundir

    def test_the_job_environment_gets_the_prepended_path(self):
        bindir = self.dir / "bin"
        bindir.mkdir()
        self.toml(f'path_prepend = ["{bindir}"]\n')
        out = self.dir / "path-seen"
        self.job(script=f'printf %s "$PATH" > {out}')
        self.run_queue()
        seen = out.read_text()
        self.assertTrue(seen.startswith(str(bindir) + os.pathsep), seen)
        self.assertIn(os.environ["PATH"], seen)

    def test_a_missing_required_tool_refuses_the_start_and_leaves_the_queue_intact(self):
        self.toml('require_on_path = ["no-such-tool-for-night-test"]\n')
        f = self.job()
        spawned = []
        real = night.subprocess.Popen
        night.subprocess.Popen = lambda *a, **k: spawned.append(a) or (_ for _ in ()).throw(AssertionError("a runner was spawned"))
        try:
            out, code = self.start(dry_run=False)
        finally:
            night.subprocess.Popen = real
        self.assertEqual(code, 1, out)
        self.assertEqual(spawned, [], "no runner may be spawned")
        self.assertTrue(f.exists(), "the job's queue file must stay")
        self.assertIsNone(night.runner_pid(), "no runner may have been started")
        text = self.summary.read_text()
        self.assertIn("no-such-tool-for-night-test", text)
        self.assertIn("Not started", text)
        self.assertIn("envjob", text)

    def test_the_dry_run_prints_the_result(self):
        self.job()
        self.toml('require_on_path = ["sh", "no-such-tool-for-night-test"]\n')
        out, code = self.start(dry_run=True)
        self.assertEqual(code, 1)
        self.assertIn("NOT FOUND", out)
        self.assertRegex(out, r"sh: /\S+/sh")
        self.assertFalse(self.summary.exists(), "a dry run writes no SUMMARY.md")
        self.toml('require_on_path = ["sh"]\n')
        out, code = self.start(dry_run=True)
        self.assertIsNone(code, out)
        self.assertIn("(dry run — not started)", out)

    def test_path_prepend_makes_a_tool_found(self):
        bindir = self.dir / "bin"
        bindir.mkdir()
        tool = bindir / "night-test-tool"
        tool.write_text("#!/bin/sh\n")
        tool.chmod(0o755)
        self.job()
        self.toml(f'require_on_path = ["night-test-tool"]\n')
        self.assertEqual(self.start(dry_run=True)[1], 1)
        self.toml(f'path_prepend = ["{bindir}"]\nrequire_on_path = ["night-test-tool"]\n')
        self.assertIsNone(self.start(dry_run=True)[1])

    def test_a_malformed_setting_refuses_too(self):
        self.job()
        self.toml('require_on_path = "go"\n')
        out, code = self.start(dry_run=True)
        self.assertEqual(code, 1)

    def test_the_runner_checks_as_well(self):
        self.toml('require_on_path = ["no-such-tool-for-night-test"]\n')
        f = self.job(script=f"echo ran > {self.dir / 'ran'}")
        rundir = self.run_queue()
        self.assertFalse((self.dir / "ran").exists(), "no job may start")
        self.assertTrue(f.exists())
        self.assertIn("no-such-tool-for-night-test", (rundir / "SUMMARY.md").read_text())

    def test_plant_without_the_check_the_start_is_not_refused(self):
        """With the check skipped the same dry run goes through: the refusal tests above need the check to be what refuses."""
        self.toml('require_on_path = ["no-such-tool-for-night-test"]\n')
        self.job()
        real = night.missing_required
        night.missing_required = lambda: ([], None)
        try:
            out, code = self.start(dry_run=True)
        finally:
            night.missing_required = real
        self.assertIsNone(code, out)


class LogFailureMarkers(unittest.TestCase):
    """A job that exits 0 but prints a failure marker (--- FAIL, a FAIL line, FAILED, VOID) in its own output is FAILED; a line tagged
    [expected] (a plant's red run) is exempt, and the runner's header (a note naming VOID) is not scanned."""

    def setUp(self):
        for d in (night.QUEUE, night.RUNNING):
            if d.exists():
                for f in d.glob("*"):
                    f.unlink()
        self.rundir = Path(tempfile.mkdtemp(dir=_tmp))

    def outcome(self, script, note=None):
        night.QUEUE.mkdir(parents=True, exist_ok=True)
        _seq[0] += 1
        seq = 1_000_000 + _seq[0]
        job = {"name": f"marker-{seq}", "cmd": ["/bin/sh", "-c", script], "cwd": _tmp, "est_min": 1.0, "timeout_min": 2.0,
               "priority": 50, "seq": seq}
        if note:
            job["note"] = note
        (night.QUEUE / f"{seq}-marker.json").write_text(json.dumps(job))
        r = Runner(night.now() + dt.timedelta(hours=1), self.rundir)
        with contextlib.redirect_stdout(io.StringIO()):
            r.run()
        return [x["outcome"] for x in r.results if x["name"] == job["name"]][0]

    def test_markers_in_a_zero_exit_job_fail_it(self):
        for line in ["--- FAIL: TestSomething (0.01s)", "    --- FAIL: TestSomething/sub (0.00s)", "FAIL\tgithub.com/x/y\t0.5s",
                     "probe new FAILED", "VOID: no model at /m", "VOID-refused: the fit check read REFUSED"]:
            got = self.outcome(f"echo 'step one'; echo '{line}'; exit 0")
            self.assertTrue(got.startswith("FAILED rc=0"), f"{line!r}: outcome {got!r}")

    def test_expected_lines_and_the_header_are_not_markers(self):
        self.assertEqual(self.outcome("echo '[expected] --- FAIL: TestPlant (0.1s)'; echo '[expected] FAIL'; exit 0"), "ok")
        self.assertEqual(self.outcome("echo 'all good: 0 FAILURES'; exit 0", note="VOID if the load is refused"), "ok")
        self.assertEqual(self.outcome("echo clean; exit 0"), "ok")

    def test_a_nonzero_exit_is_still_its_rc(self):
        self.assertEqual(self.outcome("exit 3"), "FAILED rc=3")

    def test_plant_without_the_scan_a_failure_reads_ok(self):
        orig = night.failure_marker
        night.failure_marker = lambda logp, start: None
        try:
            got = self.outcome("echo 'probe new FAILED'; exit 0")
        finally:
            night.failure_marker = orig
        self.assertEqual(got, "ok", "with the scan removed the job reads ok: the scan is what catches it")


class FakeServer(night.Server):
    """serve with a stand-in clock, load and memory, and a sleep that only moves the clock; on_sleep runs inside each sleep (the settle)."""

    def __init__(self):
        self.t = night.now()
        self.la, self.free, self.on_sleep, self.awake_calls = 0.0, 64000, None, []
        super().__init__()

    def clock(self):
        return self.t

    def load1(self):
        return self.la

    def ncores(self):
        return 8

    def free_mb(self):
        return self.free

    def on_ac(self):
        return True

    def sleep(self, s):
        if self.on_sleep:
            self.on_sleep()
        self.t += dt.timedelta(seconds=s)

    def hold_awake(self, on):
        self.awake_calls.append(on)


class Serve(unittest.TestCase):
    """B38: the always-on runner starts a job only when its gates allow, takes the timing lock before settling, holds nothing while idle,
    and re-queues a job whose own check refused (exit RETRY_RC)."""

    def setUp(self):
        # The clock runs as if it were 20:00 today, so a test's hours of not-before, settle and estimate never meet the 06:30
        # deadline because of the hour CI happens to run at (UTC on the runner); it still advances, for night.py's timeouts.
        real = night.now
        off = real().replace(hour=20, minute=0, second=0, microsecond=0) - real()
        night.now = lambda: real() + off
        self.addCleanup(setattr, night, "now", real)
        for d in (night.QUEUE, night.RUNNING):
            if d.exists():
                for f in d.glob("*"):
                    f.unlink()
        night.PAUSEFLAG.unlink(missing_ok=True)
        self.marker = Path(_tmp) / f"ran-{self.id().rsplit('.', 1)[-1]}"
        self.marker.unlink(missing_ok=True)

    def job(self, name="j", script=None, **extra):
        night.QUEUE.mkdir(parents=True, exist_ok=True)
        _seq[0] += 1
        seq = 1_000_000 + _seq[0]
        j = {"name": name, "cmd": ["/bin/sh", "-c", script or f"echo ran >> {self.marker}"], "cwd": _tmp, "est_min": 30.0,
             "timeout_min": 2.0, "priority": 50, "seq": seq, **extra}
        f = night.QUEUE / f"{seq}-{name}.json"
        f.write_text(json.dumps(j))
        return f

    def tick(self, sv):
        with contextlib.redirect_stdout(io.StringIO()):
            return sv.tick()

    def runs(self):
        return self.marker.read_text().count("ran") if self.marker.exists() else 0

    def lock_is_free(self):
        try:
            night.timing_lock.acquire(label="probe", wait_s=0).release()
            return True
        except night.timing_lock.Held:
            return False

    def test_not_before_waits_then_runs(self):
        sv = FakeServer()
        self.job(not_before=(sv.t + dt.timedelta(hours=2)).isoformat(timespec="seconds"))
        self.assertEqual(self.tick(sv), "idle")
        self.assertIn("night-only", sv.waits[0])
        self.assertEqual(self.runs(), 0)
        sv.t += dt.timedelta(hours=2, minutes=1)
        self.assertEqual(self.tick(sv), "ran")
        self.assertEqual(self.runs(), 1)

    def test_a_night_job_that_would_end_after_the_deadline_waits(self):
        sv = FakeServer()
        start = sv.t - dt.timedelta(hours=1)
        end = night.night_end(start)
        sv.t = end - dt.timedelta(minutes=10)  # a 30-minute job 10 minutes before the deadline
        self.job(not_before=start.isoformat(timespec="seconds"))
        self.assertEqual(self.tick(sv), "idle")
        self.assertIn("would end after", sv.waits[0])

    def test_heavy_load_holds_the_start(self):
        sv = FakeServer()
        self.job()
        sv.la = 4.5  # half of 8 cores is 4.0
        self.assertEqual(self.tick(sv), "idle")
        self.assertIn("load 4.50 above half of 8 cores", sv.waits[0])
        sv.la = 3.9
        self.assertEqual(self.tick(sv), "ran")

    def test_low_free_memory_holds_the_start(self):
        sv = FakeServer()
        self.job(fit_need_gb=4)
        sv.free = 4 * 1024 + 2048 - 1
        self.assertEqual(self.tick(sv), "idle")
        self.assertIn("free memory", sv.waits[0])
        sv.free = 4 * 1024 + 2048
        self.assertEqual(self.tick(sv), "ran")

    def test_plant_without_the_heavy_gate_it_runs_under_load(self):
        sv = FakeServer()
        sv.heavy = lambda j: None  # PLANT: the gate removed
        self.job(fit_need_gb=4)
        sv.la, sv.free = 7.0, 100
        self.assertEqual(self.tick(sv), "ran", "with the gate removed the job starts: the gate is what held it above")

    def test_load_during_the_settle_holds_the_start_and_frees_the_lock(self):
        sv = FakeServer()
        f = self.job()
        sv.on_sleep = lambda: setattr(sv, "la", 6.0)
        self.assertTrue(self.tick(sv).startswith("settle: load 6.00"))
        self.assertEqual(self.runs(), 0)
        self.assertTrue(f.exists(), "the job left the queue")
        self.assertTrue(self.lock_is_free(), "the lock was kept after a failed settle")

    def test_the_lock_is_held_through_the_settle(self):
        seen = []
        sv = FakeServer()
        sv.on_sleep = lambda: seen.append(night.timing_lock.read_holder(night.timing_lock.lock_path()).get("label", ""))
        self.job(name="lockcheck")
        self.assertEqual(self.tick(sv), "ran")
        self.assertTrue(seen and all("settling for lockcheck" in x for x in seen), seen)

    def test_plant_without_the_lock_the_settle_is_unguarded(self):
        seen = []
        sv = FakeServer()
        sv.on_sleep = lambda: seen.append(night.timing_lock.read_holder(night.timing_lock.lock_path()).get("label", ""))

        class NoLock:
            def release(self):
                pass
        orig = night.timing_lock.acquire
        night.timing_lock.acquire = lambda **kw: NoLock()  # PLANT: no lock taken before the settle
        try:
            self.job(name="lockcheck")
            self.tick(sv)
        finally:
            night.timing_lock.acquire = orig
        self.assertFalse(any("settling for" in x for x in seen), "with no lock taken the check still saw one: it cannot see the lock")

    def test_another_holder_means_wait(self):
        sv = FakeServer()
        f = self.job()
        fd = os.open(night.timing_lock.lock_path(), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            night.fcntl_flock(fd)
            self.assertEqual(self.tick(sv), "lock held")
        finally:
            os.close(fd)
        self.assertEqual(self.runs(), 0)
        self.assertTrue(f.exists())

    def test_idle_holds_no_lock(self):
        sv = FakeServer()
        self.assertEqual(self.tick(sv), "idle")
        self.assertTrue(self.lock_is_free())
        self.job(not_before=(sv.t + dt.timedelta(hours=1)).isoformat(timespec="seconds"))
        self.assertEqual(self.tick(sv), "idle")
        self.assertTrue(self.lock_is_free())

    def test_pause_starts_nothing(self):
        sv = FakeServer()
        self.job()
        night.PAUSEFLAG.write_text("x")
        try:
            self.assertEqual(self.tick(sv), "paused")
        finally:
            night.PAUSEFLAG.unlink()
        self.assertEqual(self.runs(), 0)

    def test_exit_75_requeues_after_the_backoff(self):
        sv = FakeServer()
        self.job(name="refuses", script="echo 'fit: REFUSED'; exit 75",
                 retry_until=(night.now() + dt.timedelta(hours=3)).isoformat(timespec="seconds"))
        self.assertEqual(self.tick(sv), "ran")
        self.assertTrue(sv.results[-1]["outcome"].startswith("retry 1"), sv.results[-1]["outcome"])
        (q,) = night.queued()
        self.assertEqual(q["attempts"], 1)
        self.assertEqual(self.tick(sv), "idle")
        self.assertIn("retry at", sv.waits[0])
        sv.t = dt.datetime.fromisoformat(q["next_try"]) + dt.timedelta(seconds=1)
        self.assertEqual(self.tick(sv), "ran")
        self.assertEqual(night.queued()[0]["attempts"], 2)

    def test_exit_75_past_retry_until_is_void(self):
        sv = FakeServer()
        self.job(name="refuses", script="exit 75", retry_until=(night.now() + dt.timedelta(minutes=5)).isoformat(timespec="seconds"))
        self.tick(sv)
        self.assertTrue(sv.results[-1]["outcome"].startswith("VOID-refused: its check refused 1 time(s)"), sv.results[-1]["outcome"])
        self.assertEqual(night.queued(), [])

    def test_exit_75_without_retry_until_stops_at_retry_max(self):
        sv = FakeServer()
        prev, night.RETRY_MAX = night.RETRY_MAX, 2
        try:
            self.job(name="refuses", script="exit 75")
            self.tick(sv)
            sv.t = dt.datetime.fromisoformat(night.queued()[0]["next_try"]) + dt.timedelta(seconds=1)
            self.tick(sv)
        finally:
            night.RETRY_MAX = prev
        self.assertTrue(sv.results[-1]["outcome"].startswith("VOID-refused: its check refused 2 time(s), 2 tries"), sv.results[-1]["outcome"])
        self.assertEqual(night.queued(), [])

    def test_plant_without_retries_exit_75_fails(self):
        sv = FakeServer()
        sv.retries = False  # PLANT: the bedtime runner's reading
        self.job(name="refuses", script="exit 75")
        self.tick(sv)
        self.assertEqual(sv.results[-1]["outcome"], "FAILED rc=75")

    def test_a_sleep_is_reported_in_the_summary(self):
        sv = FakeServer()
        d = [1000.0]
        sv.drift = lambda: d[0]
        self.tick(sv)
        d[0] += 30  # under a minute: clock noise, not a sleep
        self.tick(sv)
        self.assertEqual(sv.slept, [])
        d[0] += 45 * 60
        self.tick(sv)
        self.assertEqual(len(sv.slept), 1)
        self.assertIn("The machine slept", (sv.rundir / "SUMMARY.md").read_text())

    def missed_setup(self):
        """A night-only job whose night (22:00 yesterday to 06:30 today) has just ended without a start, with one refusal on record."""
        sv = FakeServer()
        today = sv.t.replace(hour=6, minute=31, second=0, microsecond=0)
        sv.t = today
        nb = (today - dt.timedelta(days=1)).replace(hour=22, minute=0)
        f = self.job(name="missed", not_before=nb.isoformat(timespec="seconds"), attempts=1,
                     retry_until=today.replace(hour=6, minute=0).isoformat(timespec="seconds"))
        return sv, f, today

    def test_a_missed_night_moves_to_the_next(self):
        sv, f, today = self.missed_setup()
        sv.la = 9.0
        sv.t = today - dt.timedelta(minutes=30)  # 06:01: the last wait is the load
        self.tick(sv)
        sv.t, sv.la = today, 0.0
        self.assertEqual(self.tick(sv), "idle")
        j = json.loads(f.read_text())
        self.assertEqual(j["not_before"], today.replace(hour=22, minute=0).isoformat(timespec="seconds"))
        self.assertEqual(j["retry_until"], (today + dt.timedelta(days=1)).replace(hour=6, minute=0).isoformat(timespec="seconds"))
        self.assertEqual((j["attempts"], j["missed_nights"]), (0, 1))
        summary = (sv.rundir / "SUMMARY.md").read_text()
        self.assertIn("## Missed its night", summary)
        self.assertIn("missed: the night", summary)
        self.assertIn("would end after the 06:30 deadline", summary)  # its last wait, 06:01 + 30 min
        sv.t = today.replace(hour=22, minute=1)
        self.assertEqual(self.tick(sv), "ran")
        self.assertEqual(self.runs(), 1)

    def test_plant_without_the_move_a_missed_job_never_runs_again(self):
        sv, f, today = self.missed_setup()
        sv.roll_missed = lambda j, t: False  # PLANT: no move
        for t in (today, today.replace(hour=22, minute=1), today.replace(hour=23) + dt.timedelta(days=3)):
            sv.t = t
            self.assertEqual(self.tick(sv), "idle", t)
        self.assertEqual(self.runs(), 0, "with the move removed the job still ran: the move is not what the test above sees")

    def test_status_reports_the_settle_as_busy(self):
        seen = []
        sv = FakeServer()
        sv.since = "x"
        night.PIDFILE.write_text(str(os.getpid()))
        self.addCleanup(lambda: [p.unlink(missing_ok=True) for p in (night.PIDFILE, night.SERVEFILE)])
        sv.on_sleep = lambda: seen.append(night.status_json())
        self.job(name="settler")
        self.assertEqual(self.tick(sv), "ran")
        s = seen[0]
        self.assertEqual((s["idle"], s["running"], s["phase"]), (False, "settler", "settling"))
        sv.publish()
        s = night.status_json()
        self.assertEqual((s["idle"], s["running"], s["phase"]), (True, None, "idle"))

    def test_plant_without_the_publish_the_settle_reads_idle(self):
        seen = []
        sv = FakeServer()
        sv.since = "x"
        night.PIDFILE.write_text(str(os.getpid()))
        night.SERVEFILE.unlink(missing_ok=True)
        self.addCleanup(lambda: [p.unlink(missing_ok=True) for p in (night.PIDFILE, night.SERVEFILE)])
        sv.publish = lambda: None  # PLANT: the settle is not published
        night.write_json(night.SERVEFILE, {"pid": os.getpid(), "since": "x", "phase": "idle"})
        sv.on_sleep = lambda: seen.append(night.status_json())
        self.job(name="settler")
        self.tick(sv)
        self.assertTrue(seen[0]["idle"], "with the publish removed the settle still read busy: the test does not see the publish")

    def test_a_night_job_due_holds_the_machine_awake(self):
        sv = FakeServer()
        self.assertEqual(self.tick(sv), "idle")
        self.assertEqual(sv.awake_calls[-1], False)
        self.job(not_before=(sv.t + dt.timedelta(hours=3)).isoformat(timespec="seconds"))
        self.tick(sv)
        self.assertEqual(sv.awake_calls[-1], True)


class ServeStatus(unittest.TestCase):
    def setUp(self):
        for f in (night.PIDFILE, night.SERVEFILE, night.STATE):
            f.unlink(missing_ok=True)
        self.addCleanup(lambda: [f.unlink(missing_ok=True) for f in (night.PIDFILE, night.SERVEFILE, night.STATE)])

    def test_no_runner(self):
        s = night.status_json()
        self.assertIsNone(s["runner"])
        self.assertTrue(s["idle"])

    def test_serve_idle_then_busy(self):
        night.PIDFILE.write_text(str(os.getpid()))
        night.write_json(night.SERVEFILE, {"pid": os.getpid(), "since": "x", "last": "idle", "waits": ["j: load"]})
        s = night.status_json()
        self.assertEqual((s["runner"], s["idle"], s["running"], s["waits"]), ("serve", True, None, ["j: load"]))
        night.write_json(night.STATE, {"name": "busyjob"})
        s = night.status_json()
        self.assertEqual((s["idle"], s["running"]), (False, "busyjob"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            night.cmd_status(type("A", (), {"json": True})())
        self.assertEqual(json.loads(out.getvalue())["running"], "busyjob")

    def test_the_first_line_names_the_state(self):
        # tack's launcher reads a box as busy when this line starts with "running " (or "runner going", the bedtime runner).
        def first():
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                night.cmd_status(type("A", (), {"json": False})())
            return out.getvalue().splitlines()[0]
        night.PIDFILE.write_text(str(os.getpid()))
        night.write_json(night.SERVEFILE, {"pid": os.getpid(), "since": "x", "last": "idle", "waits": []})
        self.assertTrue(first().startswith("idle, "), first())
        night.write_json(night.STATE, {"name": "busyjob", "started": night.now().isoformat(), "est_min": 1, "timeout_min": 2, "log": "/nonexistent"})
        self.assertTrue(first().startswith("running busyjob "), first())
        night.SERVEFILE.unlink()
        self.assertTrue(first().startswith("runner going"), first())

    def test_the_bedtime_runner_is_not_serve(self):
        night.PIDFILE.write_text(str(os.getpid()))
        self.assertEqual(night.status_json()["runner"], "night")


class NightStart(unittest.TestCase):
    def at(self, h, m=0):
        return dt.datetime(2026, 10, 20, h, m)

    def test_window(self):
        self.assertEqual(night.night_start("22:00", self.at(23)), self.at(22))
        self.assertEqual(night.night_start("22:00", self.at(7)), self.at(22))
        self.assertEqual(night.night_start("22:00", self.at(2)), self.at(22) - dt.timedelta(days=1))
        self.assertEqual(night.night_end(self.at(22)), self.at(6, 30) + dt.timedelta(days=1))


if __name__ == "__main__":
    unittest.main()
