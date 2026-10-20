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


if __name__ == "__main__":
    unittest.main()
