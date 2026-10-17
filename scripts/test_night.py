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


if __name__ == "__main__":
    unittest.main()
