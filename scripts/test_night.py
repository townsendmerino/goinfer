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


if __name__ == "__main__":
    unittest.main()
