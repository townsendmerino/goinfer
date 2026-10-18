#!/usr/bin/env python3
"""main_ci_guard.py refuses a push to main while main's latest CI run is unfinished, names it, and passes once it has
completed. Fed fake `gh run list` output through MAIN_CI_GUARD_RUNS. Run: python3 -B scripts/test_main_ci_guard.py"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

GUARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_ci_guard.py")


def run(i, sha, status, wf="ci", event="push", created="2026-10-18T14:00:00Z"):
    return {"databaseId": i, "status": status, "conclusion": "" if status != "completed" else "success", "headSha": sha,
            "event": event, "workflowName": wf, "displayTitle": f"commit {sha[:4]}", "url": f"https://x/runs/{i}",
            "createdAt": created}


def guard(rs):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(rs, f)
    try:
        p = subprocess.run([sys.executable, "-B", GUARD], env={**os.environ, "MAIN_CI_GUARD_RUNS": f.name},
                           capture_output=True, text=True)
        return p.returncode, p.stderr
    finally:
        os.unlink(f.name)


NEW, OLD = "b" * 40, "a" * 40


class MainCIGuard(unittest.TestCase):
    def test_refuses_while_the_latest_run_is_in_progress_and_names_it(self):
        rc, err = guard([run(2, NEW, "in_progress", "coverage", created="2026-10-18T15:00:00Z"),
                         run(3, NEW, "completed", "ci", created="2026-10-18T15:00:00Z"),
                         run(1, OLD, "completed")])
        self.assertEqual(rc, 1)
        self.assertIn("run 2 coverage: in_progress", err)
        self.assertNotIn("run 3", err)

    def test_refuses_a_queued_run(self):
        rc, _ = guard([run(5, NEW, "queued", created="2026-10-18T15:00:00Z"), run(4, OLD, "completed")])
        self.assertEqual(rc, 1)

    def test_passes_once_every_latest_run_completed(self):
        rc, err = guard([run(2, NEW, "completed", "coverage", created="2026-10-18T15:00:00Z"),
                         run(3, NEW, "completed", "ci", created="2026-10-18T15:00:00Z"),
                         run(1, OLD, "in_progress")])  # an older, superseded run does not hold the push
        self.assertEqual(rc, 0, err)

    def test_a_scheduled_run_does_not_hold_the_push(self):
        rc, err = guard([run(9, NEW, "in_progress", "govulncheck", event="schedule", created="2026-10-18T16:00:00Z"),
                         run(3, NEW, "completed", created="2026-10-18T15:00:00Z")])
        self.assertEqual(rc, 0, err)

    def test_an_unreadable_status_refuses(self):
        p = subprocess.run([sys.executable, "-B", GUARD], env={**os.environ, "MAIN_CI_GUARD_RUNS": "/nonexistent.json"},
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        self.assertIn("could not read", p.stderr)


if __name__ == "__main__":
    unittest.main()
