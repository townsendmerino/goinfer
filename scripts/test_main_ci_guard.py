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


def guard(rs, args=(), docs_only=None):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(rs, f)
    env = {**os.environ, "MAIN_CI_GUARD_RUNS": f.name}
    env.pop("MAIN_CI_GUARD_DOCS_ONLY", None)
    if docs_only is not None:
        env["MAIN_CI_GUARD_DOCS_ONLY"] = docs_only
    try:
        p = subprocess.run([sys.executable, "-B", GUARD, *args], env=env, capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.dirname(GUARD)))
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


RUNNING = [run(2, NEW, "in_progress", created="2026-10-18T15:00:00Z")]


class TheSplitDocsAndCode(unittest.TestCase):
    """A docs-only push goes through while main's CI runs; a push that changes product code waits. ci_docs_only.sh
    classifies the push's range, as CI's gate does."""

    def test_a_docs_only_push_goes_through_while_main_runs(self):
        rc, err = guard(RUNNING, ("a" * 40, "b" * 40), docs_only="true")
        self.assertEqual(rc, 0, err)
        self.assertIn("docs-only push", err)

    def test_a_code_push_waits_while_main_runs(self):
        rc, err = guard(RUNNING, ("a" * 40, "b" * 40), docs_only="false")
        self.assertEqual(rc, 1)
        self.assertIn("run 2", err)

    def test_a_push_without_a_range_is_held_like_code(self):
        rc, _ = guard(RUNNING, docs_only="true")
        self.assertEqual(rc, 1)

    def test_real_history_through_the_real_classifier(self):
        """53267aee changed docs/queue-engineering.md only; 1cc27f17 changed metal/*.go."""
        root = os.path.dirname(os.path.dirname(GUARD))
        sha = lambda r: subprocess.run(["git", "rev-parse", "--verify", "-q", r], cwd=root, capture_output=True, text=True).stdout.strip()
        docs, code = (sha("53267aee^"), sha("53267aee")), (sha("1cc27f17^"), sha("1cc27f17"))
        if not all(docs + code):
            self.skipTest("the reference commits are not in this checkout's history")
        rc, err = guard(RUNNING, docs)
        self.assertEqual(rc, 0, err)
        rc, err = guard(RUNNING, code)
        self.assertEqual(rc, 1, err)


if __name__ == "__main__":
    unittest.main()
