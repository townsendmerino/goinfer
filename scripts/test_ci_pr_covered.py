#!/usr/bin/env python3
"""ci_pr_covered.py's decision: a push to main skips the covered lanes only when a green pull-request run recorded the
identical tree, and every other case (a tree mismatch, a failed or unfinished PR run, no PR, a squash or direct push, a
docs-only PR run, a missing artifact, an API error) is "false", so the full matrix runs. Fed fake git, API and artifact
answers. Run: python3 -B scripts/test_ci_pr_covered.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ci_pr_covered  # noqa: E402

SHA = "m" * 40
BASE = "a" * 40
HEAD = "b" * 40
TREE = "t" * 40


class World:
    """One merge of PR #7 (head HEAD) into main at SHA, tree TREE, with one green PR run that recorded TREE."""

    def __init__(self):
        self.parents = [BASE, HEAD]
        self.tree = TREE
        self.prs = [{"number": 7, "merged_at": "2026-11-19T00:00:00Z", "merge_commit_sha": SHA, "head": {"sha": HEAD}}]
        self.runs = [{"id": 100, "head_sha": HEAD, "created_at": "2026-11-18T23:00:00Z", "status": "completed",
                      "conclusion": "success"}]
        self.artifacts = {100: f"tree={TREE}\ndocs_only=false\nsha={'x' * 40}\n"}
        self.api_error = None

    def git(self, *args):
        if args[0] == "rev-list":
            return " ".join([SHA, *self.parents])
        if args[0] == "rev-parse":
            return self.tree
        raise AssertionError(args)

    def api(self, path):
        if self.api_error:
            raise self.api_error
        if path.startswith("commits/"):
            return self.prs
        if path.startswith("actions/workflows/ci.yml/runs"):
            return {"workflow_runs": self.runs}
        raise AssertionError(path)

    def download(self, run_id):
        return self.artifacts.get(run_id)

    def decide(self, ref="refs/heads/main", event="push"):
        return ci_pr_covered.evaluate(SHA, ref, event, self.git, self.api, self.download)


class Decision(unittest.TestCase):
    def assertCovered(self, w, want, reason_part):
        covered, reason = w.decide() if not isinstance(w, tuple) else w[0].decide(**w[1])
        self.assertEqual(covered, want, reason)
        self.assertIn(reason_part, reason)

    def test_match_is_covered(self):
        self.assertCovered(World(), True, "passed on tree")

    def test_tree_mismatch(self):
        w = World()
        w.tree = "u" * 40
        self.assertCovered(w, False, "tested tree")

    def test_failed_pr_run(self):
        w = World()
        w.runs[0]["conclusion"] = "failure"
        self.assertCovered(w, False, "not completed/success")

    def test_unfinished_pr_run(self):
        w = World()
        w.runs[0].update(status="in_progress", conclusion=None)
        self.assertCovered(w, False, "not completed/success")

    def test_newest_pr_run_decides(self):
        """An older green run does not cover a newer red one on the same head."""
        w = World()
        w.runs.append({"id": 101, "head_sha": HEAD, "created_at": "2026-11-18T23:30:00Z", "status": "completed",
                       "conclusion": "failure"})
        w.artifacts[101] = w.artifacts[100]
        self.assertCovered(w, False, "run 101")

    def test_no_pr(self):
        w = World()
        w.prs = []
        self.assertCovered(w, False, "no merged pull request")

    def test_pr_whose_head_is_not_the_second_parent(self):
        w = World()
        w.prs[0]["head"]["sha"] = "c" * 40
        self.assertCovered(w, False, "no merged pull request")

    def test_squash_or_direct_push(self):
        w = World()
        w.parents = [BASE]
        self.assertCovered(w, False, "not a two-parent merge")

    def test_no_pr_run(self):
        w = World()
        w.runs = []
        self.assertCovered(w, False, "no pull_request ci run")

    def test_missing_artifact(self):
        w = World()
        w.artifacts = {}
        self.assertCovered(w, False, "no ci-tree artifact")

    def test_docs_only_pr_run_skipped_the_matrix(self):
        w = World()
        w.artifacts[100] = f"tree={TREE}\ndocs_only=true\n"
        self.assertCovered(w, False, "did not run the full matrix")

    def test_api_error(self):
        w = World()
        w.api_error = RuntimeError("HTTP 502")
        self.assertCovered(w, False, "error: RuntimeError: HTTP 502")

    def test_not_a_push_to_main(self):
        self.assertCovered((World(), {"event": "pull_request"}), False, "not a push to main")
        self.assertCovered((World(), {"ref": "refs/heads/feature"}), False, "not a push to main")


if __name__ == "__main__":
    unittest.main()
