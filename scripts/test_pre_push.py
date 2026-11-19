#!/usr/bin/env python3
"""The pre-push hook's range checks (scripts/pre_push_range.sh, called by scripts/pre-push.sh before its slower checks):
gofmt over the .go files the pushed range changed, and a push to refs/heads/main that is not docs-only
(scripts/ci_docs_only.sh) or a release (a tip whose subject starts "release: ", RELEASING.md) is refused. Each case is
real commits in a scratch worktree of this checkout, fed to the script as git feeds a pre-push hook.
Run: python3 -B scripts/test_pre_push.py"""
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Per-command identity: a linked worktree shares .git/config with the main checkout (as in test_ci_docs_only.py).
IDENTITY = ("-c", "user.name=pre_push test", "-c", "user.email=pre-push@test")
ZERO = "0" * 40
# The scripts under test, copied from this checkout's working tree into the scratch worktree's base commit, so the hook
# (which runs scripts/ from the top of the tree it is pushed from) runs the versions being tested.
SCRIPTS = ("scripts/pre-push.sh", "scripts/pre_push_range.sh", "scripts/ci_docs_only.sh")

UNFORMATTED = "package zz\n\nfunc  F( ) int {return 1}\n"
FORMATTED = "package zz\n\nfunc F() int { return 1 }\n"


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


class PrePushRange(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.wt = os.path.join(cls.tmp, "wt")
        git(ROOT, "worktree", "add", "-q", "--detach", cls.wt, "HEAD")
        for s in SCRIPTS:
            src = os.path.join(ROOT, s)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(cls.wt, s))
                git(cls.wt, "add", s)
        git(cls.wt, *IDENTITY, "commit", "-q", "--allow-empty", "-m", "test: the scripts under test")
        cls.base = git(cls.wt, "rev-parse", "HEAD")

    @classmethod
    def tearDownClass(cls):
        subprocess.run(["git", "worktree", "remove", "--force", cls.wt], cwd=ROOT, capture_output=True)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        git(self.wt, "reset", "-q", "--hard", self.base)

    def commit(self, files, subject="test", remove=()):
        """Write files (path -> full text), remove paths, commit on the worktree's HEAD; returns the new sha."""
        for path, text in files.items():
            full = os.path.join(self.wt, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as f:
                f.write(text)
            git(self.wt, "add", path)
        for path in remove:
            git(self.wt, "rm", "-q", path)
        git(self.wt, *IDENTITY, "commit", "-q", "-m", subject)
        return git(self.wt, "rev-parse", "HEAD")

    def push(self, lsha, rref, rsha, hook=False):
        """Feed one pre-push line to the range script (or to the whole hook) from the worktree; returns the result."""
        line = f"refs/heads/zz {lsha} {rref} {rsha}\n"
        if hook:
            # The hook as installed, under an inherited hold so it takes no timing lock of its own.
            env = dict(os.environ, GOINFER_TIMING_LOCK="inherited")
            cmd = ["bash", os.path.join(self.wt, "scripts", "pre-push.sh"), "origin", "https://example.invalid/goinfer.git"]
        else:
            env = dict(os.environ)
            cmd = ["bash", os.path.join(self.wt, "scripts", "pre_push_range.sh")]
        return subprocess.run(cmd, cwd=self.wt, env=env, input=line, capture_output=True, text=True, timeout=120)

    def assertRefused(self, r, *needles):
        self.assertEqual(r.returncode, 1, f"not refused (rc {r.returncode}): {r.stderr[-800:]}")
        self.assertIn("pre-push REFUSED", r.stderr)
        for n in needles:
            self.assertIn(n, r.stderr)

    def assertPassed(self, r):
        self.assertEqual(r.returncode, 0, f"refused (rc {r.returncode}): {r.stderr[-800:]}")

    # -- gofmt

    def test_an_unformatted_go_file_in_the_range_is_refused(self):
        head = self.commit({"zz_prepush_test/a.go": UNFORMATTED})
        self.assertRefused(self.push(head, "refs/heads/zz", self.base), "gofmt", "zz_prepush_test/a.go")

    def test_a_formatted_go_file_passes(self):
        head = self.commit({"zz_prepush_test/a.go": FORMATTED})
        self.assertPassed(self.push(head, "refs/heads/zz", self.base))

    def test_gofmt_reads_the_pushed_commit_not_the_working_tree(self):
        head = self.commit({"zz_prepush_test/a.go": UNFORMATTED})
        with open(os.path.join(self.wt, "zz_prepush_test/a.go"), "w") as f:
            f.write(FORMATTED)  # fixed in the tree, not committed: the push still carries the unformatted file
        self.assertRefused(self.push(head, "refs/heads/zz", self.base), "zz_prepush_test/a.go")

    def test_only_files_the_range_changed_are_checked(self):
        before = self.commit({"zz_prepush_test/old.go": UNFORMATTED})
        head = self.commit({"zz_prepush_test/new.go": FORMATTED})
        self.assertPassed(self.push(head, "refs/heads/zz", before))

    def test_a_deleted_go_file_is_not_checked(self):
        before = self.commit({"zz_prepush_test/old.go": UNFORMATTED})
        head = self.commit({}, remove=["zz_prepush_test/old.go"])
        self.assertPassed(self.push(head, "refs/heads/zz", before))

    def test_a_new_branch_is_checked_from_its_merge_base_with_origin_main(self):
        head = self.commit({"zz_prepush_test/a.go": UNFORMATTED})
        self.assertRefused(self.push(head, "refs/heads/zz", ZERO), "zz_prepush_test/a.go")

    def test_a_deletion_is_not_checked(self):
        self.assertPassed(self.push(ZERO, "refs/heads/zz", self.base))

    # -- refs/heads/main

    def test_product_code_pushed_to_main_is_refused_with_the_pr_instructions(self):
        head = self.commit({"zz_prepush_test/a.go": FORMATTED})
        self.assertRefused(self.push(head, "refs/heads/main", self.base), "refs/heads/main", "gh pr create", "--auto --merge")

    def test_a_docs_only_push_to_main_passes(self):
        head = self.commit({"docs/zz-prepush-test.md": "# a doc\n"})
        self.assertPassed(self.push(head, "refs/heads/main", self.base))

    def test_a_docs_file_a_test_reads_is_not_docs_only(self):
        with open(os.path.join(self.wt, "docs/env-vars.md")) as f:
            text = f.read()
        head = self.commit({"docs/env-vars.md": text + "\n"})
        self.assertRefused(self.push(head, "refs/heads/main", self.base), "refs/heads/main")

    def test_a_release_push_to_main_passes(self):
        self.commit({"zz_prepush_test/a.go": FORMATTED}, subject="changelog: collect v9.9.9")
        head = self.commit({"go.mod": open(os.path.join(self.wt, "go.mod")).read() + "\n"}, subject="release: v9.9.9 (test)")
        self.assertPassed(self.push(head, "refs/heads/main", self.base))

    def test_a_release_still_needs_gofmt(self):
        head = self.commit({"zz_prepush_test/a.go": UNFORMATTED}, subject="release: v9.9.9 (test)")
        self.assertRefused(self.push(head, "refs/heads/main", self.base), "gofmt")

    def test_release_must_be_the_subject_prefix(self):
        head = self.commit({"zz_prepush_test/a.go": FORMATTED}, subject="decoder: prepare the release: v9.9.9")
        self.assertRefused(self.push(head, "refs/heads/main", self.base), "refs/heads/main")

    def test_code_pushed_to_another_branch_or_a_tag_passes(self):
        head = self.commit({"zz_prepush_test/a.go": FORMATTED})
        self.assertPassed(self.push(head, "refs/heads/zz", self.base))
        self.assertPassed(self.push(head, "refs/tags/v9.9.9", ZERO))

    # -- the installed hook runs them, before its slower checks

    def test_the_hook_refuses_product_code_pushed_to_main(self):
        head = self.commit({"zz_prepush_test/a.go": FORMATTED})
        self.assertRefused(self.push(head, "refs/heads/main", self.base, hook=True), "refs/heads/main")

    def test_the_hook_refuses_an_unformatted_file(self):
        head = self.commit({"zz_prepush_test/a.go": UNFORMATTED})
        self.assertRefused(self.push(head, "refs/heads/zz", self.base, hook=True), "zz_prepush_test/a.go")


if __name__ == "__main__":
    unittest.main()
