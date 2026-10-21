#!/usr/bin/env python3
"""ci_docs_only.sh, rule 2 by full path: a docs commit that adds a docs/**/README.md or run.py is docs-only, while a
change to the root README.md (which Go tests read) or to a docs file a test reads is code. Each case is a real commit in
a scratch worktree of this checkout, classified by the real script. Run: python3 -B scripts/test_ci_docs_only.py"""
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The test's own commits carry this identity, passed per command: a linked worktree shares .git/config with the main
# checkout, so `git config` in it would set the identity of every commit made in the repository afterwards.
IDENTITY = ("-c", "user.name=ci_docs_only test", "-c", "user.email=ci-docs-only@test")


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def git_or_none(cwd, *args):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


class FullPathReads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.wt = os.path.join(cls.tmp, "wt")
        cls.identity = (git_or_none(ROOT, "config", "user.name"), git_or_none(ROOT, "config", "user.email"))
        git(ROOT, "worktree", "add", "-q", "--detach", cls.wt, "HEAD")

    @classmethod
    def tearDownClass(cls):
        subprocess.run(["git", "worktree", "remove", "--force", cls.wt], cwd=ROOT, capture_output=True)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_z_the_repository_identity_is_untouched(self):
        """Runs after the classifying tests (unittest orders by name): the repository's own identity is what it was."""
        self.assertEqual((git_or_none(ROOT, "config", "user.name"), git_or_none(ROOT, "config", "user.email")), self.identity)

    def classify(self, edits):
        """Commit edits (path -> text appended) on top of the worktree's HEAD and classify that one commit."""
        base = git(self.wt, "rev-parse", "HEAD")
        for path, text in edits.items():
            full = os.path.join(self.wt, path)
            os.makedirs(os.path.dirname(full) or self.wt, exist_ok=True)
            with open(full, "a") as f:
                f.write(text)
            git(self.wt, "add", path)
        git(self.wt, *IDENTITY, "commit", "-q", "-m", "test")
        head = git(self.wt, "rev-parse", "HEAD")
        out = subprocess.run(["bash", os.path.join(ROOT, "scripts", "ci_docs_only.sh"), base, head], cwd=self.wt,
                             capture_output=True, text=True).stdout.strip()
        git(self.wt, "reset", "-q", "--hard", base)
        return out

    def test_a_docs_readme_and_run_py_are_docs_only(self):
        self.assertEqual(self.classify({"docs/measurements/zz-ci-docs-only-test/README.md": "# a record\n",
                                        "docs/measurements/zz-ci-docs-only-test/run.py": "print(1)\n"}), "true")

    def test_the_root_readme_is_code(self):
        self.assertEqual(self.classify({"README.md": "\n"}), "false")

    def test_a_docs_file_a_test_reads_is_code(self):
        self.assertEqual(self.classify({"docs/env-vars.md": "\n"}), "false")


if __name__ == "__main__":
    unittest.main()
