#!/usr/bin/env python3
"""Tests for queue_citation_lint.py's J0 fix (task-work-queue-2026-09.md) — an untracked live doc
must not red the lint, and the same doc must red once it IS tracked. Mutation-checked: builds an
isolated temp git repo (never touches this checkout's own git state) so `git add`/commit here have
no side effect on the real tree. Run directly:
    python3 scripts/test_queue_citation_lint.py
"""
import contextlib
import importlib.util
import io
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "queue_citation_lint", os.path.join(HERE, "queue_citation_lint.py"))
qcl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qcl)


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


class TestUntrackedDocsSkipped(unittest.TestCase):
    """The gate the brief asks for: red before the fix (not exercised here — this pins the FIXED
    behavior), green with a note while the doc is untracked, red again the moment it is staged."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = self.tmp.name
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.email", "test@example.com")
        _git(self.repo, "config", "user.name", "Test")

        # A cited target: some .py file with real content, tracked from the start. The CITING doc
        # (docs/task-scratch.md, below) is what starts untracked -- that is the half under test.
        pkg = os.path.join(self.repo, "pkg.py")
        with open(pkg, "w") as f:
            f.write("x = 1\ny = 2\n")
        os.makedirs(os.path.join(self.repo, "docs"))
        _git(self.repo, "add", "pkg.py")
        _git(self.repo, "commit", "-q", "-m", "init")
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.repo, capture_output=True,
                              text=True, check=True).stdout.strip()

        # A minimal QUEUE.md: one resolving commit citation (main() refuses on zero SHA citations),
        # no generated index block yet (body_without_index/tail_after_index both handle that).
        queue = os.path.join(self.repo, "docs", "QUEUE.md")
        with open(queue, "w") as f:
            f.write(f"# QUEUE\n\ncommit {sha} — init\n")
        _git(self.repo, "add", "docs/QUEUE.md")
        _git(self.repo, "commit", "-q", "-m", "queue")

        # Monkeypatch the module's ROOT/QUEUE to this isolated repo, and reset its process-lifetime
        # tracked-files cache (it was already populated, if at all, against the real goinfer repo).
        self._orig_root, self._orig_queue = qcl.ROOT, qcl.QUEUE
        qcl.ROOT = pathlib.Path(self.repo)
        qcl.QUEUE = qcl.ROOT / "docs" / "QUEUE.md"
        qcl._tracked_cache = qcl._TRACKED_SENTINEL

        # Bootstrap the generated SHA index main() requires on a non---update run — a real
        # docs/QUEUE.md always already carries one; this test's own commit citation needs the same
        # one-time --update a freshly-filed queue entry would get before anyone runs the lint bare.
        code, out = self._run(["--update"])
        assert code == 0, out
        _git(self.repo, "add", "docs/QUEUE.md", "docs/citation-index.md")
        _git(self.repo, "commit", "-q", "-m", "index")

    def tearDown(self):
        qcl.ROOT, qcl.QUEUE = self._orig_root, self._orig_queue
        qcl._tracked_cache = qcl._TRACKED_SENTINEL
        self.tmp.cleanup()

    def _run(self, argv):
        """Invoke qcl.main() as the CLI would, capturing BOTH stdout and stderr as one stream —
        the note and most failure reports print to stdout, but some (e.g. unresolved path
        citations) go to stderr, and the CLI's own combined terminal output doesn't distinguish
        them either."""
        old_argv = sys.argv
        sys.argv = ["queue_citation_lint.py"] + list(argv)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                code = qcl.main()
        finally:
            sys.argv = old_argv
        return code, buf.getvalue()

    def test_untracked_doc_skipped_then_reds_once_tracked(self):
        scratch = os.path.join(self.repo, "docs", "task-scratch.md")
        # An UNRESOLVABLE citation: pkg.py exists, but no file goinfer would ever resolve exists
        # under this name -- the simplest form of "cannot resolve" (resolve_path returns nothing).
        with open(scratch, "w") as f:
            f.write("see `nonexistent_module_xyz.py:1` for details\n")

        # Step 1/2: untracked -> green, and the note names the file (not silent).
        code, out = self._run([])
        self.assertEqual(code, 0, out)
        self.assertIn("task-scratch.md", out)
        self.assertIn("note:", out)

        # Step 5 (first half): --update must NOT index the untracked file's citation.
        code, out = self._run(["--update"])
        self.assertEqual(code, 0, out)
        queue_text = qcl.index_path().read_text()
        self.assertNotIn("task-scratch.md", queue_text)

        # Step 3/4: git add (no commit) -> now tracked; must red, naming that citation.
        _git(self.repo, "add", "docs/task-scratch.md")
        qcl._tracked_cache = qcl._TRACKED_SENTINEL  # ls-files result changed; drop the cache
        code, out = self._run([])
        self.assertNotEqual(code, 0, out)
        self.assertIn("nonexistent_module_xyz.py", out)
        self.assertNotIn("note:", out)  # nothing left untracked to report

        # Step 5 (second half): once it resolves AND is tracked, --update DOES index it.
        with open(scratch, "w") as f:
            f.write("see `pkg.py:1` for details\n")
        qcl._tracked_cache = qcl._TRACKED_SENTINEL
        code, out = self._run(["--update"])
        self.assertEqual(code, 0, out)
        queue_text = qcl.index_path().read_text()
        self.assertIn("task-scratch.md", queue_text)

    def test_no_git_degrades_to_lint_everything_and_says_so(self):
        # Simulate "git ls-files fails" directly rather than deleting .git (which would also break
        # this test's own setUp/tearDown git calls) -- _tracked_files() is the single chokepoint.
        qcl._tracked_cache = None
        orig = qcl._tracked_files
        qcl._tracked_files = lambda: None
        try:
            scratch = os.path.join(self.repo, "docs", "task-scratch.md")
            with open(scratch, "w") as f:
                f.write("see `nonexistent_module_xyz.py:1` for details\n")
            code, out = self._run([])
            self.assertNotEqual(code, 0, out)  # linted despite being untracked -> reds
            self.assertIn("git ls-files unavailable", out)
        finally:
            qcl._tracked_files = orig
            qcl._tracked_cache = qcl._TRACKED_SENTINEL


class TestContentGoneRefused(unittest.TestCase):
    """N-51 (docs/audit-2026-09-10.md): --update already refuses to launder a SHIFTED citation
    (the old content found elsewhere in the file) but had no equivalent refusal for a
    CONTENT-ABSENT one (the old content found NOWHERE — deleted or rewritten, not moved). Before
    the fix, that case fell through the SHIFTED branch's `if at is not None` and silently re-keyed
    to whatever (if anything) now sits at the stale line. This pins the fix in isolation, same
    isolated-temp-repo discipline as TestUntrackedDocsSkipped (never touches this checkout's git
    state)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = self.tmp.name
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.email", "test@example.com")
        _git(self.repo, "config", "user.name", "Test")

        self.pkg = os.path.join(self.repo, "pkg.py")
        # A genuinely discriminating (>=12 chars, alnum) line 2 to cite and later delete outright.
        with open(self.pkg, "w") as f:
            f.write("x = 1\ndef a_real_distinctive_function_name():\n    pass\n")
        os.makedirs(os.path.join(self.repo, "docs"))
        self.scratch = os.path.join(self.repo, "docs", "task-scratch.md")
        with open(self.scratch, "w") as f:
            f.write("see `pkg.py:2` for details\n")

        queue = os.path.join(self.repo, "docs", "QUEUE.md")
        sha_placeholder_commit = _git_commit_all(self.repo, "init")
        with open(queue, "w") as f:
            f.write(f"# QUEUE\n\ncommit {sha_placeholder_commit} — init\n")
        _git(self.repo, "add", "docs/QUEUE.md", "docs/task-scratch.md")
        _git(self.repo, "commit", "-q", "-m", "queue+scratch")

        self._orig_root, self._orig_queue = qcl.ROOT, qcl.QUEUE
        qcl.ROOT = pathlib.Path(self.repo)
        qcl.QUEUE = qcl.ROOT / "docs" / "QUEUE.md"
        qcl._tracked_cache = qcl._TRACKED_SENTINEL

        # Bootstrap the index: pkg.py:2 gets keyed to its real, current, discriminating content.
        code, out = self._run(["--update"])
        assert code == 0, out
        self.assertIn("a_real_distinctive_function_name", qcl.index_path().read_text())
        _git(self.repo, "add", "docs/QUEUE.md", "docs/citation-index.md")
        _git(self.repo, "commit", "-q", "-m", "index")

    def tearDown(self):
        qcl.ROOT, qcl.QUEUE = self._orig_root, self._orig_queue
        qcl._tracked_cache = qcl._TRACKED_SENTINEL
        self.tmp.cleanup()

    def _run(self, argv):
        old_argv = sys.argv
        sys.argv = ["queue_citation_lint.py"] + list(argv)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                code = qcl.main()
        finally:
            sys.argv = old_argv
        return code, buf.getvalue()

    def test_update_refuses_when_old_content_is_gone_not_shifted(self):
        # Rewrite pkg.py so the cited line's exact text does not appear ANYWHERE in the file —
        # not moved to a different line (that's the SHIFTED case, already covered), genuinely
        # replaced. A different discriminating string at the SAME line number, so a naive
        # "line count didn't change" check couldn't mistake this for nothing having happened.
        with open(self.pkg, "w") as f:
            f.write("x = 1\ndef a_totally_different_unrelated_name():\n    pass\n")

        code, out = self._run(["--update"])
        self.assertNotEqual(code, 0, out)
        self.assertIn("CONTENT GONE", out)
        self.assertIn("pkg.py:2", out)
        # The OLD text should be named (what's missing), not silently dropped.
        self.assertIn("a_real_distinctive_function_name", out)

        # Confirm --update genuinely did NOT rewrite the index out from under the refusal: the
        # stale content-key must still be what's on disk, not the new function's name.
        queue_text = qcl.index_path().read_text()
        self.assertIn("a_real_distinctive_function_name", queue_text)
        self.assertNotIn("a_totally_different_unrelated_name", queue_text)


    # 2026-09-24, owner decision: a cited line that MOVED but did not CHANGE is accepted.
    def test_moved_unchanged_line_is_accepted_and_update_repoints_the_prose(self):
        # Two lines inserted above the cited one: same text, now at line 4 instead of 2. The doc
        # also cites it as a range, whose end must move with it.
        with open(self.scratch, "w") as f:
            f.write("see `pkg.py:2` for details, and `pkg.py:2-3` for the body\n")
        _git(self.repo, "add", "docs/task-scratch.md")
        _git(self.repo, "commit", "-q", "-m", "range")
        code, out = self._run(["--update"])
        self.assertEqual(code, 0, out)
        with open(self.pkg, "w") as f:
            f.write("x = 1\ndef an_inserted_helper_function_one():\ndef an_inserted_helper_function_two():\n"
                    "def a_real_distinctive_function_name():\n    pass\n")

        code, out = self._run([])
        self.assertEqual(code, 0, out)  # accepted in check mode…
        self.assertIn("MOVED but are unchanged", out)  # …and reported, not silent
        self.assertIn("-> :4", out)

        code, out = self._run(["--update"])
        self.assertEqual(code, 0, out)
        doc = pathlib.Path(self.scratch).read_text()
        self.assertIn("`pkg.py:4`", doc)
        self.assertIn("`pkg.py:4-5`", doc)  # the range keeps its length
        self.assertNotIn("pkg.py:2", doc)
        self.assertIn("pkg.py:4", qcl.index_path().read_text())
        code, out = self._run([])
        self.assertEqual(code, 0, out)
        self.assertNotIn("MOVED but are unchanged", out)

    def test_moved_line_that_now_appears_twice_is_red(self):
        with open(self.pkg, "w") as f:
            f.write("x = 1\ndef an_inserted_helper_function_one():\ndef a_real_distinctive_function_name():\n"
                    "def a_real_distinctive_function_name():\n    pass\n")
        code, out = self._run([])
        self.assertNotEqual(code, 0, out)
        self.assertIn("AMBIGUOUS", out)
        code, out = self._run(["--update"])
        self.assertNotEqual(code, 0, out)
        self.assertIn("AMBIGUOUS", out)
        self.assertIn("pkg.py:2", pathlib.Path(self.scratch).read_text())  # prose untouched on a guess


def _git_commit_all(repo, msg):
    """Commit whatever is currently staged/tracked and return the resulting short SHA — used
    where a test needs a real, resolving commit to cite before the file it commits exists yet."""
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg, "--allow-empty")
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                           capture_output=True, text=True, check=True).stdout.strip()


if __name__ == "__main__":
    unittest.main()
