#!/usr/bin/env python3
"""Tests for scripts/task_docs.py unstub. Run: python3 -B scripts/test_task_docs.py"""
import io
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import task_docs  # noqa: E402


class UnstubTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._td.name)
        self._orig_root = task_docs.ROOT
        task_docs.ROOT = self.root
        task_docs._read_cache.clear()
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=self.root, check=True)

        self._write(
            "docs/completed/task-alpha.md",
            "# Alpha\n\n> **ARCHIVED — a record, not instructions.**\n",
        )
        self._write(
            "docs/tasks/task-alpha.md",
            "# Alpha\n\n**Moved.** Archived 2026-10-01; the doc now lives at\n"
            "[`completed/task-alpha.md`](../completed/task-alpha.md).\n",
        )

    def tearDown(self):
        task_docs.ROOT = self._orig_root
        task_docs._read_cache.clear()
        self._td.cleanup()

    def _write(self, rel: str, content: str) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def _commit_all(self) -> None:
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=self.root, check=True)

    def _run_unstub(self, *argv: str) -> tuple[int, str]:
        buf = io.StringIO()
        saved = sys.argv
        try:
            sys.argv = ["task_docs.py", "unstub", *argv]
            with redirect_stdout(buf):
                rc = task_docs.main()
        finally:
            sys.argv = saved
        return rc, buf.getvalue()

    def test_repo_root_path_in_go_comment(self):
        self._write("pkg/foo.go", "package pkg\n// See docs/tasks/task-alpha.md for details.\n")
        self._commit_all()
        rc, _ = self._run_unstub("--apply")
        self.assertEqual(rc, 0)
        self.assertEqual(
            (self.root / "pkg/foo.go").read_text(encoding="utf-8"),
            "package pkg\n// See docs/completed/task-alpha.md for details.\n",
        )

    def test_relative_link_from_docs_md(self):
        self._write("docs/guide.md", "Read [alpha](tasks/task-alpha.md).\n")
        self._commit_all()
        self._run_unstub("--apply")
        self.assertEqual(
            (self.root / "docs/guide.md").read_text(encoding="utf-8"),
            "Read [alpha](completed/task-alpha.md).\n",
        )

    def test_same_directory_link_from_docs_tasks_md(self):
        self._write("docs/tasks/task-beta.md", "Follow-up to [alpha](task-alpha.md).\n")
        self._commit_all()
        self._run_unstub("--apply")
        self.assertEqual(
            (self.root / "docs/tasks/task-beta.md").read_text(encoding="utf-8"),
            "Follow-up to [alpha](../completed/task-alpha.md).\n",
        )

    def test_parent_relative_path_read_by_go_test(self):
        self._write("pkg/doc_test.go", 'package pkg\nvar p = "../docs/tasks/task-alpha.md"\n')
        self._commit_all()
        self._run_unstub("--apply")
        self.assertEqual(
            (self.root / "pkg/doc_test.go").read_text(encoding="utf-8"),
            'package pkg\nvar p = "../docs/completed/task-alpha.md"\n',
        )

    def test_link_with_fragment(self):
        self._write("docs/guide.md", "See [sec](tasks/task-alpha.md#phase-2) and docs/tasks/task-alpha.md#s1.\n")
        self._commit_all()
        self._run_unstub("--apply")
        self.assertEqual(
            (self.root / "docs/guide.md").read_text(encoding="utf-8"),
            "See [sec](completed/task-alpha.md#phase-2) and docs/completed/task-alpha.md#s1.\n",
        )

    def test_github_blob_url(self):
        self._write("README.md", "https://github.com/townsendmerino/goinfer/blob/main/docs/tasks/task-alpha.md\n")
        self._commit_all()
        self._run_unstub("--apply")
        self.assertEqual(
            (self.root / "README.md").read_text(encoding="utf-8"),
            "https://github.com/townsendmerino/goinfer/blob/main/docs/completed/task-alpha.md\n",
        )

    def test_longer_filename_and_bare_filename_must_not_change(self):
        orig = (
            "See docs/tasks/task-alpha-extra.md and [x](tasks/task-alpha-extra.md)\n"
            "and bare task-alpha.md in prose.\n"
        )
        self._write("docs/guide.md", orig)
        self._commit_all()
        self._run_unstub("--apply")
        self.assertEqual((self.root / "docs/guide.md").read_text(encoding="utf-8"), orig)

    def test_frozen_file_must_not_change(self):
        orig = "Historical note: docs/tasks/task-alpha.md.\n"
        self._write("docs/completed/old.md", orig)
        self._write("docs/measurements/m1.md", orig)
        self._commit_all()
        _, out = self._run_unstub("--apply")
        self.assertEqual((self.root / "docs/completed/old.md").read_text(encoding="utf-8"), orig)
        self.assertEqual((self.root / "docs/measurements/m1.md").read_text(encoding="utf-8"), orig)
        self.assertIn("FROZEN (never edit):        2 file(s)", out)

    def test_generated_page_must_not_change(self):
        orig = "<!-- Generated by `go test ./decoder -run Foo -update`; do not edit. -->\n[a](tasks/task-alpha.md)\n"
        self._write("docs/gen.md", orig)
        self._commit_all()
        _, out = self._run_unstub("--apply")
        self.assertEqual((self.root / "docs/gen.md").read_text(encoding="utf-8"), orig)
        self.assertIn("go test ./decoder -run Foo -update", out)

    def test_only_and_delete_stubs_and_stage(self):
        self._write("docs/guide.md", "See docs/tasks/task-alpha.md.\n")
        self._write("pkg/foo.go", "package pkg\n// docs/tasks/task-alpha.md\n")
        self._commit_all()

        self._run_unstub("--only", "docs/*", "--delete-stubs", "--apply", "--stage")
        self.assertTrue((self.root / "docs/tasks/task-alpha.md").exists())
        self.assertEqual((self.root / "docs/guide.md").read_text(encoding="utf-8"), "See docs/completed/task-alpha.md.\n")
        self.assertEqual((self.root / "pkg/foo.go").read_text(encoding="utf-8"), "package pkg\n// docs/tasks/task-alpha.md\n")

        self._run_unstub("--delete-stubs", "--apply", "--stage")
        self.assertFalse((self.root / "docs/tasks/task-alpha.md").exists())
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only"], cwd=self.root, capture_output=True, text=True, check=True
        ).stdout.splitlines()
        self.assertEqual(sorted(staged), ["docs/guide.md", "docs/tasks/task-alpha.md", "pkg/foo.go"])

    def test_generated_by_requires_do_not_edit(self):
        self._write(
            "docs/prose.md",
            "# Prose\n\nTables generated from the registry are the truth.\n",
        )
        self._write(
            "docs/gen_with_cmd.md",
            "# Matrix\n\nGenerated by `go test ./decoder -run Foo -update`; do not edit.\n",
        )
        self.assertIsNone(task_docs.generated_by("docs/prose.md"))
        self.assertEqual(
            task_docs.generated_by("docs/gen_with_cmd.md"),
            "go test ./decoder -run Foo -update",
        )


if __name__ == "__main__":
    unittest.main()
