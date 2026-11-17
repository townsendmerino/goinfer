#!/usr/bin/env python3
"""doc_id_collisions.py and doc_review_staleness.py read the queue and task docs from the private notes checkout when it has them,
and from docs/ otherwise. Run: python3 -B scripts/test_notes_readers.py"""
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import doc_id_collisions as dic  # noqa: E402
import doc_review_staleness as drs  # noqa: E402

DUP = "- **P1 · first**\n- **P1 · second**\n"


def git(d, *a):
    subprocess.run(["git", "-C", str(d), "-c", "user.email=t@example.com", "-c", "user.name=T", *a], check=True, capture_output=True)


class _Scratch(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        base = pathlib.Path(self._td.name)
        self.repo, self.notes = base / "goinfer", base / "goinfer-notes"
        for d in (self.repo / "docs", self.notes):
            d.mkdir(parents=True)
            git(d, "init", "-q")
        self._saved = (dic.ROOT, drs.ROOT, sys.argv)
        dic.ROOT = drs.ROOT = self.repo
        sys.argv = ["x"]

    def tearDown(self):
        dic.ROOT, drs.ROOT, sys.argv = self._saved
        self._td.cleanup()

    def write(self, base, rel, text):
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def commit(self, base):
        git(base, "add", ".")
        git(base, "commit", "-q", "-m", "c")


class QueueDocs(_Scratch):
    def test_docs_layout_without_a_checkout(self):
        self.write(self.repo, "docs/queue-a.md", DUP)
        self.write(self.repo, "docs/QUEUE.md", "# q\n")
        self.assertEqual([p.name for p in dic.scoped_files()], ["queue-a.md", "QUEUE.md"])
        self.assertEqual(dic.queue_dir(), self.repo / "docs")

    def test_a_sibling_checkout_with_queues_wins(self):
        self.write(self.repo, "docs/queue-a.md", DUP)
        self.write(self.notes, "queues/queue-b.md", DUP)
        self.assertEqual([p.name for p in dic.scoped_files()], ["queue-b.md"])

    def test_a_checkout_without_queues_keeps_docs(self):
        self.write(self.repo, "docs/queue-a.md", DUP)
        self.write(self.notes, "plans/plan-x.md", "x\n")
        self.assertEqual([p.name for p in dic.scoped_files()], ["queue-a.md"])

    def test_explicit_notes_path(self):
        other = self.notes.parent / "elsewhere"
        self.write(other, "queues/queue-c.md", DUP)
        sys.argv = ["x", "--notes", str(other)]
        self.assertEqual([p.name for p in dic.scoped_files()], ["queue-c.md"])

    def test_collisions_are_found_in_the_notes(self):
        self.write(self.notes, "queues/queue-b.md", DUP)
        self.assertEqual(list(dic.find_collisions(self.notes / "queues/queue-b.md")), ["P1"])


class TaskDocs(_Scratch):
    def test_the_docs_come_from_the_checkout_and_their_dates_from_its_git(self):
        self.write(self.repo, "docs/tasks/task-old.md", "# old\n")
        self.write(self.notes, "tasks/task-a.md", "# a\n<!-- doc-reviewed: 2000-01-01 -->\n")
        self.write(self.notes, "tasks/parked/task-b.md", "# b\n")
        self.commit(self.notes)
        self.assertEqual(drs.notes_root(), self.notes.resolve())
        self.assertEqual(sorted(p.name for p in drs.review_scoped_docs()), ["task-a.md", "task-b.md"])
        self.assertIsNotNone(drs.last_commit_date("tasks/task-a.md", drs.task_base()))
        self.assertIsNone(drs.last_commit_date("tasks/task-a.md"))  # goinfer's own git has never seen it

    def test_without_tasks_in_the_checkout_docs_tasks_is_read(self):
        self.write(self.repo, "docs/tasks/task-old.md", "# old\n")
        self.write(self.notes, "plans/plan-x.md", "x\n")
        self.assertIsNone(drs.notes_root())
        self.assertEqual([p.name for p in drs.review_scoped_docs()], ["task-old.md"])

    def test_the_index_beside_the_queue_in_the_checkout_is_read(self):
        row = "| `notes:tasks/task-a.md|pkg/x.go:3` | goinfer | `func f()` |"
        self.write(self.notes, "tasks/task-a.md", "# a\n")
        self.write(self.notes, "queues/citation-index.md", f"{drs.MARK_BEGIN}\n{row}\n")
        self.assertEqual(drs.parse_citation_index(), {"notes:tasks/task-a.md": [("pkg/x.go", "goinfer")]})


if __name__ == "__main__":
    unittest.main()
