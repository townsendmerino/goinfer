#!/usr/bin/env python3
"""Tests for scripts/changelog.py: fragments render newest first and verbatim, and a release writes the section the
manual move used to and removes the fragments."""
import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import changelog  # noqa: E402

HEAD = "# Changelog\n\nintro\n\n## [Unreleased]\n\npointer\n\n## [v0.1.0] — 2026-01-01\n\n- old\n"


def repo(frags):
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "changelog.d").mkdir()
    (d / "changelog.d" / "README.md").write_text("how\n")
    for name, body in frags.items():
        (d / "changelog.d" / name).write_text(body)
    (d / "CHANGELOG.md").write_text(HEAD)
    g = ["git", "-c", "user.email=t@example", "-c", "user.name=t"]
    subprocess.run(g[:1] + ["init", "-q"], cwd=d, check=True)
    subprocess.run(g + ["add", "."], cwd=d, check=True)
    subprocess.run(g + ["commit", "-q", "-m", "x"], cwd=d, check=True)
    return d


class Render(unittest.TestCase):
    def test_newest_first_and_verbatim(self):
        d = repo({"2026-10-20-01-a.md": "- a\n", "2026-10-27-b.md": "- b\n  more\n\n", "2026-10-21-c.md": "- c\n"})
        self.assertEqual(changelog.render(d), "\n- b\n  more\n\n- c\n- a\n\n")

    def test_none_is_empty(self):
        self.assertEqual(changelog.render(repo({})), "")

    def test_a_misnamed_file_is_refused(self):
        with self.assertRaises(SystemExit):
            changelog.render(repo({"notes.md": "- x\n"}))

    def test_a_fragment_must_be_one_entry(self):
        with self.assertRaises(SystemExit):
            changelog.render(repo({"2026-10-27-x.md": "not a bullet\n"}))


class Release(unittest.TestCase):
    def test_writes_the_section_and_removes_the_fragments(self):
        d = repo({"2026-10-27-b.md": "- b\n", "2026-10-26-a.md": "- a\n"})
        changelog.release(d, "v0.2.0", "2026-10-27")
        self.assertEqual((d / "CHANGELOG.md").read_text(),
                         HEAD.replace("## [v0.1.0]", "## [v0.2.0] — 2026-10-27\n\n- b\n- a\n\n## [v0.1.0]"))
        self.assertEqual(sorted(p.name for p in (d / "changelog.d").iterdir()), ["README.md"])

    def test_refuses_an_existing_version_and_an_empty_release(self):
        with self.assertRaises(SystemExit):
            changelog.release(repo({"2026-10-27-b.md": "- b\n"}), "v0.1.0", "2026-10-27")
        with self.assertRaises(SystemExit):
            changelog.release(repo({}), "v0.2.0", "2026-10-27")


if __name__ == "__main__":
    unittest.main()
