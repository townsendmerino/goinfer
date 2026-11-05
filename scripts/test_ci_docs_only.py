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

    def run_script(self, base, head, *args):
        return subprocess.run(["bash", os.path.join(ROOT, "scripts", "ci_docs_only.sh"), *args, base, head], cwd=self.wt,
                              capture_output=True, text=True).stdout.strip()

    def classify(self, edits, subs=None, both=False):
        """Commit edits (path -> text appended) and subs (path -> (old, new) replaced once) on top of the worktree's HEAD
        and classify that one commit. both: return (verdict, class) instead of the verdict."""
        base = git(self.wt, "rev-parse", "HEAD")
        for path, text in edits.items():
            full = os.path.join(self.wt, path)
            os.makedirs(os.path.dirname(full) or self.wt, exist_ok=True)
            with open(full, "a") as f:
                f.write(text)
            git(self.wt, "add", path)
        for path, (old, new) in (subs or {}).items():
            full = os.path.join(self.wt, path)
            with open(full) as f:
                text = f.read()
            assert old in text, (path, old)
            with open(full, "w") as f:
                f.write(text.replace(old, new, 1))
            git(self.wt, "add", path)
        git(self.wt, *IDENTITY, "commit", "-q", "-m", "test")
        head = git(self.wt, "rev-parse", "HEAD")
        out = self.run_script(base, head)
        cls = self.run_script(base, head, "--classify") if both else None
        git(self.wt, "reset", "-q", "--hard", base)
        return (out, cls) if both else out

    def test_a_docs_readme_and_run_py_are_docs_only(self):
        self.assertEqual(self.classify({"docs/measurements/zz-ci-docs-only-test/README.md": "# a record\n",
                                        "docs/measurements/zz-ci-docs-only-test/run.py": "print(1)\n"}), "true")

    def test_the_root_readme_is_code(self):
        self.assertEqual(self.classify({"README.md": "\n"}), "false")

    def test_a_docs_file_a_test_reads_is_code(self):
        self.assertEqual(self.classify({"docs/env-vars.md": "\n"}), "false")

    # -- the CI quick path's classes (--classify): what runs `lint` alone, and what still runs the full suite.
    # The default mode (the direct-push rule, coverage.yml) does not widen: the last test pins that.
    def cls(self, edits, subs=None):
        return self.classify(edits, subs, both=True)[1]

    def test_records_are_quick(self):
        self.assertEqual(self.cls({"docs/measurements/zz-ci-test.md": "# r\n"}), "records")

    def test_docs_changelog_fragments_and_other_top_level_markdown_are_quick(self):
        self.assertEqual(self.cls({"docs/zz-ci-test.md": "x\n"}), "docs")
        self.assertEqual(self.cls({"changelog.d/zz-ci-test.md": "x\n"}), "docs")
        self.assertEqual(self.cls({"THIRD_PARTY_LICENSES.md": "\n"}), "docs")

    def test_site_content_is_quick_and_site_code_is_not(self):
        self.assertEqual(self.cls({"apps/site/different/zz-ci-test.md": "x\n"}), "docs")
        self.assertEqual(self.cls({"apps/site/internal/site/templates/models.html": "\n"}), "docs")
        self.assertEqual(self.cls({"apps/site/internal/site/render.go": "//x\n"}), "full")
        self.assertEqual(self.cls({"apps/site/go.mod": "\n"}), "full")

    def test_a_file_a_test_reads_is_full_whatever_its_class(self):
        for path in ("README.md", "CHANGELOG.md", "CLAUDE.md", "apps/site/data/claims.json", "docs/env-vars.md"):
            self.assertEqual(self.cls({path: "\n"}), "full", path)

    def test_code_scripts_and_workflows_are_full(self):
        for path in ("decoder/doc.go", "scripts/readme_smoke.sh", ".github/workflows/ci.yml", "testdata/env_reads.txt"):
            self.assertEqual(self.cls({path: "\n"}), "full", path)

    def test_a_mix_with_one_code_file_is_full(self):
        self.assertEqual(self.cls({"docs/zz-ci-test.md": "x\n", "decoder/doc.go": "//x\n"}), "full")

    def test_a_mix_of_quick_classes_is_quick(self):
        self.assertEqual(self.cls({"docs/zz-ci-test.md": "x\n", "docs/measurements/zz-ci-test.md": "x\n"}), "quick")

    def test_a_goinfer_version_bump_is_version_only(self):
        bump = {"backend/metal/go.mod": ("github.com/townsendmerino/goinfer v0.24.1", "github.com/townsendmerino/goinfer v0.24.2")}
        self.assertEqual(self.cls({}, bump), "version")

    def test_other_changes_to_a_go_mod_are_full(self):
        aikit = {"backend/metal/go.mod": ("github.com/townsendmerino/aikit v", "github.com/townsendmerino/aikit v9.")}
        self.assertEqual(self.cls({}, aikit), "full")
        self.assertEqual(self.cls({"backend/metal/go.mod": "require example.com/other v1.0.0\n"}), "full")
        # a goinfer bump riding with a different dependency's bump is not version-only either
        mixed = {"backend/metal/go.mod": ("github.com/townsendmerino/goinfer v0.24.1", "github.com/townsendmerino/goinfer v0.24.2"),
                 "go.mod": ("golang.org/x/sys v", "golang.org/x/sys v9.")}
        self.assertEqual(self.cls({}, mixed), "full")

    def test_a_goinfer_lookalike_is_not_a_goinfer_version(self):
        self.assertEqual(self.cls({"backend/metal/go.mod": "require github.com/townsendmerino/goinfer-evil v1.0.0\n"}), "full")

    def test_a_go_sum_line_for_a_goinfer_module_is_version_only(self):
        line = "github.com/townsendmerino/goinfer v0.99.0 h1:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=\n"
        self.assertEqual(self.cls({"backend/metal/go.sum": line}), "version")
        other = "example.com/other v0.99.0 h1:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=\n"
        self.assertEqual(self.cls({"backend/metal/go.sum": other}), "full")

    def test_the_default_mode_does_not_widen(self):
        for edits in ({"changelog.d/zz-ci-test.md": "x\n"}, {"SECURITY.md": "x\n"}, {"apps/site/different/zz-ci-test.md": "x\n"}):
            self.assertEqual(self.classify(edits), "false", edits)
        bump = {"backend/metal/go.mod": ("github.com/townsendmerino/goinfer v0.24.1", "github.com/townsendmerino/goinfer v0.24.2")}
        self.assertEqual(self.classify({}, bump), "false")
        self.assertEqual(self.classify({"docs/zz-ci-test.md": "x\n"}), "true")


if __name__ == "__main__":
    unittest.main()
