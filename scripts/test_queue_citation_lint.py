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
        # These pin the LEGACY line-index machinery (content keys, "moved but unchanged", launder refusal), which only runs with the door
        # OPEN; production has had it shut since CC0 step 4. They stay until that machinery is deleted.
        self._orig_door = qcl.CLOSE_THE_DOOR
        qcl.CLOSE_THE_DOOR = False
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
        qcl.CLOSE_THE_DOOR = self._orig_door
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
        # These pin the LEGACY line-index machinery (content keys, "moved but unchanged", launder refusal), which only runs with the door
        # OPEN; production has had it shut since CC0 step 4. They stay until that machinery is deleted.
        self._orig_door = qcl.CLOSE_THE_DOOR
        qcl.CLOSE_THE_DOOR = False
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
        qcl.CLOSE_THE_DOOR = self._orig_door
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
    # The FULL hash, never --short. The lint only recognises a hash that holds both a digit and a letter (looks_like_sha), so a
    # 7-character one is all digits about 4% of the time, and then QUEUE.md has "ZERO SHA citations" and setUp fails. Commits made in
    # the same second from the same tree get the same hash, so the tests of one class failed together: 7 of 40 runs red, clumps of
    # 3, 4 and 22 tests, found 2026-10-10 while adding CC0's tests. A 40-character hash is all digits with probability ~6e-9.
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                           capture_output=True, text=True, check=True).stdout.strip()



class TestHookGitDirDoesNotRedirectSiblingLookups(unittest.TestCase):
    """A git hook exports GIT_DIR for the repository being pushed (absolute, from a worktree). subject_of names the repo
       it searches by cwd, so an inherited GIT_DIR made it search the PUSHING repo for another repo's SHA and report it
       as not resolving: the pre-push hook refused a push from a goinfer worktree on two aikit SHAs (2026-10-07)."""

    def test_sha_in_sibling_resolves_under_a_foreign_git_dir(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = os.path.join(td, "a"), os.path.join(td, "b")
            for r in (a, b):
                os.makedirs(r)
                _git(r, "init", "-q")
                _git(r, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "in " + os.path.basename(r))
            sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=b, capture_output=True, text=True,
                                 check=True).stdout.strip()
            orig = qcl.sibling_repos
            saved = os.environ.get("GIT_DIR")
            qcl.sibling_repos = lambda: [b]
            os.environ["GIT_DIR"] = os.path.join(a, ".git")
            try:
                got = qcl.subject_of(sha)
            finally:
                qcl.sibling_repos = orig
                if saved is None:
                    os.environ.pop("GIT_DIR", None)
                else:
                    os.environ["GIT_DIR"] = saved
            self.assertTrue(isinstance(got, str) and got.endswith("in b"), f"subject_of = {got!r}, want the sibling's subject")


# ---------------------------------------------------------------------------------------------------
# CC0 (docs/tasks/task-code-comments-2026-10.md): symbol citations `path.go:Name`, pinned records
# `<!-- citations-at: <commit> -->`, and the closing of the path:line door. Mutation-checked: each
# red case below is a mistake someone will make, and each green case is an edit that used to cost a
# citation re-point and must not any more.
# ---------------------------------------------------------------------------------------------------

GO_FIXTURE = """package svc

import (
	"fmt"
)

// Server serves.
type Server struct {
	addr string
	Port int
}

type Stack[T any] struct{ items []T }

type Pair[K comparable, V any] struct {
	k K
	v V
}

type Alias = Server

type (
	Reader interface{ Read() int }
	Writer interface {
		Write(p []byte) int
	}
)

const Version = "1"

const (
	ModeA = iota
	ModeB
	// ModeC has a doc comment inside the block.
	ModeC
	First, Second = 1, 2
)

var global = 1

var Left, Right = 3, 4

var (
	tableA = []int{
		1, 2,
	}
	tableB = func() int {
		return 2
	}
	_ = fmt.Sprint
)

// NewServer builds one.
func NewServer(addr string) *Server { return &Server{addr: addr} }

func (s *Server) Handle(n int) int {
	var local = 5
	const localConst = 6
	type localType int
	return n + local + localConst
}

func (s Server) Addr() string { return s.addr }

func (*Server) NoName() {}

func (Server) NoNameValue() {}

func (s *Stack[T]) Push(v T) { s.items = append(s.items, v) }

func (p *Pair[K, V]) Swap() (V, K) { return p.v, p.k }

func Map[T, U any](in []T, f func(T) U) []U { return nil }

func init() {}

const raw = `
func FakeInRaw() {}
const FakeConstInRaw = 1
type FakeTypeInRaw int
`

/*
func FakeInBlockComment() {}
*/

// func FakeInLineComment() {}

func Last() {}
"""

PY_FIXTURE = """class Tool:
    def run(self):
        return 1

    async def arun(self):
        return 2


def helper():
    return 3


async def ahelper():
    return 4
"""


class TestGoDeclarations(unittest.TestCase):
    """The declaration forms a `path.go:Name` citation can name, and the look-alikes that must not count."""

    def setUp(self):
        self.names = qcl.go_declarations(GO_FIXTURE)

    def test_every_declaration_form_is_found(self):
        want = {
            # types: plain, generic, alias, grouped (one-line and multi-line specs)
            "Server", "Stack", "Pair", "Alias", "Reader", "Writer",
            # consts and vars: single, grouped, iota continuation, multi-name specs, func-valued
            "Version", "ModeA", "ModeB", "ModeC", "First", "Second",
            "global", "Left", "Right", "tableA", "tableB", "raw",
            # funcs: plain, generic, init
            "NewServer", "Map", "init", "Last",
            # methods: pointer, value, unnamed pointer/value receivers, generic receivers (one and two parameters)
            "Server.Handle", "Server.Addr", "Server.NoName", "Server.NoNameValue", "Stack.Push", "Pair.Swap",
        }
        missing = want - self.names
        self.assertFalse(missing, f"declarations not found: {sorted(missing)}")

    def test_look_alikes_are_not_declarations(self):
        for fake in ("FakeInRaw", "FakeConstInRaw", "FakeTypeInRaw", "FakeInBlockComment", "FakeInLineComment",
                     "local", "localConst", "localType", "addr", "Port", "Read", "Write", "_", "Handle", "Push"):
            self.assertNotIn(fake, self.names, f"{fake} must not count as a declaration")

    def test_python_def_and_class(self):
        names = qcl.py_declarations(PY_FIXTURE)
        for want in ("Tool", "Tool.run", "Tool.arun", "helper", "ahelper"):
            self.assertIn(want, names)
        self.assertNotIn("run", names)  # a method is cited as Class.method

    def test_symbol_regex_does_not_collide_with_path_or_bare_citations(self):
        t = "see `decoder/x.go:42` and `decoder/x.go:Model.run` and `decoder/x.go` and `a/b.py:helper`"
        self.assertEqual([m.group(2) for m in qcl.PATH_RE.finditer(t)], ["42"])
        self.assertEqual([(m.group(1), m.group(2)) for m in qcl.SYMBOL_RE.finditer(t)],
                         [("decoder/x.go", "Model.run"), ("a/b.py", "helper")])
        # a bare reference is one NOT followed by a colon, so neither citation form is also a bare one
        self.assertEqual([m.group(1) for m in qcl.BARE_RE.finditer(t)], ["decoder/x.go"])

    def test_enclosing_declaration_for_migration(self):
        lines = GO_FIXTURE.split("\n")
        at = lambda needle: next(i for i, l in enumerate(lines, 1) if needle in l)
        enc = lambda line: (lambda d: d.name if d else None)(qcl.enclosing_declaration(lines, line))
        self.assertEqual(enc(at("return n + local")), "Server.Handle")
        self.assertEqual(enc(at("func (s *Stack[T]) Push")), "Stack.Push")
        self.assertEqual(enc(at("ModeB")), "ModeB")
        self.assertEqual(enc(at("tableB = func")), "tableB")
        self.assertEqual(enc(at("return 2")), "tableB")                   # a line inside a func literal in a var spec
        self.assertEqual(enc(at("// ModeC has a doc comment")), "ModeC")  # a doc comment inside a block belongs to its spec
        self.assertEqual(enc(at("// NewServer builds one.")), "NewServer")  # a doc comment belongs to the decl below it
        self.assertIsNone(enc(at('"fmt"')))                                # the import block is outside any declaration
        self.assertIsNone(enc(1))                                          # the package clause too


class _LintRepo(unittest.TestCase):
    """An isolated git repo with a cited Go file, a Python file, a QUEUE.md and a bootstrapped index, the lint pointed at it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = self.tmp.name
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.email", "test@example.com")
        _git(self.repo, "config", "user.name", "Test")
        os.makedirs(os.path.join(self.repo, "pkg"))
        os.makedirs(os.path.join(self.repo, "docs"))
        self.write("pkg/svc.go", GO_FIXTURE)
        self.write("pkg/tool.py", PY_FIXTURE)
        self.base = _git_commit_all(self.repo, "init")  # a real commit: the pin target and QUEUE.md's required SHA citation
        self.write("docs/QUEUE.md", f"# QUEUE\n\ncommit {self.base} — init\n")
        _git(self.repo, "add", "docs/QUEUE.md")
        _git(self.repo, "commit", "-q", "-m", "queue")
        self._orig = (qcl.ROOT, qcl.QUEUE, getattr(qcl, "CLOSE_THE_DOOR", False), qcl._MODCACHE)
        qcl.ROOT = pathlib.Path(self.repo)
        qcl.QUEUE = qcl.ROOT / "docs" / "QUEUE.md"
        qcl._tracked_cache = qcl._TRACKED_SENTINEL
        qcl._MODCACHE = pathlib.Path(self.repo) / "no-module-cache-here"   # never the developer's real cache
        qcl.PINNED_FILE_CACHE.clear()
        code, out = self.run_lint(["--update"])
        assert code == 0, out
        _git(self.repo, "add", "docs/QUEUE.md", "docs/citation-index.md")
        _git(self.repo, "commit", "-q", "-m", "index")

    def tearDown(self):
        qcl.ROOT, qcl.QUEUE, qcl.CLOSE_THE_DOOR, qcl._MODCACHE = self._orig
        qcl._tracked_cache = qcl._TRACKED_SENTINEL
        qcl.PINNED_FILE_CACHE.clear()
        self.tmp.cleanup()

    def write(self, rel, text):
        p = os.path.join(self.repo, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(text)

    def read(self, rel):
        with open(os.path.join(self.repo, rel)) as f:
            return f.read()

    def doc(self, text, name="task-x.md", track=True):
        """Write and (by default) track a live doc, so the lint sees it."""
        self.write(f"docs/{name}", text)
        if track:
            _git(self.repo, "add", f"docs/{name}")
        qcl._tracked_cache = qcl._TRACKED_SENTINEL

    def run_lint(self, argv=()):
        old = sys.argv
        sys.argv = ["queue_citation_lint.py"] + list(argv)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                code = qcl.main()
        finally:
            sys.argv = old
        return code, buf.getvalue()

    def assertGreen(self, argv=()):
        code, out = self.run_lint(argv)
        self.assertEqual(code, 0, out)
        return out

    def assertRed(self, *needles, argv=()):
        code, out = self.run_lint(argv)
        self.assertNotEqual(code, 0, out)
        for n in needles:
            self.assertIn(n, out)
        return out


class TestSymbolCitations(_LintRepo):
    """CC0.b: the file must declare the name. Survives every edit that does not rename or move it ACROSS FILES."""

    def test_declared_names_are_green_in_every_form(self):
        self.doc("cites `pkg/svc.go:Server.Handle`, `pkg/svc.go:Stack.Push`, `pkg/svc.go:Pair.Swap`, "
                 "`pkg/svc.go:ModeB`, `pkg/svc.go:Right`, `pkg/svc.go:Reader`, `pkg/svc.go:Map`, "
                 "`pkg/tool.py:Tool.run`, and `pkg/tool.py:helper`.\n")
        self.assertGreen()

    def test_renamed_func_is_red(self):
        self.doc("see `pkg/svc.go:NewServer`\n")
        self.assertGreen()
        self.write("pkg/svc.go", GO_FIXTURE.replace("func NewServer(", "func BuildServer("))
        self.assertRed("pkg/svc.go:NewServer", "task-x.md")

    def test_renamed_method_receiver_is_red(self):
        self.doc("see `pkg/svc.go:Server.Handle`\n")
        self.write("pkg/svc.go", GO_FIXTURE.replace("func (s *Server) Handle(", "func (s *Server) Serve("))
        self.assertRed("Server.Handle")

    def test_removed_const_inside_a_block_is_red(self):
        self.doc("see `pkg/svc.go:ModeB`\n")
        self.assertGreen()
        self.write("pkg/svc.go", GO_FIXTURE.replace("\tModeB\n", ""))
        self.assertRed("pkg/svc.go:ModeB")

    def test_name_declared_only_in_a_raw_string_is_red(self):
        self.doc("see `pkg/svc.go:FakeInRaw`\n")
        self.assertRed("pkg/svc.go:FakeInRaw")

    def test_a_file_that_exists_nowhere_is_a_hard_error(self):
        self.doc("see `pkg/gone.go:Anything`\n")
        self.assertRed("pkg/gone.go")

    def test_cited_func_moved_within_its_file_is_green(self):
        self.doc("see `pkg/svc.go:Last` and `pkg/svc.go:NewServer`\n")
        self.assertGreen()
        moved = GO_FIXTURE.replace("func Last() {}\n", "")
        moved = moved.replace("// NewServer builds one.", "func Last() {}\n\n// NewServer builds one.")
        self.write("pkg/svc.go", "// a new header comment\n// of three lines\n// above everything\n" + moved)
        self.assertGreen()

    def test_comments_rewritten_above_a_cited_func_are_green(self):
        self.doc("see `pkg/svc.go:Server.Handle`\n")
        self.write("pkg/svc.go", GO_FIXTURE.replace(
            "func (s *Server) Handle(", "// Handle was rewritten, and now says a good many more words\n// than it did when this citation was written.\n// It runs to three lines.\nfunc (s *Server) Handle("))
        self.assertGreen()

    def test_sh_line_citation_is_red_unpinned_and_green_pinned(self):
        """A script has no declaration form, so with the door shut its line can be cited only in a pinned record."""
        self.write("pkg/run.sh", "#!/bin/sh\necho one\necho two\n")
        c = _git_commit_all(self.repo, "a script")
        self.doc("the script prints at `pkg/run.sh:2` and `pkg/run.sh:3`\n")
        self.assertRed("pin this doc")
        self.doc(f"# R\n\n<!-- citations-at: {c} -->\n\nthe script prints at `pkg/run.sh:2` and `pkg/run.sh:3`\n")
        self.assertGreen()

    def test_python_removed_method_is_red(self):
        self.doc("see `pkg/tool.py:Tool.arun`\n")
        self.assertGreen()
        self.write("pkg/tool.py", PY_FIXTURE.replace("    async def arun(self):\n        return 2\n", ""))
        self.assertRed("pkg/tool.py:Tool.arun")


class TestPinnedDocs(_LintRepo):
    """CC0.a: a record is checked against the commit it describes, never against HEAD, and --update never rewrites it."""

    def pinned(self, body, commit=None):
        return f"# A record\n\n<!-- citations-at: {commit or self.base} -->\n\n{body}"

    def test_pinned_doc_whose_code_changed_at_head_is_green(self):
        self.doc(self.pinned("the guard is at `pkg/svc.go:60` and `pkg/svc.go:61-62`\n"))
        self.assertGreen()
        # HEAD's file is now a different, shorter file: an unpinned doc would be red several ways
        self.write("pkg/svc.go", "package svc\n")
        _git_commit_all(self.repo, "rewrite")
        self.assertGreen()

    def test_pinned_path_absent_at_its_commit_is_red(self):
        self.doc(self.pinned("see `pkg/later.go:3`\n"))
        self.write("pkg/later.go", "package svc\n\nfunc Later() {}\n")
        _git_commit_all(self.repo, "add later.go after the pin")
        self.assertRed("pkg/later.go", "absent at", argv=())

    def test_pinned_line_past_eof_at_its_commit_is_red(self):
        n = len(GO_FIXTURE.rstrip("\n").split("\n"))
        self.doc(self.pinned(f"see `pkg/svc.go:{n + 5}`\n"))
        self.assertRed("pkg/svc.go", "past the end")

    def test_pinned_range_end_past_eof_is_red(self):
        n = len(GO_FIXTURE.rstrip("\n").split("\n"))
        self.doc(self.pinned(f"see `pkg/svc.go:{n - 1}-{n + 9}`\n"))
        self.assertRed("past the end")

    def test_a_marker_quoted_in_code_is_notation_not_a_pin(self):
        """The doc that introduces the scheme quotes the marker in a code span and a fence; neither pins it, and neither is a misplaced marker."""
        self.doc("# Scheme\n\n" + "\n" * 60 + "A record carries `<!-- citations-at: <commit> -->` near its top, e.g.\n\n"
                 "```\n<!-- citations-at: abcdef1 -->\n```\n\nsee `pkg/svc.go:Server.Handle`\n")
        self.assertGreen()
        self.assertEqual(qcl.pinned_marker(self.read("docs/task-x.md")), (None, None))

    def test_en_dash_range_and_comma_list_numbers_are_all_checked(self):
        n = len(GO_FIXTURE.rstrip("\n").split("\n"))
        self.doc(self.pinned(f"ranges `pkg/svc.go:3–{n - 1}` and lists `pkg/svc.go:3,10,{n - 2}` are inside the file\n"))
        self.assertGreen()
        self.doc(self.pinned(f"an en-dash range past EOF `pkg/svc.go:3–{n + 40}`\n"), name="task-y.md")
        self.assertRed("past the end", "task-y.md")
        _git(self.repo, "rm", "-q", "-f", "docs/task-y.md")
        self.doc(self.pinned(f"a list with one late number `pkg/svc.go:3,10,{n + 40}`\n"), name="task-z.md")
        qcl.PINNED_FILE_CACHE.clear()
        self.assertRed("past the end", "task-z.md")

    def test_pinned_to_a_commit_that_does_not_resolve_is_red(self):
        self.doc(self.pinned("see `pkg/svc.go:3`\n", commit="deadbeefdeadbeef"))
        self.assertRed("deadbeef")

    def test_marker_must_be_near_the_top(self):
        self.doc("# A record\n" + "\n" * 60 + f"<!-- citations-at: {self.base} -->\nsee `pkg/svc.go:3`\n")
        self.assertRed("near the top")

    def test_update_never_rewrites_a_pinned_doc_and_indexes_nothing_for_it(self):
        text = self.pinned("see `pkg/svc.go:60`\n")
        self.doc(text, name="record-x.md")
        _git(self.repo, "commit", "-q", "-am", "pinned doc")
        # shift the cited line at HEAD: an unpinned doc would be re-pointed
        self.write("pkg/svc.go", "// two\n// new lines\n" + GO_FIXTURE)
        _git_commit_all(self.repo, "shift")
        self.assertGreen(["--update"])
        self.assertEqual(self.read("docs/record-x.md"), text)
        self.assertNotIn("record-x.md", pathlib.Path(self.repo, "docs", "citation-index.md").read_text())
        self.assertGreen()

    def test_unpinned_doc_with_the_same_numbers_is_red(self):
        """The control for the case above: the same numbers in an UNPINNED doc are refused outright (the door), not tracked."""
        self.doc("see `pkg/svc.go:60`\n", name="live-x.md")
        self.assertRed("pin this doc")

    def test_aikit_path_in_a_pinned_doc_resolves_in_the_module_cache_at_the_commits_version(self):
        # go.mod at the pinned commit requires aikit v9.9.9; the cache holds that version only
        self.write("go.mod", "module example.com/goinfer\n\ngo 1.22\n\nrequire github.com/townsendmerino/aikit v9.9.9\n")
        c = _git_commit_all(self.repo, "go.mod pins aikit v9.9.9")
        cache = pathlib.Path(self.repo) / "mc"
        mod = cache / "github.com" / "townsendmerino" / "aikit@v9.9.9" / "linalg"
        mod.mkdir(parents=True)
        (mod / "quant.go").write_text("package linalg\n\nfunc Quant() {}\n")
        qcl._MODCACHE = cache
        qcl.PINNED_FILE_CACHE.clear()
        self.doc(self.pinned("see `linalg/quant.go:3`\n", commit=c))
        self.assertGreen()
        # past the end of THAT version's file: red
        self.doc(self.pinned("see `linalg/quant.go:40`\n", commit=c))
        qcl.PINNED_FILE_CACHE.clear()
        self.assertRed("past the end")

    def test_missing_module_is_a_missing_download_not_a_skip(self):
        self.write("go.mod", "module example.com/goinfer\n\ngo 1.22\n\nrequire github.com/townsendmerino/aikit v9.9.9\n")
        c = _git_commit_all(self.repo, "go.mod pins aikit v9.9.9")
        self.doc(self.pinned("see `linalg/quant.go:3`\n", commit=c))
        self.assertRed("CANNOT SEARCH", "go mod download")


class TestCloseTheDoor(_LintRepo):
    """CC0.c: a path:line in an unpinned doc is red once the door is closed. The switch is explicit so the
    tests state which behaviour they exercise whatever the shipped default is."""

    def test_line_citation_in_an_unpinned_doc_is_red_when_the_door_is_closed(self):
        self.doc("see `pkg/svc.go:60`\n")
        qcl.CLOSE_THE_DOOR = False
        self.assertGreen(["--update"])
        qcl.CLOSE_THE_DOOR = True
        self.assertRed("pin this doc (`citations-at`) or name the declaration", "pkg/svc.go:60")

    def test_symbol_and_pinned_citations_stay_green_when_the_door_is_closed(self):
        self.doc("see `pkg/svc.go:Server.Handle`\n", name="live-x.md")
        self.doc(f"# R\n\n<!-- citations-at: {self.base} -->\n\nsee `pkg/svc.go:60`\n", name="record-x.md")
        qcl.CLOSE_THE_DOOR = True
        self.assertGreen()

    def test_the_door_message_names_every_offending_doc(self):
        self.doc("see `pkg/svc.go:60`\n", name="live-a.md")
        self.doc("see `pkg/svc.go:61` and `pkg/tool.py:2`\n", name="live-b.md")
        qcl.CLOSE_THE_DOOR = True
        self.assertRed("live-a.md", "live-b.md", "pkg/tool.py:2")



if __name__ == "__main__":
    unittest.main()
