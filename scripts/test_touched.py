#!/usr/bin/env python3
"""touched.py runs the source-scan tests (scripts/touched_scans.py) for the files they read, and the table covers every test
that reads Go source. Run: python3 -B scripts/test_touched.py"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import touched_scans  # noqa: E402

_spec = importlib.util.spec_from_file_location("touched", os.path.join(HERE, "touched.py"))
touched = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(touched)  # main() runs only under __main__


def selected(changed):
    tests, _ = touched.plan(set(changed))
    return {(d, t) for (_, d), sel in tests.items() if sel for t in re.findall(r"\w+", sel)}


class VerdictLine(unittest.TestCase):
    """touched.py ends with one VERDICT line, so `| tail -1` reads the outcome instead of a pipe's exit status."""

    RAN = ["gofmt", "build root", "vet root (1 pkg)", "test cmd/x", "test cmd/y"]

    def test_green_is_pass(self):
        head, line, rc = touched.summary(0, self.RAN, [])
        self.assertEqual(head, "touched: green (CI runs the rest on push)")
        self.assertTrue(line.startswith("VERDICT: PASS — touched: 5 checks passed"), line)
        self.assertEqual(rc, 0)

    def test_red_is_fail(self):
        head, line, rc = touched.summary(2, self.RAN, [])
        self.assertTrue(head.startswith("touched: RED"), head)
        self.assertTrue(line.startswith("VERDICT: FAIL — touched: 2 of 5 checks failed"), line)
        self.assertEqual(rc, 1)

    def test_skips_are_named(self):
        _, line, _ = touched.summary(0, self.RAN[:3], ["cuda", "test decoder"])
        self.assertTrue(line.startswith("VERDICT: PASS"))
        self.assertIn("did not run: cuda, test decoder", line)

    def test_everything_skipped_is_inconclusive(self):
        """Only gofmt ran, every module was skipped (cuda outside the Mac's go.work, say): nothing was judged."""
        for ran in (["gofmt"], []):
            head, line, rc = touched.summary(0, ran, ["cuda"])
            self.assertNotIn("green", head)
            self.assertTrue(line.startswith("VERDICT: INCONCLUSIVE"), line)
            self.assertEqual(rc, 2)

    def test_every_exit_of_run_prints_a_verdict(self):
        """Each `return` in main() and run() follows a VERDICT print, so no path ends without one."""
        src = open(os.path.join(HERE, "touched.py")).read()
        for fn in ("main", "run"):
            body = re.search(rf"^def {fn}\(.*?(?=^def |^if __name__)", src, re.M | re.S).group(0)
            lines = body.splitlines()
            for i, l in enumerate(lines):
                if l.strip().startswith("return ") and "run(args, run_for)" not in l:
                    self.assertRegex(lines[i - 1], r"print\((verdict\(|line\))", f"{fn}: `{l.strip()}` has no VERDICT print before it")


class PlanSelectsTheScanTests(unittest.TestCase):
    def test_a_decoder_source_change_selects_the_dispatch_census(self):
        # 352fea05: a new file in decoder/ added two type switches, and the census that counts them did not run.
        self.assertIn(("decoder", "TestDispatchCensus"), selected(["decoder/recommended_sampling.go"]))

    def test_a_file_read_from_another_package_selects_its_reader(self):
        self.assertIn(("decoder", "TestFP8_shapeCheckDistinguishesTranspose"), selected(["decoder/quant/fp8.go"]))

    def test_a_scanner_in_an_untouched_package_runs(self):
        self.assertIn(("internal/app/cliutil", "TestLdflagsTargetsExist"), selected(["internal/app/serveapp/openai.go"]))

    def test_a_kernel_source_change_selects_its_readers_with_no_go_file_changed(self):
        got = selected(["backend/cuda/moe.cu"])
        self.assertIn(("backend/cuda/kernels", "TestKernelFMALint"), got)
        self.assertIn(("decoder", "TestResidentMoECap_pinsKernelConstant"), got)

    def test_a_metal_shader_change_selects_its_package_and_graded_pin(self):
        tests, _ = touched.plan({"backend/metal/kernels/decode_norm.metal"})
        self.assertIn(("backend/metal", "backend/metal/kernels"), tests)
        self.assertIsNone(tests[("backend/metal", "backend/metal/kernels")])
        tests_vit, _ = touched.plan({"backend/metal/internal/gpudev/vit_siglip.metal"})
        self.assertIn(("backend/metal", "backend/metal/internal/gpudev"), tests_vit)
        self.assertIsNone(tests_vit[("backend/metal", "backend/metal/internal/gpudev")])
        got = selected(["backend/metal/kernels/decode_attention_fa_blk.metal"])
        self.assertIn(("backend/metal", "TestAttnFABlkIsTheGradedKernel"), got)

    def test_a_doc_a_test_reads_selects_it(self):
        self.assertIn(("decoder", "TestEnvVars_docAndCodeAgree"), selected(["docs/env-vars.md"]))

    def test_a_new_model_fixture_selects_the_fixture_censuses(self):
        # 1ea85681: a new fixture directory was in neither census, and touched.py ran neither, so CI went red.
        got = selected(["testdata/gemma4-moe-kv-tiny32/config.json", "testdata/gemma4-moe-kv-tiny32/model.safetensors"])
        self.assertIn(("decoder", "TestSerializeCensus_everyFixtureIsListedOrExcluded"), got)
        self.assertIn(("decoder", "TestSeams_subCaptureRefusedWhereGateSCannotRun"), got)

    def test_a_ci_workflow_change_selects_its_readers(self):
        got = selected([".github/workflows/ci.yml"])
        for t in [("internal/runtime/sampling", "TestSweepCoverage_fullSweepRunsSomewhere"), ("multimodal", "TestRaceSubset_runsInFullSomewhere"),
                  ("internal/runtime/kvmgr", "TestOracle_runsInFullSomewhere"), ("chat", "TestReleasing_standaloneBuildGateExists")]:
            self.assertIn(t, got)

    def test_an_unscanned_change_selects_nothing_extra(self):
        self.assertEqual(selected(["LICENSE"]), set())

    def test_a_runtime_change_selects_the_censuses_that_pin_it_from_decoder(self):
        # The cache's fields are declared in a leaf and classified in decoder; a family is registered in a leaf and profiled in
        # decoder. Any non-test file under decoder/ or internal/runtime/ selects them, whatever its package is called.
        for changed in ("internal/runtime/cpuengine/kvcache.go", "internal/runtime/a_new_leaf/x.go", "decoder/arch/arch.go"):
            got = selected([changed])
            for t in ("TestKVCache_everyFieldHasAState", "TestResidentAdmission_registryCovered", "TestHardwareMatrix_fresh",
                      "TestLoaderIdentity_table", "TestKVBytesForCtx_isTheDeclaredRows"):
                self.assertIn(("decoder", t), got, changed)
            self.assertIn(("internal/guard/memgate", "TestMemGate_cpu"), got, changed)
        got = selected(["internal/runtime/cpuengine/kvcache_test.go", "internal/app/serveapp/openai.go", "internal/runtime/x/testdata/y.go"])
        self.assertNotIn(("internal/guard/memgate", "TestMemGate_cpu"), got, "a test file, another tree and a fixture are not the runtime's code")

    def test_a_generated_table_selects_its_freshness_check(self):
        self.assertIn(("decoder", "TestHardwareMatrix_fresh"), selected(["docs/hardware-matrix.md"]))
        self.assertIn(("pull", "TestReadme_familyCountMatchesTheMatrix"), selected(["README.md"]))
        self.assertIn(("internal/guard/gateidentity", "TestIdentityAssets_coverTheManifest"), selected(["testdata/parity/qwen2.json"]))

    def test_any_go_change_runs_the_pull_request_lanes_quality_scans(self):
        # The lint lane runs the package whole under its own tag on every pull request, so the day loop does too: a ratchet a
        # change breaks is red here first.
        for changed in ("cmd/gate/main.go", "backend/metal/x_test.go", "internal/guard/qualityscan/testdata/funclen.allow"):
            tests, _ = touched.plan({changed})
            self.assertIn(("", "internal/guard/qualityscan"), tests, changed)
            self.assertIsNone(tests[("", "internal/guard/qualityscan")], "the whole package, not a -run")
        tag = touched.PKG_TAGS["internal/guard/qualityscan"]
        ci = open(os.path.join(ROOT, ".github", "workflows", "ci.yml")).read()
        self.assertIn(f"go test -count=1 -tags {tag} ./internal/guard/qualityscan/", ci, "the day loop's tag is the lint lane's")


# Patterns that mark a function body as reading source from disk. A helper that matches marks every test in its package that
# calls it, so a scanner that reads through a helper is caught too.
READS = re.compile(r'parser\.Parse(File|Dir)\(|"\*\.go"|"\*\.cu"|\.go"\)|\.cu"\)|HasSuffix\([^)]*"\.(go|cu)"\)|'
                   r'"_test\.go"|go/ast|token\.NewFileSet')

# Test functions the patterns flag that read no repository source from disk: each parses an in-memory string, or reads only
# go.mod or fixtures.
NOT_SOURCE_READERS = {
    "TestLogger_getsLevelTextAndCaller": "compares a stack frame's file name; it reads no file",
    "TestScan_flagsWhatItShouldAndNothingElse": "parses a Go source held in the test as a string; it reads no file",
    "TestSetLogger_takesWhatALoadWrites": "compares a stack frame's file name; it reads no file",
    "TestCompare_classes": "sides built by hand; the paths it names are strings, and it reads no file",
    "TestGitSource_preMoveRevisionReadsTheInternalRegroup": "a synthetic repo in a temp dir (a pre-move and a post-move commit)",
    "TestCommentsOnly_gitCommitGreenAndRed": "a synthetic repo in a temp dir",
    "TestQuick_checkpointAloneGetsItsOwnCell": "a synthetic repo in a temp dir (ckFixture)",
    "TestCommentsOnly_worktreeMode": "a synthetic repo in a temp dir",
    "TestIdentityDumperTemplate_isGoAndHasItsMarkers": "parses the embedded dumper template",
    "TestLedger_classifyOutcomes": "a scratch tree (ledgerScratch)",
    "TestLedger_sourceKeyMatchesThePythonImplementation": "a scratch tree (ledgerScratch)",
    "TestMutation_sedInvocationIsPortable": "sed arguments only",
    "TestQuick_deferCells_checkpointCell": "a synthetic package (ckFixture)",
    "TestQuick_deferCells_slowOfAnUntouchedDeviceModule": "a synthetic package (ckFixture)",
    "TestScanTree_skipsIgnoredFiles": "two synthetic files in a temp dir",
    "TestForTestGlobals_scannerAccuracy": "synthetic Go files in a temp dir",
    "TestQuick_checkpointLoadersGetTheirOwnCell": "a synthetic package (ckFixture)",
    "TestQuick_smallCheckpointPackageRunsWholeInTheLane": "a synthetic package (ckFixture)",
    "TestRowGofmtAndLint": "path strings in a synthetic report",
    "TestRepoGroup_plants": "file names in a synthetic CI step's gofmt output",
    "TestGoCodeTokens_commentsOnly": "parses in-memory sources",
    "TestParent": "a synthetic package in a temp dir",
    "TestSettleCalls_callWrittenInsideReasoning": "a file name inside a tool call's arguments",
    "TestForcedCall_harmonyBudgetCloseParsesAsACall": "a file name inside a tool call's arguments",
    "TestBuilderFile": "asks BuilderFile about paths given as strings; it reads no file",
    "TestFuncKey_namesTheReceiverType": "parses a Go source held in the test as a string; it reads no file",
    "TestReachableLiterals_aRequireHeavyCallIsAHeavyCheck": "a synthetic package written to a temp dir",
}


def functions(path):
    """(name, body) for each top-level func in a Go file."""
    with open(path, encoding="utf-8", errors="replace") as f:
        src = f.read()
    parts = re.split(r"(?m)^func ", src)
    for p in parts[1:]:
        m = re.match(r"(?:\([^)]*\)\s*)?(\w+)", p)
        if m:
            yield m.group(1), re.sub(r"(?m)^\s*//.*$", "", p)  # a comment that names a path reads nothing


def unlisted():
    """{test name: file} for each test the patterns mark as reading source that no table names."""
    listed = {s.test for s in touched_scans.SCANS} | set(touched_scans.EXCLUDED) | set(NOT_SOURCE_READERS)
    whole = {s.pkg for s in touched_scans.SCANS if s.test == touched_scans.WHOLE}  # every test of these is selected
    flagged = {}
    for dirpath, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "testdata", "node_modules")]
        if os.path.relpath(dirpath, ROOT) in whole:
            continue
        gos = [os.path.join(dirpath, f) for f in files if f.endswith(".go")]
        if not any(f.endswith("_test.go") for f in gos):
            continue
        # Plain calls only: a method call (x.f) does not reach a package helper f, and init is never called by name.
        funcs = [(n, b, f, set(re.findall(r"(?<![.\w])(\w+)\(", b)) - {"init"}) for f in gos for n, b in functions(f) if n != "init"]
        # Propagate through helpers in test files only: a non-test helper (cmd/gate's scanners) is mostly driven against a
        # synthetic tree in a temp dir, and the tests that point one at the real tree are listed in SCANS by hand.
        tfuncs = [x for x in funcs if x[2].endswith("_test.go")]
        readers = {n for n, b, _, _ in tfuncs if READS.search(b)}
        for _ in range(3):  # helpers that call helpers
            readers |= {n for n, _, _, calls in tfuncs if calls & readers}
        for n, _, f, _ in funcs:
            if f.endswith("_test.go") and re.match(r"(Test|Fuzz)[A-Z_]", n) and n in readers and n not in listed:
                flagged[n] = os.path.relpath(f, ROOT)
    return flagged


class NotesCheckout(unittest.TestCase):
    """Task, queue, plan and scoping docs live in a sibling notes checkout once they leave the public tree; touched.py maps its
    changes to the docs/ paths the scans read, and adds nothing when there is no checkout."""

    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.root = os.path.join(self._td.name, "goinfer")
        self.notes = os.path.join(self._td.name, "goinfer-notes")
        os.mkdir(self.root)

    def tearDown(self):
        self._td.cleanup()

    def git(self, *a):
        subprocess.run(["git", "-C", self.notes, "-c", "user.email=t@example.com", "-c", "user.name=T", *a], check=True,
                       capture_output=True)

    def make_notes(self):
        os.mkdir(self.notes)
        self.git("init", "-q")
        for rel in ("tasks/task-a.md", "queues/QUEUE.md", "plans/plan-x.md", "completed/old.md", "README.md"):
            os.makedirs(os.path.dirname(os.path.join(self.notes, rel)), exist_ok=True)
            open(os.path.join(self.notes, rel), "w").write("one\n")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "init")

    def test_no_checkout_adds_nothing(self):
        self.assertEqual(touched_scans.notes_changes(self.root), set())

    def test_a_clean_checkout_adds_nothing(self):
        self.make_notes()
        self.assertEqual(touched_scans.notes_changes(self.root), set())

    def test_edits_and_new_files_map_to_their_docs_paths(self):
        self.make_notes()
        for rel in ("tasks/task-a.md", "queues/QUEUE.md", "completed/old.md", "README.md"):
            open(os.path.join(self.notes, rel), "a").write("two\n")
        open(os.path.join(self.notes, "plans/plan-new.md"), "w").write("x\n")  # untracked
        self.assertEqual(touched_scans.notes_changes(self.root),
                         {"docs/tasks/task-a.md", "docs/QUEUE.md", "docs/completed/old.md", "docs/plan-new.md"})

    def test_a_notes_change_selects_the_scan_that_reads_docs(self):
        self.make_notes()
        open(os.path.join(self.notes, "tasks/task-a.md"), "a").write("two\n")
        got = selected(touched_scans.notes_changes(self.root))
        self.assertIn(("internal/guard/testcmdtags", "TestDocs_goTestCommandsCompileTheTestsTheyName"), got)


class TableIsComplete(unittest.TestCase):
    def test_every_source_reading_test_is_listed_or_excluded(self):
        self.maxDiff = None
        self.assertEqual(unlisted(), {}, "tests that read source and are in neither touched_scans.SCANS nor EXCLUDED "
                         "(add each to SCANS with the paths it reads, or to EXCLUDED or NOT_SOURCE_READERS with why)")

    def test_every_entry_matches_the_tree(self):
        self.maxDiff = None
        self.assertEqual(touched_scans.problems(ROOT), [], "SCANS names a test its package does not declare, or a scope that "
                         "selects no file: point each at where the test or the file is now, or drop it")


class _KeepsRAN(unittest.TestCase):
    """check_module() appends to touched.RAN; each test here leaves it as it found it."""

    def setUp(self):
        self._ran = list(touched.RAN)
        touched.RAN.clear()

    def tearDown(self):
        touched.RAN[:] = self._ran


def _other_os():
    """A GOOS that is not this host's: a target whose test binaries cannot execute here."""
    return "linux" if sys.platform == "darwin" else "darwin"


def _scratch(root, files):
    for rel, body in files.items():
        os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
        with open(os.path.join(root, rel), "w") as f:
            f.write(body)


class PackagesForTheTarget(_KeepsRAN):
    """classify() and check_module() on scratch modules with one package of each shape. For a target that is not the host (go list and go
    vet cross-check it; go test is not run, since its binaries cannot execute): a package whose files are all excluded is skipped by name, one
    with no test file for the target is a named skip, not a vacuous pass, and any other go list error fails. For the host: a -run that selects
    only excluded tests is a named skip, and a diff whose every package is excluded judges nothing. Needs a Go toolchain (on nobara, put
    /usr/local/go/bin on PATH first: the Bash tool's PATH has none, and this class then skips)."""

    @classmethod
    def setUpClass(cls):
        import shutil
        import tempfile
        if shutil.which("go") is None:
            raise unittest.SkipTest("no go toolchain on PATH")
        cls.tmp = tempfile.mkdtemp(prefix="touched-pkgs-")
        cls.target = _other_os()
        tag = "//go:build " + ("linux" if cls.target == "darwin" else "darwin") + "\n\n"  # a host-only file: excluded for the target
        _scratch(os.path.join(cls.tmp, "m"), {
            "go.mod": "module m\n\ngo 1.21\n",
            "empty/a.go": tag + "package empty\n",
            "empty/a_test.go": tag + "package empty\n",
            "testonly/a.go": tag + "package testonly\n",
            "testonly/a_test.go": "package testonly\n\nimport \"testing\"\n\nfunc TestX(t *testing.T) {}\n",
            "notests/a.go": "package notests\n\nfunc F() int { return 1 }\n",
            "notests/a_test.go": tag + "package notests\n",
            "ok/a.go": "package ok\n\nfunc F() int { return 1 }\n",
            "ok/a_test.go": "package ok\n\nimport \"testing\"\n\nfunc TestX(t *testing.T) {}\n",
        })
        _scratch(os.path.join(cls.tmp, "b"), {  # its own module, so its failing build cannot hide one of m's
            "go.mod": "module b\n\ngo 1.21\n",
            "broken/a.go": "pakage broken\n",
        })
        never = "//go:build touched_never\n\n"
        _scratch(os.path.join(cls.tmp, "h"), {  # for the host: tests run
            "go.mod": "module h\n\ngo 1.21\n",
            "p/a.go": "package p\n\nfunc F() int { return 1 }\n",
            "p/a_test.go": "package p\n\nimport \"testing\"\n\nfunc TestPortable(t *testing.T) {}\n",
            "p/b_test.go": never + "package p\n\nimport \"testing\"\n\nfunc TestExcluded(t *testing.T) {}\n",
            "gone/a.go": never + "package gone\n",
            "gone/a_test.go": never + "package gone\n",
        })
        cls.cwd = os.getcwd()
        os.chdir(cls.tmp)
        base = dict(os.environ, GOWORK="off", GOFLAGS="", CGO_ENABLED="0")
        base.pop("GOARCH", None)
        cls.env = dict(base, GOOS=cls.target, GOARCH="arm64" if cls.target == "darwin" else "amd64")
        cls.hostenv = base
        cls.host = subprocess.run(["go", "env", "GOHOSTOS"], text=True, capture_output=True, env=base).stdout.strip()

    @classmethod
    def tearDownClass(cls):
        import shutil
        os.chdir(cls.cwd)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _check(self, env, *a, **k):
        import contextlib
        import io
        from unittest import mock
        out = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(out):
            bad = touched.check_module(*a, **k)
        return bad, out.getvalue()

    def test_classify(self):
        got = touched.classify("m", ["./empty", "./testonly", "./notests", "./ok"], [], self.env)
        self.assertEqual(got, {"empty": "empty", "testonly": "ok", "notests": "notests", "ok": "ok"})
        got = touched.classify("b", ["./broken"], [], self.env)
        self.assertTrue(got["broken"].startswith("error: "), got)

    def test_classify_keys_a_package_with_no_dir_by_its_arg(self):
        # go list gives a missing package no Dir. It is keyed by the arg, as an error, and nothing else is keyed in its place.
        got = touched.classify("m", ["./ok", "./nope"], [], self.env)
        self.assertEqual(set(got), {"ok", "nope"}, got)
        self.assertEqual(got["ok"], "ok")
        self.assertIn("directory not found", got["nope"])  # go list's own error for it, not the no-entry fallback

    def test_check_module_skips_by_name(self):
        skips = []
        files = ["m/empty/a.go", "m/testonly/a_test.go", "m/notests/a.go", "m/ok/a.go"]
        tests = {("m", "m/empty"): "", ("m", "m/notests"): "", ("m", "m/ok"): ""}
        bad, text = self._check(self.env, "m", "m", files, tests, self.target, skips, host_os=self.host)
        self.assertEqual(bad, 0, text)  # no broken package shares this module, so a failing build here is m's own
        self.assertRegex(text, r"ok\s+build m ")
        self.assertIn(f"skip vet m/empty: build constraints exclude all its files for {self.target}", text)
        self.assertIn(f"vet m/empty (no files for {self.target})", skips)
        self.assertIn(f"test m/empty (no files for {self.target})", skips)
        self.assertIn(f"test m/notests (no test files for {self.target})", skips)
        self.assertRegex(text, r"ok\s+vet m \(3 pkg\)")  # testonly, notests and ok vet clean; a vet failure would fail this
        self.assertIn(f"test m/ok (cross target {self.target})", skips)  # a cross target's tests do not run
        self.assertNotIn("test m/ok", " ".join(touched.RAN))

    def test_check_module_fails_the_broken(self):
        skips = []
        bad, text = self._check(self.env, "b", "b", ["b/broken/a.go"], {("b", "b/broken"): ""}, self.target, skips, host_os=self.host)
        self.assertRegex(text, r"FAIL\s+build b ")
        self.assertIn("FAIL list b/broken", text)  # a go list error is a failure, never a skip
        self.assertNotIn("b/broken (no files", " ".join(skips))
        # The failures are exactly: the module build, broken's listing before vet, and its listing before test.
        self.assertEqual(bad, 3, text)

    def test_a_run_selecting_only_excluded_tests_is_a_skip(self):
        skips = []
        bad, text = self._check(self.hostenv, "h", "h", ["h/p/b_test.go"], {("h", "h/p"): "^(TestExcluded)$"}, self.host, skips, host_os=self.host)
        self.assertEqual(bad, 0, text)
        self.assertIn("no selected test is built for this target", text)
        self.assertTrue(any("no selected test built" in x for x in skips), skips)
        self.assertNotIn("test h/p -run ^(TestExcluded)$", touched.RAN)  # not counted as a check that passed
        skips = []
        bad, text = self._check(self.hostenv, "h", "h", ["h/p/a_test.go"], {("h", "h/p"): "^(TestPortable)$"}, self.host, skips, host_os=self.host)
        self.assertEqual(bad, 0, text)
        self.assertIn("test h/p -run ^(TestPortable)$", touched.RAN)

    def test_every_package_excluded_judges_nothing(self):
        skips = []
        bad, text = self._check(self.hostenv, "h", "h", ["h/gone/a.go"], {("h", "h/gone"): None}, self.host, skips, host_os=self.host)
        self.assertEqual(bad, 0, text)
        self.assertEqual(touched.RAN, [], text)  # no module build counted, so the run's summary is INCONCLUSIVE, not PASS
        self.assertIn(f"build h (no files for {self.host})", skips)
        self.assertEqual(touched.summary(bad, touched.RAN, skips)[2], 2)


class CudaCrossChecks(_KeepsRAN):
    """check_module() on backend/cuda vets the other OS's build module-wide, both ways, and a target that is not the host runs no go test.
    It drives check_module through a spy for touched.sh, so it needs no Go toolchain."""

    def _calls(self, target, host, files=("backend/cuda/x.go",), classified=(((".", "ok"),))):
        import contextlib
        import io
        from unittest import mock
        calls = []

        def spy(*a, **k):
            calls.append((a, (k.get("env") or {}).get("GOOS")))
            return subprocess.CompletedProcess(a, 0, "", "")

        tests = {("backend/cuda", "backend/cuda"): "^(TestX)$"}
        with mock.patch.object(touched, "sh", spy), mock.patch.object(touched, "classify", lambda *a: dict(classified)), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.bad = touched.check_module("backend/cuda", "backend/cuda", files, tests, target, [], host_os=host)
        self.out = out.getvalue()
        return calls

    def test_both_ways(self):
        linux = [c for c in self._calls("linux", "linux") if c[0][:2] == ("go", "vet")]
        darwin = [c for c in self._calls("darwin", "darwin") if c[0][:2] in (("go", "vet"), ("go", "build"))]
        self.assertIn((("go", "vet", "./..."), "darwin"), linux, "a linux run must vet the darwin build of backend/cuda, module-wide")
        self.assertIn((("go", "vet", "-tags", "cuda goinfer_testhooks", "./..."), "linux"), darwin, "off linux: the linux build at the cuda tags, module-wide")
        self.assertIn((("go", "build", "-tags", "cuda goinfer_testhooks", "./..."), "linux"), darwin)
        self.assertIn((("go", "build", "./..."), "darwin"), darwin, "off linux, the host's own untagged build carries the target GOOS")

    def test_a_cross_target_runs_no_go_test(self):
        native = self._calls("linux", "linux")
        cross = self._calls("darwin", "linux")
        self.assertTrue(any(c[0][:2] == ("go", "test") for c in native), "a native target runs go test")
        self.assertFalse(any(c[0][:2] == ("go", "test") for c in cross), "GOOS=darwin on a linux host must not run go test")
        # The touched package's own vet at the target GOOS: the native run's module-wide darwin cross-vet is a different call.
        target_vet = (("go", "vet", "."), "darwin")
        self.assertIn(target_vet, cross, "but it still vets the touched package for darwin")
        self.assertNotIn(target_vet, native, "a call the native run also makes would not show the cross target was vetted")

    def test_a_cross_target_with_no_go_file_claims_no_build(self):
        calls = self._calls("darwin", "linux", files=())
        self.assertEqual([c for c in calls if c[0][:2] in (("go", "vet"), ("go", "build"), ("go", "test"))], [])
        self.assertIn("runs nothing for it", self.out)
        self.assertNotIn("builds and vets", self.out)

    def test_an_unclassified_package_fails(self):
        # classify() keying nothing for a touched package is a failure, never a default "ok" that vets and tests it.
        calls = self._calls("linux", "linux", classified=())
        self.assertFalse(any(c[0][:2] == ("go", "test") for c in calls), self.out)
        self.assertEqual(self.bad, 2, self.out)  # the listing before vet and the one before test, each counted as a failure
        self.assertIn("FAIL list backend/cuda/.", self.out)  # before vet
        self.assertIn("FAIL list backend/cuda ", self.out)  # before test
        self.assertNotIn((("go", "vet", "-tags", "cuda goinfer_testhooks", "."), "linux"), calls, "an unclassified package is not vetted")



class ClassifyKeys(unittest.TestCase):
    """classify() on go list output fed through a spy for touched.sh: which entries it keys, and what an arg with no keyable entry gets.
    Needs no Go toolchain."""

    def setUp(self):
        import tempfile
        self.root = os.path.realpath(tempfile.mkdtemp(prefix="touched-keys-"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def _classify(self, args, entries, rc=0, stderr=""):
        from unittest import mock
        out = "".join(json.dumps(e) + "\n" for e in entries)
        with mock.patch.object(touched, "sh", lambda *a, **k: subprocess.CompletedProcess(a, rc, out, stderr)):
            return touched.classify(self.root, args, [], {})

    def test_an_entry_with_neither_dir_nor_import_path_keys_nothing(self):
        # normpath("") is ".": an entry with no key must not land on the root package, least of all as "empty", which skips it.
        ok = {"Dir": self.root, "ImportPath": "m", "TestGoFiles": ["a_test.go"]}
        stray = {"Error": {"Err": "build constraints exclude all Go files in x"}}
        got = self._classify([".", "./x"], [ok, stray])
        self.assertEqual(got["."], "ok", got)
        self.assertEqual(set(got), {".", "x"}, got)
        self.assertTrue(got["x"].startswith("error: "), got)  # not "empty": a stray verdict is not this arg's

    def test_a_dir_outside_the_module_is_not_keyed_by_its_import_path(self):
        out = {"Dir": os.path.dirname(self.root), "ImportPath": "m/pkg", "TestGoFiles": ["a_test.go"]}
        got = self._classify(["./pkg"], [out])
        self.assertEqual(set(got), {"pkg"}, got)
        self.assertTrue(got["pkg"].startswith("error: "), got)

    def test_an_arg_with_no_entry_is_an_error(self):
        self.assertEqual(self._classify(["./y"], [], rc=1, stderr="go: boom\nmore"), {"y": "error: go: boom"})
        self.assertEqual(self._classify(["./y"], []), {"y": "error: go list reported nothing for it"})


class CrossTargetMessage(_KeepsRAN):
    """A cross-target test skip claims a build and vet only when the target's build or vet ran. Spies touched.sh and classify."""

    def _check(self, kinds):
        import contextlib
        import io
        from unittest import mock
        spy = lambda *a, **k: subprocess.CompletedProcess(a, 0, "", "")
        with mock.patch.object(touched, "sh", spy), mock.patch.object(touched, "classify", lambda *a: dict(kinds)), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            bad = touched.check_module("m", "m", ["m/a/x.go"], {("m", "m/b"): None}, "darwin", [], host_os="linux")
        return bad, out.getvalue()

    def test_built_says_so(self):
        bad, text = self._check({"a": "ok", "b": "ok"})
        self.assertEqual(bad, 0, text)
        self.assertIn("builds and vets only", text)

    def test_every_touched_package_excluded_claims_nothing(self):
        bad, text = self._check({"a": "empty", "b": "ok"})
        self.assertEqual(bad, 0, text)
        self.assertNotIn("builds and vets", text)
        self.assertIn("runs nothing for it", text)


class TableProblems(unittest.TestCase):
    """touched_scans.problems() on a scratch repository: an entry whose file moved or whose test left its package is named,
    and one the tree still matches is not. touched.py fails a run on any of them."""

    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.root = self._td.name
        _scratch(self.root, {
            "a/a.go": "package a\n",
            "a/a_test.go": "package a\n\nimport \"testing\"\n\nfunc TestHere(t *testing.T) {}\n",
            "b/b_test.go": "package b\n\nimport \"testing\"\n\nfunc TestMoved(t *testing.T) {}\n",
            "c/new.go": "package c\n",  # not added: a file the diff created counts
            "kernels/k.cu": "",
        })
        git = lambda *a: subprocess.run(["git", "-C", self.root, *a], check=True, capture_output=True)
        git("init", "-q")
        git("add", "a", "b", "kernels")

    def tearDown(self):
        self._td.cleanup()

    def problems(self, *scans):
        from unittest import mock
        with mock.patch.object(touched_scans, "SCANS", list(scans)):
            return touched_scans.problems(self.root)

    def test_an_entry_the_tree_matches_is_clean(self):
        S, F, NT, T, CU, UNDER = (touched_scans.Scan, touched_scans.F, touched_scans.NT, touched_scans.T, touched_scans.CU,
                                  touched_scans.UNDER)
        self.assertEqual(self.problems(S("a", "TestHere", F("a/a.go", "c/*.go") + NT("a", "c") + T("b") + CU("kernels") + UNDER("a")),
                                       S("a", touched_scans.WHOLE, touched_scans.TREE_GO)), [])

    def test_a_scope_whose_file_moved_is_named(self):
        S, F, NT, CU = touched_scans.Scan, touched_scans.F, touched_scans.NT, touched_scans.CU
        got = self.problems(S("a", "TestHere", F("a/a.go", "a/gone.go")), S("a", "TestHere", NT("b") + CU("a")),
                            S("a", "TestHere", F("gone/*.go") + touched_scans.UNDER("gone")))
        self.assertEqual(len(got), 5, got)  # a/gone.go, no non-test file in b, no .cu in a, nothing under gone/ twice
        self.assertIn("a TestHere: its scope file 'a/gone.go' selects no file (moved, renamed or deleted)", got)

    def test_a_file_deleted_from_disk_but_still_tracked_does_not_count(self):
        os.remove(os.path.join(self.root, "a", "a.go"))
        got = self.problems(touched_scans.Scan("a", "TestHere", touched_scans.F("a/a.go")))
        self.assertEqual(len(got), 1, got)

    def test_a_test_listed_under_a_package_it_left_is_named(self):
        got = self.problems(touched_scans.Scan("a", "TestMoved", touched_scans.F("a/a.go")),
                            touched_scans.Scan("kernels", touched_scans.WHOLE, touched_scans.F("a/a.go")))
        self.assertEqual(got, ["a TestMoved: not declared in a (moved, renamed or deleted)",
                               "kernels (whole package): kernels declares no test"])

    def test_a_tree_that_cannot_be_listed_is_one_problem_not_every_entry(self):
        from unittest import mock
        with mock.patch.object(touched_scans, "SCANS", [touched_scans.Scan("a", "TestHere", touched_scans.F("a/a.go"))] * 3):
            got = touched_scans.problems(os.path.join(self.root, "no-such-dir"))
        self.assertEqual(len(got), 1, got)
        self.assertIn("cannot list the files", got[0])

    def test_touched_fails_the_run_on_one(self):
        """run() counts a stale entry as a failed step, and a clean table as nothing."""
        import contextlib
        import io
        from unittest import mock
        out = io.StringIO()
        done = lambda *a, **k: subprocess.CompletedProcess(a, 0, "", "")
        with mock.patch.object(touched, "sh", done), mock.patch.object(touched, "plan", lambda *a: ({}, [])), \
                mock.patch.object(touched_scans, "notes_changes", lambda root: set()), \
                mock.patch.object(touched_scans, "problems", lambda root: ["a TestHere: its scope file 'a/gone.go' selects no file"]), \
                contextlib.redirect_stdout(out):
            rc = touched.run([], {})
        self.assertEqual(rc, 1, out.getvalue())
        self.assertIn("a TestHere: its scope file 'a/gone.go' selects no file", out.getvalue())
        self.assertTrue(out.getvalue().strip().splitlines()[-1].startswith("VERDICT: FAIL"), out.getvalue())
        out = io.StringIO()
        with mock.patch.object(touched, "sh", done), mock.patch.object(touched, "plan", lambda *a: ({}, [])), \
                mock.patch.object(touched_scans, "notes_changes", lambda root: set()), \
                mock.patch.object(touched_scans, "problems", lambda root: []), contextlib.redirect_stdout(out):
            rc = touched.run([], {})
        self.assertEqual(rc, 0, out.getvalue())
        self.assertTrue(out.getvalue().strip().splitlines()[-1].startswith("VERDICT: n/a"), out.getvalue())


class TestsLeftBehind(unittest.TestCase):
    """A file split out of a slow package leaves its test there. plan() selects slow/foo_test.go's tests for a change to
    leaf/foo.go when slow imports leaf and both are in one module, on a scratch tree."""

    def setUp(self):
        import tempfile
        from unittest import mock
        self._td = tempfile.TemporaryDirectory()
        self.root = self._td.name
        test = lambda pkg, *names: f"package {pkg}\n\nimport \"testing\"\n\n" + "".join(f"func {n}(t *testing.T) {{}}\n" for n in names)
        _scratch(self.root, {
            "go.mod": "module example.com/m\n\ngo 1.21\n",
            "slow/slow.go": "package slow\n\nimport (\n\t\"example.com/m/leaf\"\n\t\"example.com/m/sub/other\"\n)\n\nfunc F() { leaf.F(); other.F() }\n",
            "slow/foo_test.go": test("slow", "TestFoo", "TestFooEdge"),
            "slow/bar_test.go": test("slow", "TestBar"),
            "slow/own.go": "package slow\n",
            "slow/own_test.go": test("slow", "TestOwn"),
            "leaf/foo.go": "package leaf\n\nfunc F() {}\n",
            "leaf/nothing.go": "package leaf\n",
            "stranger/bar.go": "package stranger\n",  # slow does not import it
            "sub/go.mod": "module example.com/m/sub\n\ngo 1.21\n",
            "sub/other/foo.go": "package other\n\nfunc F() {}\n",  # imported, but another module's
        })
        self._patches = [mock.patch.object(touched, "ROOT", self.root), mock.patch.object(touched, "SLOW", {"slow"}),
                         mock.patch.object(touched_scans, "SCANS", [])]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self._td.cleanup()

    def test_a_leaf_change_selects_the_test_left_in_the_slow_package(self):
        tests, skipped = touched.plan({"leaf/foo.go"})
        self.assertEqual(tests[("", "slow")], "^(TestFoo|TestFooEdge)$")
        self.assertIsNone(tests[("", "leaf")], "and the leaf's own package runs whole, as before")
        self.assertEqual(skipped, [])

    def test_it_adds_to_the_tests_beside_a_change_in_the_slow_package(self):
        tests, _ = touched.plan({"leaf/foo.go", "slow/own.go"})
        self.assertEqual(tests[("", "slow")], "^(TestFoo|TestFooEdge|TestOwn)$")

    def test_only_a_package_the_slow_one_imports_and_only_in_its_module(self):
        for changed in ({"stranger/bar.go"}, {"sub/other/foo.go"}, {"leaf/nothing.go"}, {"leaf/foo_test.go"}):
            tests, _ = touched.plan(changed)
            self.assertNotIn(("", "slow"), tests, changed)


class PackageTags(_KeepsRAN):
    """A package in touched.PKG_TAGS is listed and tested under its own tag, in place of its module's. Spies touched.sh."""

    def test_the_test_run_carries_the_package_tag(self):
        import contextlib
        import io
        from unittest import mock
        calls = []

        def spy(*a, **k):
            calls.append(a)
            return subprocess.CompletedProcess(a, 0, "", "")

        seen = []

        def classify(mod, args, targs, env):
            seen.append((tuple(args), tuple(targs)))
            return {os.path.normpath(a): "ok" for a in args}

        tests = {("", "internal/guard/qualityscan"): None, ("", "chat"): None}
        with mock.patch.object(touched, "sh", spy), mock.patch.object(touched, "classify", classify), \
                contextlib.redirect_stdout(io.StringIO()):
            bad = touched.check_module("", "root", [], tests, "linux", [], host_os="linux")
        self.assertEqual(bad, 0)
        self.assertIn(("go", "test", "-tags", "quality_scan", "-count=1", "./internal/guard/qualityscan"), calls)
        self.assertIn(("go", "test", "-tags", "goinfer_testhooks", "-count=1", "./chat"), calls, "any other package keeps its module's tags")
        self.assertIn((("./internal/guard/qualityscan",), ("-tags", "quality_scan")), seen, "and it is classified under the same tag")


class WholeModuleVet(_KeepsRAN):
    """The root module is vetted whole, under each tag set CI's lint lane vets it with: a type or a test hook that moved breaks the
    packages that still use it and the files only a tag compiles, and the diff need touch none of them. Spies touched.sh."""

    def _vets(self, mod, label, files):
        import contextlib
        import io
        from unittest import mock
        calls = []

        def spy(*a, **k):
            calls.append(a)
            return subprocess.CompletedProcess(a, 0, "", "")

        classify = lambda mod, args, targs, env: {os.path.normpath(a): "ok" for a in args}
        with mock.patch.object(touched, "sh", spy), mock.patch.object(touched, "classify", classify), \
                contextlib.redirect_stdout(io.StringIO()):
            bad = touched.check_module(mod, label, files, {}, "linux", [], host_os="linux")
        self.assertEqual(bad, 0)
        return [c for c in calls if c[:2] == ("go", "vet")]

    def test_the_root_module_is_vetted_whole_under_the_lint_lanes_tag_sets(self):
        self.assertEqual(self._vets("", "root", ["chat/chat.go"]),
                         [("go", "vet", "-tags", "goinfer_testhooks", "./..."), ("go", "vet", "-tags", "realckpt goinfer_testhooks", "./...")])
        with open(os.path.join(ROOT, ".github", "workflows", "ci.yml")) as f:
            ci = f.read()
        for line in ("run: go vet -tags goinfer_testhooks ./...", "run: go vet -tags 'realckpt goinfer_testhooks' ./..."):
            self.assertIn(line, ci, "touched.VET_WHOLE mirrors the lint lane's vet steps; change both together")

    def test_any_other_module_vets_its_touched_packages(self):
        self.assertEqual(self._vets("backend/webgpu", "webgpu", ["backend/webgpu/residency.go"]),
                         [("go", "vet", "-tags", "gpu goinfer_testhooks", ".")])


class ScriptTests(unittest.TestCase):
    """touched.script_tests() on a scratch tree: a script test a workflow runs is selected when it changed, or when it names a
    changed file under scripts/ or .github/workflows/, by the file's name or by importing it."""

    def setUp(self):
        import tempfile
        from unittest import mock
        self._td = tempfile.TemporaryDirectory()
        self.root = self._td.name
        _scratch(self.root, {
            ".github/workflows/ci.yml": "      - run: python3 -B scripts/test_imports.py\n"
                                        "      - run: cd scripts && python3 test_names.py && python3 test_reads.py\n"
                                        "      - run: python3 -B scripts/test_neither.py\n"
                                        "      - run: python3 -B scripts/test_from.py && python3 -B scripts/test_list.py\n",
            "scripts/tool.py": "",
            "scripts/run.sh": "",
            "scripts/test_imports.py": "import os\nimport tool  # noqa: E402\n",
            "scripts/test_from.py": "from tool import helper\n",
            "scripts/test_list.py": "import json, tool\n",
            "scripts/test_names.py": "TOOL = os.path.join(HERE, \"tool.py\")\n",
            "scripts/test_reads.py": "subprocess.run([\"bash\", os.path.join(HERE, \"run.sh\")])\n"
                                     "ci = open(\".github/workflows/ci.yml\").read()\n",
            # Names that only contain the script's: this test's own, a longer module's, another file's, a compiled one's.
            "scripts/test_neither.py": "\"\"\"Run: python3 -B scripts/test_tool.py\"\"\"\nimport toolbox\nOTHER = \"mytool.py\"\nPYC = \"tool.pyc\"\n",
            "scripts/test_orphan.py": "import tool\n",  # no workflow runs it
        })
        self._patch = mock.patch.object(touched, "ROOT", self.root)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._td.cleanup()

    def test_a_changed_script_selects_the_tests_that_import_it_or_name_its_file(self):
        self.assertEqual(touched.script_tests({"scripts/tool.py"}),
                         {t: "scripts/tool.py" for t in ("scripts/test_imports.py", "scripts/test_from.py", "scripts/test_list.py",
                                                         "scripts/test_names.py")})

    def test_a_shell_script_and_a_workflow_select_the_test_that_names_them(self):
        self.assertEqual(touched.script_tests({"scripts/run.sh"}), {"scripts/test_reads.py": "scripts/run.sh"})
        self.assertEqual(touched.script_tests({".github/workflows/ci.yml"}), {"scripts/test_reads.py": ".github/workflows/ci.yml"})

    def test_a_changed_test_runs_itself(self):
        self.assertEqual(touched.script_tests({"scripts/test_neither.py"}), {"scripts/test_neither.py": "scripts/test_neither.py"})

    def test_a_test_no_workflow_runs_is_not_selected_and_is_named_when_it_changed(self):
        self.assertNotIn("scripts/test_orphan.py", touched.script_tests({"scripts/tool.py", "scripts/test_orphan.py"}))
        changed = {"scripts/test_orphan.py", "scripts/test_names.py", "scripts/tool.py", "scripts/test_deleted.py"}
        self.assertEqual(touched.tests_no_workflow_runs(changed), ["scripts/test_orphan.py"])

    def test_a_change_outside_scripts_and_workflows_selects_nothing(self):
        self.assertEqual(touched.script_tests({"docs/tool.py", "decoder/run.sh", "tool.py"}), {})

    def test_a_tree_with_no_workflow_selects_nothing(self):
        import shutil
        shutil.rmtree(os.path.join(self.root, ".github"))
        self.assertEqual(touched.script_tests({"scripts/tool.py", "scripts/test_names.py"}), {})
        self.assertEqual(touched.tests_no_workflow_runs({"scripts/test_names.py"}), ["scripts/test_names.py"])


class ScriptTestsOnThisTree(unittest.TestCase):
    def test_a_day_loop_change_runs_the_tests_that_run_the_day_loop(self):
        # scripts/test_timing_lock.py copies touched.py and touched_scans.py into a scratch repository and runs them, and
        # scripts/test_day_loaders.py runs touched.py: neither is the test file named after the script.
        got = touched.script_tests({"scripts/touched.py"})
        for t in ("scripts/test_touched.py", "scripts/test_timing_lock.py", "scripts/test_day_loaders.py"):
            self.assertIn(t, got)
        got = touched.script_tests({"scripts/touched_scans.py"})
        for t in ("scripts/test_touched.py", "scripts/test_timing_lock.py"):
            self.assertIn(t, got)

    def test_a_lint_step_is_the_lint_lanes_command(self):
        with open(os.path.join(ROOT, ".github", "workflows", "ci.yml")) as f:
            ci = f.read()
        self.assertTrue(touched.LINT_STEPS)
        for _, _, argv in touched.LINT_STEPS:
            self.assertIn("python3 -B " + " ".join(argv), ci, "touched.LINT_STEPS mirrors a lint lane step; change both together")


class RunsScriptTests(_KeepsRAN):
    """run() runs each selected script test the way the lint lane does and counts it as a check; a changed changelog fragment runs
    the lane's render. Spies touched.sh, so nothing is executed."""

    def _run(self, selected, python, changed=("scripts/x.py",), orphans=(), nested=False):
        """run()'s exit code, its output and the commands it ran, with script_tests() giving selected and each python command
        answered by python(argv). self.kw keeps each command's keyword arguments. nested: the run is told a script test started it."""
        import contextlib
        import io
        from unittest import mock
        calls, self.kw = [], {}

        def spy(*a, **k):
            calls.append(a)
            self.kw[a] = k
            if a[:2] == ("git", "diff"):
                return subprocess.CompletedProcess(a, 0, "".join(c + "\n" for c in changed), "")
            if a[0] == sys.executable:
                return python(a)
            return subprocess.CompletedProcess(a, 0, "", "")

        out = io.StringIO()
        with mock.patch.object(touched, "sh", spy), mock.patch.object(touched, "plan", lambda *a: ({}, [])), \
                mock.patch.object(touched, "script_tests", lambda changed: dict(selected)), \
                mock.patch.object(touched, "tests_no_workflow_runs", lambda changed: list(orphans)), \
                mock.patch.object(touched_scans, "notes_changes", lambda root: set()), \
                mock.patch.object(touched_scans, "problems", lambda root: []), \
                mock.patch.dict(os.environ, {"TOUCHED_BASE": "HEAD"}), contextlib.redirect_stdout(out):
            os.environ.pop(touched.NESTED, None)  # the day loop runs this file as one of its script tests
            if nested:
                os.environ[touched.NESTED] = "1"
            rc = touched.run([], {})
        return rc, out.getvalue(), calls

    ONE = {"scripts/test_x.py": "scripts/x.py"}

    def test_a_script_test_that_runs_this_script_does_not_start_them_again(self):
        ok = lambda a: subprocess.CompletedProcess(a, 0, "", ".\nRan 1 test in 0.1s\n\nOK\n")
        test = (sys.executable, "-B", "scripts/test_x.py")
        rc, text, calls = self._run(self.ONE, ok)
        self.assertIn(test, calls, text)
        self.assertEqual(self.kw[test]["env"].get(touched.NESTED), "1", "a script test is told that a day-loop run started it")
        rc, text, calls = self._run(self.ONE, ok, nested=True)
        self.assertNotIn(test, calls, "and a run that is told so runs no script test")
        self.assertEqual(rc, 0, text)

    def test_a_failing_script_test_fails_the_run(self):
        fail = lambda a: subprocess.CompletedProcess(a, 1, "", "F..\nAssertionError: 1 != 0\n\nRan 3 tests in 0.1s\n\nFAILED (failures=1)\n")
        rc, text, calls = self._run(self.ONE, fail)
        self.assertEqual(rc, 1, text)
        self.assertRegex(text, r"FAIL script test scripts/test_x\.py")
        self.assertIn("AssertionError: 1 != 0", text, "the failure's own output is shown")
        self.assertTrue(text.strip().splitlines()[-1].startswith("VERDICT: FAIL — touched: 1 of 1 checks failed"), text)

    def test_a_passing_one_is_a_check_run_as_the_lint_lane_runs_it(self):
        ok = lambda a: subprocess.CompletedProcess(a, 0, "", "...\nRan 3 tests in 0.1s\n\nOK (skipped=2)\n")
        rc, text, calls = self._run(self.ONE, ok)
        self.assertEqual(rc, 0, text)
        self.assertIn((sys.executable, "-B", "scripts/test_x.py"), calls)
        self.assertRegex(text, r"ok\s+script test scripts/test_x\.py \(names scripts/x\.py\): 3 tests, 2 skipped")  # a skip is not a pass
        self.assertTrue(text.strip().splitlines()[-1].startswith("VERDICT: PASS — touched: 1 checks passed"), text)

    def test_one_whose_every_test_was_skipped_is_a_named_skip_not_a_pass(self):
        skipped = lambda a: subprocess.CompletedProcess(a, 0, "", "ss\nRan 2 tests in 0.1s\n\nOK (skipped=2)\n")
        rc, text, _ = self._run(self.ONE, skipped)
        self.assertRegex(text, r"skip script test scripts/test_x\.py \(names scripts/x\.py\): each of its 2 test\(s\) was skipped here")
        self.assertEqual(rc, 2, text)  # nothing judged the change
        self.assertTrue(text.strip().splitlines()[-1].startswith("VERDICT: INCONCLUSIVE"), text)

    def test_one_that_times_out_fails(self):
        def hang(a):
            raise subprocess.TimeoutExpired(a, touched.SCRIPT_TIMEOUT_S)
        rc, text, _ = self._run(self.ONE, hang)
        self.assertEqual(rc, 1, text)
        self.assertIn(f"timed out after {touched.SCRIPT_TIMEOUT_S} s", text)

    def test_a_changelog_fragment_runs_the_lanes_render(self):
        render = (sys.executable, "-B", "scripts/changelog.py", "render")
        ok = lambda a: subprocess.CompletedProcess(a, 0, "- **x**\n", "")
        rc, text, calls = self._run({}, ok, changed=("changelog.d/2026-01-01-x.md",))
        self.assertEqual(rc, 0, text)
        self.assertIn(render, calls)
        bad = lambda a: subprocess.CompletedProcess(a, 1, "", "changelog: 2026-01-01-x.md: a fragment is one `- ` entry ending in a newline\n")
        rc, text, _ = self._run({}, bad, changed=("changelog.d/2026-01-01-x.md",))
        self.assertEqual(rc, 1, text)
        self.assertIn("a fragment is one `- ` entry", text)
        rc, text, calls = self._run({}, ok, changed=("docs/changelog.d/x.md",))
        self.assertNotIn(render, calls, "only the repository's own changelog.d")

    def test_a_go_file_runs_the_lanes_comment_census(self):
        census = (sys.executable, "-B", "scripts/comment_census.py", "tree")
        ok = lambda a: subprocess.CompletedProcess(a, 0, "VERDICT: PASS\n", "")
        for path in ("internal/runtime/mem/workingset.go", "testdata/comment_trackers.ratchet", "scripts/comment_census.py"):
            rc, text, calls = self._run({}, ok, changed=(path,))
            self.assertIn(census, calls, path)
            self.assertIn("ok   comment census (scripts/comment_census.py tree)", text, path)
            if not path.endswith(".go"):  # a Go file also runs its module's checks, which this spy does not answer
                self.assertEqual(rc, 0, text)
        red = lambda a: subprocess.CompletedProcess(a, 1, "  internal/runtime/mem/workingset.go: 1 > 0: 15:B4\nVERDICT: RED\n", "")
        rc, text, _ = self._run({}, red, changed=("scripts/comment_census.py",))
        self.assertEqual(rc, 1, text)
        self.assertIn("workingset.go: 1 > 0: 15:B4", text, "the site is in the output")
        for path in ("decoder/x_test.go", "docs/server.md", "decoder/testdata/x.go", "notes/x.go.txt"):  # what the census does not read
            rc, text, calls = self._run({}, ok, changed=(path,))
            self.assertNotIn(census, calls, path)

    def test_a_changed_test_no_workflow_runs_is_a_named_skip_not_a_pass(self):
        ok = lambda a: subprocess.CompletedProcess(a, 0, "", "")
        rc, text, calls = self._run({}, ok, changed=("scripts/test_orphan.py",), orphans=("scripts/test_orphan.py",))
        self.assertIn("skip script test scripts/test_orphan.py: no workflow runs it", text)
        self.assertFalse([c for c in calls if c[0] == sys.executable], "and the day loop does not run it either")
        self.assertEqual(rc, 2, text)  # nothing judged the change
        self.assertTrue(text.strip().splitlines()[-1].startswith("VERDICT: INCONCLUSIVE"), text)

    def test_no_script_named_is_still_nothing_to_do(self):
        ok = lambda a: subprocess.CompletedProcess(a, 0, "", "")
        rc, text, calls = self._run({}, ok, changed=("docs/x.md",))
        self.assertEqual(rc, 0, text)
        self.assertTrue(text.strip().splitlines()[-1].startswith("VERDICT: n/a — touched: no .go file"), text)


class SiteScopes(unittest.TestCase):
    """The two scopes that select the site's tests, on a scratch tree: PATHS_OF, the paths a workflow's pull_request trigger lists,
    and NAMED_IN, the paths the text of a set of files names."""

    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.root = self._td.name
        _scratch(self.root, {
            ".github/workflows/site.yml": "name: site\n\non:\n  push:\n    branches: [main]\n    paths:\n      - 'pushed/only.md'\n"
                                          "  pull_request:\n    paths:\n      - 'apps/site/**'\n      - 'README.md'\n"
                                          "      - \"docs/book/**\"\n  workflow_dispatch:\n\njobs:\n  build:\n    steps:\n"
                                          "      - run: echo\n        with:\n          paths:\n            - 'not/a/trigger.md'\n",
            "apps/site/different/01-page.md": "---\ntitle: x\nfigures:\n  - {text: \"12\", source: docs/record.md}\nsources:\n"
                                              "  - pkg/source.go\n  - \"quoted/skill.md\"\n  - gonedir\n  - heredir\n---\n\n"
                                              "See [the code](https://github.com/townsendmerino/goinfer/blob/main/linked/code.go).\n",
            "apps/site/data/claims.json": "[{\"path\": \"docs/claimed.md\"}]\n",
            "apps/site/different/notes.txt": "elsewhere/unread.go\n",  # not a page: the glob is *.md
            "heredir/kept.go": "package heredir\n",
        })
        S = touched_scans
        self.scans = [S.Scan("apps/site/internal/site", S.WHOLE, S.PATHS_OF(".github/workflows/site.yml")
                             + S.NAMED_IN("apps/site/different/*.md", "apps/site/data/*.json"))]

    def tearDown(self):
        self._td.cleanup()

    def picks(self, *changed):
        from unittest import mock
        with mock.patch.object(touched_scans, "SCANS", self.scans):
            return touched_scans.select(set(changed), self.root) == {"apps/site/internal/site": [touched_scans.WHOLE]}

    def test_the_paths_a_workflow_runs_pull_requests_for(self):
        for p in ("apps/site/internal/site/site.go", "apps/site/go.mod", "README.md", "docs/book/ch1/intro.md"):
            self.assertTrue(self.picks(p), p)
        for p in ("pushed/only.md", "docs/README.md", "docs/booklet.md", "not/a/trigger.md"):
            self.assertFalse(self.picks(p), p)

    def test_the_paths_the_pages_and_the_data_name(self):
        for p in ("pkg/source.go", "quoted/skill.md", "docs/record.md", "linked/code.go", "docs/claimed.md"):
            self.assertTrue(self.picks(p), p)
        self.assertFalse(self.picks("pkg/other.go", "docs/unclaimed.md", "elsewhere/unread.go"))

    def test_a_named_directory_counts_once_it_is_gone(self):
        # A page lists a directory by its name. A file under it is not named, until the directory itself has left the tree:
        # its files are then in the diff as deletions.
        self.assertTrue(self.picks("gonedir/inside/x.go"))
        self.assertFalse(self.picks("heredir/kept.go"), "the directory is still there")
        self.assertFalse(self.picks("heredir/removed.go"), "a file that left a directory that stayed")

    def test_a_scope_with_nothing_to_read_is_a_problem(self):
        from unittest import mock
        subprocess.run(["git", "-C", self.root, "init", "-q"], check=True, capture_output=True)
        S = touched_scans
        gone = [S.Scan("apps/site/internal/site", S.WHOLE, S.PATHS_OF(".github/workflows/gone.yml") + S.NAMED_IN("apps/site/nowhere/*.md"))]
        with mock.patch.object(S, "SCANS", self.scans + gone):
            got = [p for p in S.problems(self.root) if "its scope" in p]
        self.assertEqual(got, ["apps/site/internal/site (whole package): its scope workflow_paths '.github/workflows/gone.yml' selects no file (moved, renamed or deleted)",
                               "apps/site/internal/site (whole package): its scope named_in 'apps/site/nowhere/*.md' selects no file (moved, renamed or deleted)"])


class SiteTestsOnThisTree(unittest.TestCase):
    """The lint lane runs the site module's tests on every pull request, and they read the rest of the repository. The day loop
    runs them for a change to anything a page or a data file names and to anything site.yml runs for."""

    SITE = ("apps/site", "apps/site/internal/site")

    def whole(self, *changed):
        tests, _ = touched.plan(set(changed))
        return self.SITE in tests and tests[self.SITE] is None

    def test_a_file_a_page_lists_as_a_source_runs_the_sites_tests(self):
        # A source that moves away is in the diff under its old name, which the page still lists.
        with open(os.path.join(ROOT, "apps", "site", "different", "15-upgrade-cant-change-answers.md")) as f:
            front = f.read().split("\n---", 1)[0]
        block = re.search(r"^sources:\n((?:[ \t]+-.*\n?)+)", front, re.M).group(1)
        sources = re.findall(r"^[ \t]+-[ \t]+\"?([^\"\n]+?)\"?[ \t]*$", block, re.M)
        self.assertGreater(len(sources), 2, sources)
        for s in sources:
            self.assertTrue(self.whole(s), s)

    def test_what_site_yml_runs_for_runs_the_sites_tests(self):
        for p in ("apps/site/different/a-new-page.md", "apps/site/internal/site/site.go", "docs/capability-matrix.json",
                  "pull/curated.json", "README.md", "docs/book/a-new-chapter.md"):
            self.assertTrue(self.whole(p), p)
        self.assertFalse(self.whole("decoder/a_file_no_page_names.go", "docs/a-doc-no-page-names.md"))

    def test_the_lint_lane_runs_them_too(self):
        with open(os.path.join(ROOT, ".github", "workflows", "ci.yml")) as f:
            ci = f.read()
        self.assertRegex(ci, r"working-directory: apps/site\n(?:.*\n){0,3}?\s+run: go test -count=1 -v \./\.\.\.",
                         "the day loop mirrors the lint lane's site step; change both together")


if __name__ == "__main__":
    unittest.main()
