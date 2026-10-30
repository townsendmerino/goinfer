#!/usr/bin/env python3
"""touched.py runs the source-scan tests (scripts/touched_scans.py) for the files they read, and the table covers every test
that reads Go source. Run: python3 -B scripts/test_touched.py"""
import importlib.util
import os
import re
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
        line, rc = touched.summary(0, self.RAN, [])
        self.assertTrue(line.startswith("VERDICT: PASS — touched: 5 checks passed"), line)
        self.assertEqual(rc, 0)

    def test_red_is_fail(self):
        line, rc = touched.summary(2, self.RAN, [])
        self.assertTrue(line.startswith("VERDICT: FAIL — touched: 2 of 5 checks failed"), line)
        self.assertEqual(rc, 1)

    def test_skips_are_named(self):
        line, _ = touched.summary(0, self.RAN[:3], ["cuda", "test decoder"])
        self.assertTrue(line.startswith("VERDICT: PASS"))
        self.assertIn("did not run: cuda, test decoder", line)

    def test_everything_skipped_is_inconclusive(self):
        """Only gofmt ran, every module was skipped (cuda outside the Mac's go.work, say): nothing was judged."""
        for ran in (["gofmt"], []):
            line, rc = touched.summary(0, ran, ["cuda"])
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

    def test_gguf_go_selects_the_case_label_scan(self):
        self.assertIn(("decoder", "TestGGUFConfig_everyArchitectureReadsMaxPositions"), selected(["decoder/gguf.go"]))

    def test_a_scanner_in_an_untouched_package_runs(self):
        self.assertIn(("internal/cliutil", "TestLdflagsTargetsExist"), selected(["internal/serveapp/openai.go"]))

    def test_a_kernel_source_change_selects_its_readers_with_no_go_file_changed(self):
        got = selected(["cuda/moe.cu"])
        self.assertIn(("cuda", "TestKernelFMALint"), got)
        self.assertIn(("decoder", "TestResidentMoECap_pinsKernelConstant"), got)

    def test_a_doc_a_test_reads_selects_it(self):
        self.assertIn(("decoder", "TestEnvVars_docAndCodeAgree"), selected(["docs/env-vars.md"]))

    def test_a_new_model_fixture_selects_the_fixture_censuses(self):
        # 1ea85681: a new fixture directory was in neither census, and touched.py ran neither, so CI went red.
        got = selected(["testdata/gemma4-moe-kv-tiny32/config.json", "testdata/gemma4-moe-kv-tiny32/model.safetensors"])
        self.assertIn(("decoder", "TestSerializeCensus_everyFixtureIsListedOrExcluded"), got)
        self.assertIn(("decoder", "TestSeams_subCaptureRefusedWhereGateSCannotRun"), got)

    def test_a_ci_workflow_change_selects_its_readers(self):
        got = selected([".github/workflows/ci.yml"])
        for t in [("internal/sampling", "TestSweepCoverage_fullSweepRunsSomewhere"), ("multimodal", "TestRaceSubset_runsInFullSomewhere"),
                  ("internal/kvmgr", "TestOracle_runsInFullSomewhere"), ("chat", "TestReleasing_standaloneBuildGateExists")]:
            self.assertIn(t, got)

    def test_an_unscanned_change_selects_nothing_extra(self):
        self.assertEqual(selected(["README.md"]), set())


# Patterns that mark a function body as reading source from disk. A helper that matches marks every test in its package that
# calls it, so a scanner that reads through a helper is caught too.
READS = re.compile(r'parser\.Parse(File|Dir)\(|"\*\.go"|"\*\.cu"|\.go"\)|\.cu"\)|HasSuffix\([^)]*"\.(go|cu)"\)|'
                   r'"_test\.go"|go/ast|token\.NewFileSet')

# Test functions the patterns flag that read no repository source from disk: each parses an in-memory string, or reads only
# go.mod or fixtures.
NOT_SOURCE_READERS = {
    "TestCommentsOnly_gitCommitGreenAndRed": "a synthetic repo in a temp dir",
    "TestCommentsOnly_worktreeMode": "a synthetic repo in a temp dir",
    "TestIdentityDumperTemplate_isGoAndHasItsMarkers": "parses the embedded dumper template",
    "TestLedger_classifyOutcomes": "a scratch tree (ledgerScratch)",
    "TestLedger_sourceKeyMatchesThePythonImplementation": "a scratch tree (ledgerScratch)",
    "TestMutation_sedInvocationIsPortable": "sed arguments only",
    "TestQuick_deferCells_checkpointCell": "a synthetic package (ckFixture)",
    "TestQuick_deferCells_slowOfAnUntouchedDeviceModule": "a synthetic package (ckFixture)",
    "TestScanTree_skipsIgnoredFiles": "two synthetic files in a temp dir",
    "TestQuick_checkpointLoadersGetTheirOwnCell": "a synthetic package (ckFixture)",
    "TestQuick_smallCheckpointPackageRunsWholeInTheLane": "a synthetic package (ckFixture)",
    "TestRowGofmtAndLint": "path strings in a synthetic report",
    "TestRepoGroup_plants": "file names in a synthetic CI step's gofmt output",
    "TestGoCodeTokens_commentsOnly": "parses in-memory sources",
    "TestParent": "a synthetic package in a temp dir",
    "TestSettleCalls_callWrittenInsideReasoning": "a file name inside a tool call's arguments",
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
    flagged = {}
    for dirpath, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "testdata", "node_modules")]
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


class TableIsComplete(unittest.TestCase):
    def test_every_source_reading_test_is_listed_or_excluded(self):
        self.maxDiff = None
        self.assertEqual(unlisted(), {}, "tests that read source and are in neither touched_scans.SCANS nor EXCLUDED "
                         "(add each to SCANS with the paths it reads, or to EXCLUDED or NOT_SOURCE_READERS with why)")

    def test_every_listed_test_exists(self):
        names = set()
        for dirpath, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in (".git", "node_modules")]
            for f in files:
                if f.endswith("_test.go"):
                    names |= {n for n, _ in functions(os.path.join(dirpath, f))}
        missing = sorted(s.test for s in touched_scans.SCANS if s.test not in names)
        self.assertEqual(missing, [], "SCANS names tests that do not exist (renamed or deleted)")


if __name__ == "__main__":
    unittest.main()
