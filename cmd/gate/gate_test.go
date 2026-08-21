package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// Acceptance (b): mutation-check both ways, against the REAL toolchain.
//
// These tests build a scratch module, run the gate's own cell runner over it, and assert the
// verdict flips with the defect. A gate that cannot be shown to go RED on the thing it exists to
// catch is not a gate — that is the same "a gate must be able to run, and able to fail" rule the
// skip buckets implement, applied to the runner itself.
// ---------------------------------------------------------------------------

// scratchModule writes a one-package module containing the given test source and returns its dir.
func scratchModule(t *testing.T, src string) string {
	t.Helper()
	dir := t.TempDir()
	write := func(name, body string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module scratch\n\ngo 1.27\n")
	write("scratch_test.go", src)
	return dir
}

const srcRed = `package scratch

import "testing"

func TestGreen(t *testing.T) {}
func TestSkipped(t *testing.T) { t.Skip("no tiny fixture at testdata/x.safetensors") }
func TestBroken(t *testing.T)  { t.Fatalf("parity cosine 0.31, want >= 0.99") }
`

const srcGreen = `package scratch

import "testing"

func TestGreen(t *testing.T) {}
func TestSkipped(t *testing.T) { t.Skip("no tiny fixture at testdata/x.safetensors") }
func TestBroken(t *testing.T)  {}
`

func runScratch(t *testing.T, cfg *gateConfig, dirs ...string) (int, string, *results, []cellResult) {
	t.Helper()
	cfg.Cells = nil
	for i, d := range dirs {
		cfg.Cells = append(cfg.Cells, cell{Name: fmt.Sprintf("cell%d", i), Pkgs: []string{"./..."}, Dir: d})
	}
	res := newResults()
	var cells []cellResult
	for _, c := range cfg.Cells {
		cells = append(cells, runCell(c, cfg, res, t.TempDir()))
	}
	var buf bytes.Buffer
	var rc int
	if cfg.Decision == "census" {
		rc = reportCensus(&buf, cfg, res, false)
	} else {
		rc = reportTally(&buf, cfg, res, cells, provenance{Commit: "test", Date: "-", Host: "-"})
	}
	return rc, buf.String(), res, cells
}

func TestMutation_censusGoesRedOnFailureAndGreenWithout(t *testing.T) {
	rc, out, _, _ := runScratch(t, censusConfig(nil), scratchModule(t, srcRed))
	if rc != 1 {
		t.Fatalf("census on a failing package: rc=%d, want 1\n%s", rc, out)
	}
	if !strings.Contains(out, "TestBroken") {
		t.Errorf("the failing test is not named in the report:\n%s", out)
	}
	// The skip must still be counted and bucketed — a red run must not lose its census.
	if !strings.Contains(out, "SKIP  1") {
		t.Errorf("skip not counted in a red run:\n%s", out)
	}
	if !strings.Contains(out, "missing-fixture  1") {
		t.Errorf("skip not bucketed as missing-fixture:\n%s", out)
	}

	rc, out, _, _ = runScratch(t, censusConfig(nil), scratchModule(t, srcGreen))
	if rc != 0 {
		t.Fatalf("census with the defect removed: rc=%d, want 0\n%s", rc, out)
	}
	if !strings.Contains(out, "PASS  2") {
		t.Errorf("want PASS 2 after the fix:\n%s", out)
	}
}

// TestMutation_tallyIntegrity is THE property `set -e` would have destroyed, and the reason the
// shell gates deliberately omitted it: a failure in the FIRST cell must not stop the second cell
// running, and both counts must survive into the verdict.
func TestMutation_tallyIntegrity(t *testing.T) {
	cfg := heavyConfig()
	cfg.Precondition = nil // no models zoo needed: this tests the tally, not the tier
	red := scratchModule(t, srcRed)
	green := scratchModule(t, srcGreen)

	rc, out, _, cells := runScratch(t, cfg, red, green)
	if rc != 1 {
		t.Fatalf("rc=%d, want 1 (a failing first cell must make the gate red)\n%s", rc, out)
	}
	if len(cells) != 2 {
		t.Fatalf("the second cell did not run — the matrix aborted on the first failure: %d cells", len(cells))
	}
	if cells[1].Pass == 0 {
		t.Errorf("second cell reports 0 passes; its work was lost:\n%s", out)
	}
	// PASSED 3 = 1 from the red cell (TestGreen) + 2 from the green cell. The count must include
	// work done AFTER the failure.
	if !strings.Contains(out, "PASSED 3") || !strings.Contains(out, "FAILED 1") {
		t.Errorf("verdict lost part of the tally; want PASSED 3 / FAILED 1:\n%s", out)
	}
}

// TestMutation_zeroPassIsRed: heavy's rule is that a run in which everything skipped is RED. A
// gate that reports green having executed nothing is the failure mode the whole tier exists to
// prevent — GOINFER_HEAVY_TESTS=1 does not stop a test skipping for a missing checkpoint.
func TestMutation_zeroPassIsRed(t *testing.T) {
	cfg := heavyConfig()
	cfg.Precondition = nil
	src := "package scratch\n\nimport \"testing\"\n\nfunc TestOnlySkips(t *testing.T) { t.Skip(\"no checkpoint\") }\n"
	rc, out, _, _ := runScratch(t, cfg, scratchModule(t, src))
	if rc != 1 {
		t.Fatalf("all-skipped run: rc=%d, want 1\n%s", rc, out)
	}
	if !strings.Contains(out, "nothing actually ran") {
		t.Errorf("the verdict does not say WHY it is red:\n%s", out)
	}
}

// TestMutation_hiddenFailure: a package that dies without emitting a per-test failure — a panic in
// a goroutine, a fatal error, a timeout — must still be RED. Counting only `--- FAIL:` lines
// reports GREEN on a crashed package.
func TestMutation_hiddenFailure(t *testing.T) {
	cfg := heavyConfig()
	cfg.Precondition = nil
	src := `package scratch

import (
	"os"
	"testing"
)

func TestExitsHard(t *testing.T) { os.Exit(3) }
`
	rc, out, _, cells := runScratch(t, cfg, scratchModule(t, src))
	if rc != 1 {
		t.Fatalf("hard-exiting package: rc=%d, want 1\n%s", rc, out)
	}
	if !cells[0].Hidden {
		t.Errorf("the failure was not recognised as hidden (rc=%d):\n%s", cells[0].RC, out)
	}
	if !strings.Contains(out, "HIDDEN failure") {
		t.Errorf("the report does not name the hidden failure:\n%s", out)
	}
}

// ---------------------------------------------------------------------------
// Stream-level behaviour: parity with what skip_census.py computed.
// ---------------------------------------------------------------------------

func consumeLines(lines ...string) *results {
	r := newResults()
	r.consume(strings.NewReader(strings.Join(lines, "\n")))
	return r
}

func ev(action, pkg, test, out string) string {
	return fmt.Sprintf(`{"Action":%q,"Package":%q,"Test":%q,"Output":%q}`, action, pkg, test, out)
}

func TestCensus_emptyStreamIsNotAPass(t *testing.T) {
	cfg := censusConfig(nil)
	var buf bytes.Buffer
	rc := reportCensus(&buf, cfg, newResults(), false)
	if rc != 1 {
		t.Fatalf("empty stream: rc=%d, want 1", rc)
	}
	if !strings.Contains(buf.String(), "NO TESTS OBSERVED") {
		t.Errorf("empty stream did not say it observed nothing:\n%s", buf.String())
	}
}

// A package-level failure exits 0 in skip_census.py. E8 changes the substrate, not the verdict
// (acceptance a) — so it still exits 0 here, but it must be ANNOUNCED, because a suppressed
// failure nobody is told about is indistinguishable from no failure.
func TestCensus_packageLevelFailIsAnnouncedNotCounted(t *testing.T) {
	r := consumeLines(
		ev("output", "scratch", "TestA", "=== RUN   TestA\n"),
		ev("pass", "scratch", "TestA", ""),
		ev("output", "scratch", "", "panic: runtime error [recovered]\n"),
		ev("fail", "scratch", "", ""),
	)
	var buf bytes.Buffer
	rc := reportCensus(&buf, censusConfig(nil), r, false)
	out := buf.String()
	if rc != 0 {
		t.Fatalf("rc=%d, want 0 (parity with skip_census.py)", rc)
	}
	if !strings.Contains(out, "PACKAGE-LEVEL FAILS") || !strings.Contains(out, "NOT counted in this verdict") {
		t.Errorf("the suppressed failure was not announced:\n%s", out)
	}
	if !strings.Contains(out, "NATIVE CRASH") {
		t.Errorf("panic output was not recognised as a crash:\n%s", out)
	}
}

// The two migrated scripts disagreed on subtests: heavy_gate anchored its grep at column 0 (top
// level only), skip_census keyed on (Package, Test) from the JSON (subtests included). Each gate
// must reproduce its OWN number.
func TestTopLevelOnly_matchesEachScriptsCounting(t *testing.T) {
	lines := []string{
		ev("output", "p", "TestA", "=== RUN   TestA\n"),
		ev("pass", "p", "TestA", ""),
		ev("output", "p", "TestA/one", "    === RUN   TestA/one\n"),
		ev("pass", "p", "TestA/one", ""),
		ev("output", "p", "TestA/two", "    === RUN   TestA/two\n"),
		ev("pass", "p", "TestA/two", ""),
	}
	r := consumeLines(lines...)

	var buf bytes.Buffer
	reportCensus(&buf, censusConfig(nil), r, false)
	if !strings.Contains(buf.String(), "PASS  3") {
		t.Errorf("census must count subtests (skip_census.py did):\n%s", buf.String())
	}

	cfg := heavyConfig()
	cfg.Precondition = nil
	var buf2 bytes.Buffer
	reportTally(&buf2, cfg, r, []cellResult{{Cell: cell{Name: "p"}, Pass: 1}}, provenance{})
	if !strings.Contains(buf2.String(), "PASSED 1") {
		t.Errorf("heavy must count top-level tests only (heavy_gate.sh anchored at column 0):\n%s", buf2.String())
	}
}

// The skip-reason extractor: the LAST non-marker line, so a test that logged before skipping still
// reports the skip message rather than the log.
func TestSkipReason_takesTheLastNonMarkerLine(t *testing.T) {
	got := skipReason([]string{
		"=== RUN   TestX\n",
		"    x_test.go:10: probing ~/models\n",
		"    x_test.go:12: no tiny fixture at testdata/x.safetensors\n",
		"--- SKIP: TestX (0.00s)\n",
	})
	want := "x_test.go:12: no tiny fixture at testdata/x.safetensors"
	if got != want {
		t.Fatalf("skipReason = %q, want %q", got, want)
	}
}

// Bucket rules ported from skip_census.py, INCLUDING their order (first match wins). A rule
// reordering silently reclassifies skips, which changes what GOINFER_REQUIRE_FIXTURES blocks on.
func TestClassifySkip_parityWithPythonRules(t *testing.T) {
	cases := []struct{ reason, want string }{
		{"x_test.go:9: GOINFER_HEAVY_TESTS unset — real 35B decode", "heavy-model"},
		{"set GOINFER_SERVE_MODEL to run the live-server soak", "integration-env"},
		{"no golden recorded yet; run scripts/pin_qwen3_5_forward.py", "missing-golden"},
		{"no tiny fixture at testdata/qwen3_5/model.safetensors", "missing-fixture"},
		{"no CUDA device present", "no-gpu-device"},
		{"waiting on nothing in particular", "other"},
		// Order matters: this one matches BOTH heavy-model and missing-fixture, and heavy wins.
		{"no real checkpoint under ~/models", "heavy-model"},
	}
	for _, c := range cases {
		if got := classifySkip(c.reason); got != c.want {
			t.Errorf("classifySkip(%q) = %q, want %q", c.reason, got, c.want)
		}
	}
}

// Long output lines must not truncate the stream. bufio.Scanner's default 64 KiB token limit would
// turn one parity dump into a scan error and drop every event after it — a silent undercount, which
// is worse than a crash because the report still looks like a report.
func TestConsume_survivesVeryLongOutputLines(t *testing.T) {
	huge := strings.Repeat("x", 300*1024)
	r := consumeLines(
		ev("output", "p", "TestBig", huge),
		ev("pass", "p", "TestBig", ""),
		ev("output", "p", "TestAfter", "=== RUN   TestAfter\n"),
		ev("pass", "p", "TestAfter", ""),
	)
	if len(r.final) != 2 {
		t.Fatalf("events after a 300 KiB line were dropped: got %d tests, want 2", len(r.final))
	}
}

// Refusal is a third outcome, not a flavour of red: with no models dir, "green" and "red" are both
// lies. heavy_gate.sh exited 2; so does this.
func TestHeavy_refusesWithoutModelsDir(t *testing.T) {
	t.Setenv("GOINFER_GATE_MODELS", filepath.Join(t.TempDir(), "definitely-not-here"))
	var buf bytes.Buffer
	if rc := run([]string{"heavy"}, &buf); rc != 2 {
		t.Fatalf("rc=%d, want 2 (refused)\n%s", rc, buf.String())
	}
	if !strings.Contains(buf.String(), "REFUSED") {
		t.Errorf("refusal not stated:\n%s", buf.String())
	}
}
