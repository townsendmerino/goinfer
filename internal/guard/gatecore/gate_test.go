package gatecore

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/testgate"
)

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

func runScratch(t *testing.T, cfg *GateConfig, dirs ...string) (int, string, *Results, []CellResult) {
	t.Helper()
	cfg.Cells = nil
	for i, d := range dirs {
		cfg.Cells = append(cfg.Cells, Cell{Name: fmt.Sprintf("cell%d", i), Pkgs: []string{"./..."}, Dir: d})
	}
	res := NewResults()
	var cells []CellResult
	for _, c := range cfg.Cells {
		cells = append(cells, RunCell(c, cfg, res, t.TempDir()))
	}
	var buf bytes.Buffer
	var rc int
	if cfg.Decision == "census" {
		rc = ReportCensus(&buf, cfg, res, false)
	} else {
		rc = ReportTally(&buf, cfg, res, cells, Provenance{Commit: "test", Date: "-", Host: "-"})
	}
	return rc, buf.String(), res, cells
}

func TestMutation_censusGoesRedOnFailureAndGreenWithout(t *testing.T) {
	rc, out, _, _ := runScratch(t, CensusConfig(nil), scratchModule(t, srcRed))
	if rc != 1 {
		t.Fatalf("census on a failing package: rc=%d, want 1\n%s", rc, out)
	}
	if !strings.Contains(out, "TestBroken") {
		t.Errorf("the failing test is not named in the report:\n%s", out)
	}
	if !strings.Contains(out, "SKIP  1") {
		t.Errorf("skip not counted in a red run:\n%s", out)
	}
	if !strings.Contains(out, "missing-fixture  1") {
		t.Errorf("skip not bucketed as missing-fixture:\n%s", out)
	}

	rc, out, _, _ = runScratch(t, CensusConfig(nil), scratchModule(t, srcGreen))
	if rc != 0 {
		t.Fatalf("census with the defect removed: rc=%d, want 0\n%s", rc, out)
	}
	if !strings.Contains(out, "PASS  2") {
		t.Errorf("want PASS 2 after the fix:\n%s", out)
	}
}

func TestMutation_tallyIntegrity(t *testing.T) {
	cfg := HeavyConfig()
	cfg.Precondition = nil
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
	if !strings.Contains(out, "PASSED 3") || !strings.Contains(out, "FAILED 1") {
		t.Errorf("verdict lost part of the tally; want PASSED 3 / FAILED 1:\n%s", out)
	}
}

func TestMutation_zeroPassIsRed(t *testing.T) {
	cfg := HeavyConfig()
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

func TestMutation_hiddenFailure(t *testing.T) {
	cfg := HeavyConfig()
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

func consumeLines(lines ...string) *Results {
	r := NewResults()
	r.Consume(strings.NewReader(strings.Join(lines, "\n")))
	return r
}

func ev(action, pkg, test, out string) string {
	return fmt.Sprintf(`{"Action":%q,"Package":%q,"Test":%q,"Output":%q}`, action, pkg, test, out)
}

func TestCensus_emptyStreamIsNotAPass(t *testing.T) {
	cfg := CensusConfig(nil)
	var buf bytes.Buffer
	rc := ReportCensus(&buf, cfg, NewResults(), false)
	if rc != 1 {
		t.Fatalf("empty stream: rc=%d, want 1", rc)
	}
	if !strings.Contains(buf.String(), "NO TESTS OBSERVED") {
		t.Errorf("empty stream did not say it observed nothing:\n%s", buf.String())
	}
}

func TestCensus_packageLevelFailIsAnnouncedNotCounted(t *testing.T) {
	r := consumeLines(
		ev("output", "scratch", "TestA", "=== RUN   TestA\n"),
		ev("pass", "scratch", "TestA", ""),
		ev("output", "scratch", "", "panic: runtime error [recovered]\n"),
		ev("fail", "scratch", "", ""),
	)
	var buf bytes.Buffer
	rc := ReportCensus(&buf, CensusConfig(nil), r, false)
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
	ReportCensus(&buf, CensusConfig(nil), r, false)
	if !strings.Contains(buf.String(), "PASS  3") {
		t.Errorf("census must count subtests:\n%s", buf.String())
	}

	cfg := HeavyConfig()
	cfg.Precondition = nil
	var buf2 bytes.Buffer
	ReportTally(&buf2, cfg, r, []CellResult{{Cell: Cell{Name: "p"}, Pass: 1}}, Provenance{})
	if !strings.Contains(buf2.String(), "PASSED 1") {
		t.Errorf("heavy must count top-level tests only:\n%s", buf2.String())
	}
}

func TestSkipReason_takesTheLastNonMarkerLine(t *testing.T) {
	got := SkipReason([]string{
		"=== RUN   TestX\n",
		"    x_test.go:10: probing ~/models\n",
		"    x_test.go:12: no tiny fixture at testdata/x.safetensors\n",
		"--- SKIP: TestX (0.00s)\n",
	})
	want := "x_test.go:12: no tiny fixture at testdata/x.safetensors"
	if got != want {
		t.Fatalf("SkipReason = %q, want %q", got, want)
	}
}

func TestClassifySkip_parityWithPythonRules(t *testing.T) {
	cases := []struct{ reason, want string }{
		{"x_test.go:9: GOINFER_HEAVY_TESTS unset — real 35B decode", "heavy-model"},
		{"set GOINFER_SERVE_MODEL to run the live-server soak", "integration-env"},
		{"no golden recorded yet; run scripts/pin_qwen3_5_forward.py", "missing-golden"},
		{"no tiny fixture at testdata/qwen3_5/model.safetensors", "missing-fixture"},
		{"no CUDA device present", "no-gpu-device"},
		{"waiting on nothing in particular", "other"},
		{"no real checkpoint under ~/models", "heavy-model"},
	}
	for _, c := range cases {
		if got := ClassifySkip(c.reason); got != c.want {
			t.Errorf("ClassifySkip(%q) = %q, want %q", c.reason, got, c.want)
		}
	}
}

func TestClassifySkip_tokensFirst(t *testing.T) {
	cases := []struct{ reason, want string }{
		{"x_test.go:9: " + testgate.SkipHeavy + " heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in", "heavy-model"},
		{"x_test.go:9: " + testgate.SkipMeasure + " a measurement, not a guard: set GOINFER_MEASURE_TESTS=1 to run it", "integration-env"},
		{"x_test.go:9: " + testgate.SkipDevice + " no cuda device: cuInit: CUDA_ERROR_NO_DEVICE", "no-gpu-device"},
		{"x_test.go:9: " + testgate.SkipDevice + " no metal device: real checkpoint absent", "no-gpu-device"},
		{"x_test.go:9: " + testgate.SkipAsset + " gemma-heavy.gguf not found under the models directory", "missing-fixture"},
		{"x_test.go:9: " + testgate.SkipAsset + " webgpu-device.bin not found", "missing-fixture"},
		{"x_test.go:9: cuInit: no device", "no-gpu-device"},
	}
	for _, c := range cases {
		if got := ClassifySkip(c.reason); got != c.want {
			t.Errorf("ClassifySkip(%q) = %q, want %q", c.reason, got, c.want)
		}
	}
}

func TestConsume_survivesVeryLongOutputLines(t *testing.T) {
	huge := strings.Repeat("x", 300*1024)
	r := consumeLines(
		ev("output", "p", "TestBig", huge),
		ev("pass", "p", "TestBig", ""),
		ev("output", "p", "TestAfter", "=== RUN   TestAfter\n"),
		ev("pass", "p", "TestAfter", ""),
	)
	if len(r.Final) != 2 {
		t.Fatalf("events after a large line were dropped: got %d tests, want 2", len(r.Final))
	}
}

func TestCellEnv_dropsAnAmbientMeasureOptIn(t *testing.T) {
	base := []string{"PATH=/bin", "GOINFER_MEASURE_TESTS=1", "GOINFER_HEAVY_TESTS=1"}
	got := CellEnv(base, map[string]string{"CGO_ENABLED": "0"})
	for _, kv := range got {
		if strings.HasPrefix(kv, "GOINFER_MEASURE_TESTS=") {
			t.Fatalf("CellEnv kept the ambient %q: %v", kv, got)
		}
	}
	if !slices.Contains(got, "PATH=/bin") || !slices.Contains(got, "GOINFER_HEAVY_TESTS=1") || !slices.Contains(got, "CGO_ENABLED=0") {
		t.Fatalf("CellEnv lost a variable it should keep: %v", got)
	}
	got = CellEnv(base, map[string]string{"GOINFER_MEASURE_TESTS": "1"})
	if !slices.Contains(got, "GOINFER_MEASURE_TESTS=1") {
		t.Fatalf("a cell that sets GOINFER_MEASURE_TESTS itself must keep it: %v", got)
	}
}
