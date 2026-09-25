package fitcmd

import (
	"bytes"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/giw"
)

// captureStdout swaps os.Stdout for the duration of fn and returns what was written — Run prints
// its report there (stderr stays for load progress/errors, matching every other subcommand in
// this repo, e.g. internal/pullcmd).
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	orig := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	os.Stdout = w
	fn()
	w.Close()
	os.Stdout = orig
	var buf bytes.Buffer
	if _, err := buf.ReadFrom(r); err != nil {
		t.Fatalf("read pipe: %v", err)
	}
	return buf.String()
}

// TestFreeBytesFor_cpuUsesThePassedInValue is the M-19 gate (docs/audit-2026-09-10.md):
// freeBytesFor's "cpu" branch must report EXACTLY the caller-supplied hostFreeBeforeLoad, proving
// it no longer re-queries HostRAMAvailableBytes() itself after Run has already loaded the model
// into that same RAM budget. A non-positive value must report unknown, matching the old
// HostRAMAvailableBytes()-returns-0-means-unknown convention.
func TestFreeBytesFor_cpuUsesThePassedInValue(t *testing.T) {
	const want = 123456789
	if got, ok := freeBytesFor("cpu", want); !ok || got != want {
		t.Errorf("freeBytesFor(\"cpu\", %d) = (%d, %v), want (%d, true)", want, got, ok, want)
	}
	if got, ok := freeBytesFor("cpu", 0); ok || got != 0 {
		t.Errorf("freeBytesFor(\"cpu\", 0) = (%d, %v), want (0, false)", got, ok)
	}
}

func TestRun_noPathIsUsageError(t *testing.T) {
	if code := Run(nil); code != 2 {
		t.Errorf("Run(nil) = %d, want 2 (usage error)", code)
	}
	if code := Run([]string{"-ctx", "4096"}); code != 2 {
		t.Errorf("Run with only flags, no path = %d, want 2", code)
	}
}

func TestRun_missingCheckpointFails(t *testing.T) {
	if code := Run([]string{"/no/such/checkpoint/dir"}); code != 1 {
		t.Errorf("Run(missing path) = %d, want 1 (load error)", code)
	}
}

// TestRun_realFixtureReportsCPU is the end-to-end smoke test: testdata/llama-tiny is TRACKED in
// git, so this runs in CI unconditionally (same reasoning decoder/fitplan_test.go's dense row
// uses). Asserts Run succeeds and its stdout report at minimum names the "cpu" backend and a
// placement — CompiledBackends() always includes "cpu", so this must hold in every build,
// GPU-tagged or not.
func TestRun_realFixtureReportsCPU(t *testing.T) {
	var code int
	out := captureStdout(t, func() {
		code = Run([]string{"../../testdata/llama-tiny", "-ctx", "512"})
	})
	if code != 0 {
		t.Fatalf("Run = %d, want 0; output:\n%s", code, out)
	}
	if !bytes.Contains([]byte(out), []byte("cpu")) {
		t.Errorf("output does not mention the cpu backend at all:\n%s", out)
	}
	for _, want := range []string{"RESIDENT", "EXPERT-CACHED", "WEIGHT-PAGED", "DECLINE"} {
		if bytes.Contains([]byte(out), []byte(want)) {
			return // found a real placement word — Plan ran and Run printed its result
		}
	}
	t.Errorf("output names no placement (RESIDENT/EXPERT-CACHED/WEIGHT-PAGED/DECLINE):\n%s", out)
}

// TestRun_pinnedCtxThatCannotFitDeclinesNotShrinks exercises the CtxPinned wiring specifically:
// an absurd -ctx must be refused (or fall to weight-paged on cpu), never silently reduced — the
// printed ctx in the header line must stay the one the user asked for.
func TestRun_pinnedCtxThatCannotFitDeclinesNotShrinks(t *testing.T) {
	var code int
	out := captureStdout(t, func() {
		code = Run([]string{"../../testdata/llama-tiny", "-ctx", "999999999"})
	})
	if code != 0 {
		t.Fatalf("Run = %d, want 0 (a bad plan is reported, not a process failure); output:\n%s", code, out)
	}
	if !bytes.Contains([]byte(out), []byte("ctx=999999999")) {
		t.Errorf("header does not echo the pinned ctx unchanged:\n%s", out)
	}
	if bytes.Contains([]byte(out), []byte("RESIDENT")) {
		t.Errorf("an impossible pinned ctx must not report RESIDENT anywhere:\n%s", out)
	}
}

// TestRun_measureSelfMeasuresOnAdmittedBackend is tasks/task-fit-to-hardware.md §5's self-measure
// option: on the tracked llama-tiny fixture (CPU-only in this untagged build, so "cpu" is the
// only admitted backend and therefore the one selfMeasure picks), -measure must actually load
// and decode, printing a real "(measure)" line with a positive rate — not just echo the dry-run
// report above it.
func TestRun_measureSelfMeasuresOnAdmittedBackend(t *testing.T) {
	var code int
	out := captureStdout(t, func() {
		code = Run([]string{"../../testdata/llama-tiny", "-ctx", "512", "-measure"})
	})
	if code != 0 {
		t.Fatalf("Run = %d, want 0; output:\n%s", code, out)
	}
	if !bytes.Contains([]byte(out), []byte("(measure) cpu")) {
		t.Fatalf("output has no '(measure) cpu' line — the probe did not run or picked a different backend:\n%s", out)
	}
	if !bytes.Contains([]byte(out), []byte("tok/s")) {
		t.Errorf("measure line has no tok/s rate:\n%s", out)
	}
}

// TestRun_measureOffByDefault confirms -measure is opt-in: without it, Run must never print a
// "(measure)" line — the probe is a real extra load+decode, not something every dry run should
// pay for silently (tasks/task-fit-to-hardware.md §0: "explicit flags stay, and win").
func TestRun_measureOffByDefault(t *testing.T) {
	out := captureStdout(t, func() {
		Run([]string{"../../testdata/llama-tiny", "-ctx", "512"})
	})
	if bytes.Contains([]byte(out), []byte("(measure)")) {
		t.Errorf("output contains a '(measure)' line without -measure being passed:\n%s", out)
	}
}

// TestRun_measureRateExcludesPrefill is M-20's behavioral half (docs/audit-2026-09-10.md):
// selfMeasure's printed rate used to be n/(prefill+decode) — roughly a third of the true decode
// rate on the CPU staged path (the audit's own measurement) — because the clock started before
// Generate was even called. It now starts after the FIRST token arrives, so the printed line must
// say so explicitly (not the old "includes a N-token prompt prefill" phrasing, which described
// the bug rather than a fix for it) and must report decode steps counted as probeDecode-1, not
// probeDecode — proving the first step was excluded from the denominator, not just relabeled.
func TestRun_measureRateExcludesPrefill(t *testing.T) {
	out := captureStdout(t, func() {
		Run([]string{"../../testdata/llama-tiny", "-ctx", "512", "-measure"})
	})
	if !bytes.Contains([]byte(out), []byte("after the first token")) {
		t.Fatalf("measure line does not say it excludes the first token/prefill:\n%s", out)
	}
	if bytes.Contains([]byte(out), []byte("includes a")) {
		t.Errorf("measure line still uses the old 'includes a N-token prompt prefill' phrasing "+
			"(M-20 not actually fixed, just relabeled):\n%s", out)
	}
	// probeDecode is 32 (fit.go); the printed decode-step count must be 31 (32-1), proving the
	// first token's step was excluded from the count, not just from the label.
	if !bytes.Contains([]byte(out), []byte(" 31 decode steps ")) {
		t.Errorf("measure line does not report 31 decode steps (probeDecode=32 minus the excluded "+
			"first token) — the denominator may still include the first step:\n%s", out)
	}
}

// TestRun_closesFirstLoadBeforeMeasuring is M-20's other half: selfMeasure loads the SAME
// checkpoint again, so Run must close its own first Load BEFORE calling selfMeasure — leaving it
// open the whole time means two full quantized copies resident simultaneously on the CPU backend,
// which can itself trip the memory guard the probe exists to measure honestly. The interruption
// that matters (a real large checkpoint tripping the guard) can't be reproduced with this repo's
// tiny fixtures, so this is asserted structurally instead, the same technique
// TestTranscode_writesViaTempThenRenames (internal/prequant) uses for an analogous ordering
// property: parse Run's AST and confirm the statement calling m.Close() appears BEFORE the `if
// *measure` block that calls selfMeasure, not after.
func TestRun_closesFirstLoadBeforeMeasuring(t *testing.T) {
	fset := token.NewFileSet()
	af, err := parser.ParseFile(fset, "fit.go", nil, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	var fn *ast.FuncDecl
	ast.Inspect(af, func(n ast.Node) bool {
		if d, ok := n.(*ast.FuncDecl); ok && d.Name.Name == "Run" {
			fn = d
		}
		return true
	})
	if fn == nil {
		t.Fatal("Run not found — this guard is watching nothing")
	}
	render := func(e ast.Expr) string {
		var b strings.Builder
		_ = printer.Fprint(&b, fset, e)
		return b.String()
	}
	var closeCallIdx, measureIfIdx = -1, -1
	for i, stmt := range fn.Body.List {
		switch s := stmt.(type) {
		case *ast.ExprStmt:
			if call, ok := s.X.(*ast.CallExpr); ok {
				if sel, ok := call.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "Close" && render(sel.X) == "m" {
					closeCallIdx = i
				}
			}
		case *ast.AssignStmt:
			for _, rhs := range s.Rhs {
				if call, ok := rhs.(*ast.CallExpr); ok {
					if sel, ok := call.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "Close" && render(sel.X) == "m" {
						closeCallIdx = i
					}
				}
			}
		case *ast.IfStmt:
			ast.Inspect(s, func(n ast.Node) bool {
				if call, ok := n.(*ast.CallExpr); ok {
					if id, ok := call.Fun.(*ast.Ident); ok && id.Name == "selfMeasure" {
						measureIfIdx = i
					}
				}
				return true
			})
		}
	}
	if closeCallIdx == -1 {
		t.Fatal("no m.Close() call found in Run — this guard is watching the wrong function")
	}
	if measureIfIdx == -1 {
		t.Fatal("no selfMeasure call found in Run — this guard is watching the wrong function")
	}
	if closeCallIdx >= measureIfIdx {
		t.Errorf("m.Close() (statement %d) does not precede the selfMeasure call (statement %d) — "+
			"selfMeasure's own fresh Load can run while the first copy is still resident (M-20)",
			closeCallIdx, measureIfIdx)
	}
}

// TestFreshSidecar_findsTheOneChatAndServeBuild: a default chat or serve load (--backend cpu) writes this
// host's cpu target, <base>.int4.cpu-amd64.giw or .cpu-arm64.giw. fit used to look only for the canonical
// target, so it never found that sidecar and did a full direct load of the .gguf instead (measured on the
// 0.5B: 2.98 s and 1.0 GB RSS, against a mapped read through the sidecar).
func TestFreshSidecar_findsTheOneChatAndServeBuild(t *testing.T) {
	raw, err := os.ReadFile("../../testdata/glm-tiny.gguf")
	if err != nil {
		t.Skipf("tiny fixture: %v", err)
	}
	dir := t.TempDir()
	src := filepath.Join(dir, "tiny.gguf")
	if err := os.WriteFile(src, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-time.Hour)
	if err := os.Chtimes(src, old, old); err != nil {
		t.Fatal(err)
	}
	if _, _, ok := freshSidecar(src, "int4"); ok {
		t.Fatal("found a sidecar before one was built")
	}
	// Backend "" keeps canonical int4 bytes in RAM, which the .giw writer needs to emit any target (the
	// production transcode does the same); a Backend "cpu" load on arm64 keeps only the row4 repack.
	m, err := decoder.Load(src, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	target := decoder.GIWTargetForBackend("cpu")
	blob, err := decoder.SerializeWeightsForTarget(m.Weights(), "tiny", target)
	m.Close()
	if err != nil {
		t.Fatal(err)
	}
	cpuSidecar := filepath.Join(dir, "tiny.int4."+string(target)+".giw")
	if target == decoder.GIWTargetNone {
		cpuSidecar = filepath.Join(dir, "tiny.int4.canonical.giw") // a GOARCH with no cpu-specific target
	}
	if err := os.WriteFile(cpuSidecar, giw.Write(blob, nil), 0o644); err != nil {
		t.Fatal(err)
	}
	got, be, ok := freshSidecar(src, "int4")
	if !ok || got != cpuSidecar {
		t.Fatalf("freshSidecar = %q, %v; want the cpu-target sidecar %s a default chat/serve load builds", got, ok, filepath.Base(cpuSidecar))
	}
	if target != decoder.GIWTargetNone && be != "cpu" {
		t.Errorf("backend = %q, want cpu: a cpu-target sidecar can be row4-only, which only the literal cpu backend reads", be)
	}
	if _, err := decoder.Load(got, decoder.Options{Quant: "int4", Backend: be}); err != nil {
		t.Errorf("the found sidecar does not load with the backend freshSidecar chose: %v", err)
	}
}
