package fitcmd

import (
	"bytes"
	"os"
	"testing"
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
