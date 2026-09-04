package servecheck

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

// TestRun_zeroModelsReportsSkippedNotFullyPassed pins V-17 (docs/review-2026-09-04.md): against
// a server with zero models loaded, Chat/Structured/Stop/CountTokens never run at all (Run's own
// else-if branch appends a single Skip row instead) — but the summary line used to print "all N
// checks passed" unconditionally, reading as full coverage when only the models-list row and a
// skip actually happened. Same "a SKIP IS NOT A PASS" doctrine this repo already applies to Go
// test output (CLAUDE.md).
func TestRun_zeroModelsReportsSkippedNotFullyPassed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[]}`))
	}))
	defer srv.Close()

	out := captureStdout(t, func() {
		code := Run([]string{srv.URL}, "goinfer")
		if code != 0 {
			t.Errorf("Run returned %d, want 0 — a skip is not a failure", code)
		}
	})

	if strings.Contains(out, "all 2 checks passed") {
		t.Errorf("summary reads as full coverage despite a skipped row:\n%s", out)
	}
	if !strings.Contains(out, "skipped") {
		t.Errorf("summary does not mention the skip at all:\n%s", out)
	}
	// The one row that actually ran (models list) must still be credited.
	if !strings.Contains(out, "1 of 2 checks passed") {
		t.Errorf("summary does not name the real ran/total split:\n%s", out)
	}
}

// captureStdout redirects os.Stdout for the duration of fn and returns what was written.
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	orig := os.Stdout
	os.Stdout = w
	fn()
	w.Close()
	os.Stdout = orig
	out, _ := io.ReadAll(r)
	return string(out)
}
