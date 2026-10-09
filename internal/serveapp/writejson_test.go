package serveapp

import (
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// writeJSON used to send the status line and then stream the encoder's output with its error dropped, so a body JSON cannot carry (NaN, an infinity) reached the client as
// HTTP 200 with an EMPTY body: S6's served check on the 35B read "no answer" for a night (2026-10-09). An unencodable body must be a 500 that names the problem, and an ordinary body
// must be byte-for-byte what it was (200, the JSON and its trailing newline).
func TestWriteJSON_unencodableBodyIsAServerError(t *testing.T) {
	for _, bad := range []float64{math.NaN(), math.Inf(-1), math.Inf(1)} {
		rec := httptest.NewRecorder()
		writeJSON(rec, http.StatusOK, map[string]any{"choices": []any{map[string]any{"logprob": bad}}})
		if rec.Code != http.StatusInternalServerError {
			t.Errorf("%v: status %d, want 500 (an empty 200 is the defect)", bad, rec.Code)
		}
		if body := rec.Body.String(); !strings.Contains(body, "could not be encoded") || !strings.Contains(body, "unsupported value") {
			t.Errorf("%v: body %q does not name the encoding failure", bad, body)
		}
	}
}

func TestWriteJSON_ordinaryBodyIsUnchanged(t *testing.T) {
	rec := httptest.NewRecorder()
	writeJSON(rec, http.StatusCreated, map[string]any{"a": 1, "b": "x<y"})
	if rec.Code != http.StatusCreated || rec.Body.String() != "{\"a\":1,\"b\":\"x\\u003cy\"}\n" || rec.Header().Get("Content-Type") != "application/json" {
		t.Errorf("got %d %q %q", rec.Code, rec.Body.String(), rec.Header().Get("Content-Type"))
	}
}

// A zero-probability candidate is normal (a forced end-of-thinking token leaves every other candidate impossible), and its logprob is -Inf, which JSON cannot carry: the server reports
// OpenAI's -9999 for it. Other values, and NaN (a bug that writeJSON must keep reporting), are untouched.
func TestJSONLogprob_impossibleIsFiniteAndNothingElseMoves(t *testing.T) {
	if got := jsonLogprob(math.Inf(-1)); got != -9999.0 {
		t.Errorf("-Inf -> %v, want -9999", got)
	}
	for _, x := range []float64{0, -0.25, -17.5} {
		if got := jsonLogprob(x); got != x {
			t.Errorf("%v -> %v: an ordinary logprob must pass through unchanged", x, got)
		}
	}
	if got := jsonLogprob(math.NaN()); !math.IsNaN(got) {
		t.Errorf("NaN -> %v, want NaN left for writeJSON to report", got)
	}
	// the sentinel survives the encoder the response goes through
	rec := httptest.NewRecorder()
	writeJSON(rec, http.StatusOK, map[string]any{"logprob": jsonLogprob(math.Inf(-1))})
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "-9999") {
		t.Errorf("got %d %q", rec.Code, rec.Body.String())
	}
}
