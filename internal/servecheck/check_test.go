package servecheck

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// A checker that cannot go red is not a checker. These drive the client against servers that
// are deliberately wrong in one specific way each, and assert it NOTICES — the same
// break-it-first discipline this project applies to its own lints and gates. They need no
// model, so they run in CI, which is where the routes a harness uses currently have no cover
// at all (ten of the serveapp test files skip without one).

func newFake(t *testing.T, h http.HandlerFunc) *Client {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return &Client{BaseURL: srv.URL}
}

func sseFrames(w http.ResponseWriter, frames ...string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	f, _ := w.(http.Flusher)
	for _, fr := range frames {
		fmt.Fprintf(w, "data: %s\n\n", fr)
		if f != nil {
			f.Flush()
		}
	}
}

func delta(s string) string {
	b, _ := json.Marshal(map[string]any{"choices": []map[string]any{{"delta": map[string]string{"content": s}}}})
	return string(b)
}

// A stream that never sends a usage block must FAIL: several harnesses require usage for
// accounting, and a stream can omit it silently.
func TestChat_missingUsageFails(t *testing.T) {
	c := newFake(t, func(w http.ResponseWriter, r *http.Request) {
		sseFrames(w, delta("hi"), delta(" there"), "[DONE]")
	})
	res := c.Chat(context.Background(), "m", "hi", 8, "chat")
	if res.OK {
		t.Error("a stream with no usage block must fail")
	}
	if !strings.Contains(res.Detail, "NO usage") {
		t.Errorf("the reason must name the missing usage, got %q", res.Detail)
	}
}

func TestChat_okWithUsage(t *testing.T) {
	c := newFake(t, func(w http.ResponseWriter, r *http.Request) {
		u, _ := json.Marshal(map[string]any{"usage": map[string]int{"completion_tokens": 2}})
		sseFrames(w, delta("hi"), delta(" there"), string(u), "[DONE]")
	})
	res := c.Chat(context.Background(), "m", "hi", 8, "chat")
	if !res.OK {
		t.Errorf("a well-formed stream must pass, got %q", res.Detail)
	}
	if !strings.Contains(res.Detail, "TTFT") {
		t.Errorf("the row must carry a number, got %q", res.Detail)
	}
}

// count_tokens disagreeing with billed usage must FAIL: a harness budgets its context with
// the first number and is charged the second.
func TestCountTokens_mismatchFails(t *testing.T) {
	c := newFake(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if strings.HasSuffix(r.URL.Path, "/count_tokens") {
			_, _ = w.Write([]byte(`{"input_tokens": 17}`))
			return
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"x"}}],"usage":{"prompt_tokens":23}}`))
	})
	res := c.CountTokens(context.Background(), "m")
	if res.OK {
		t.Error("a count_tokens/usage mismatch must fail")
	}
	for _, want := range []string{"17", "23"} {
		if !strings.Contains(res.Detail, want) {
			t.Errorf("the reason must name BOTH numbers, got %q", res.Detail)
		}
	}
}

func TestCountTokens_matchPasses(t *testing.T) {
	c := newFake(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if strings.HasSuffix(r.URL.Path, "/count_tokens") {
			_, _ = w.Write([]byte(`{"input_tokens": 23}`))
			return
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"x"}}],"usage":{"prompt_tokens":23}}`))
	})
	if res := c.CountTokens(context.Background(), "m"); !res.OK {
		t.Errorf("matching counts must pass, got %q", res.Detail)
	}
}

// A leaked stop string must FAIL — the harness-visible symptom is corrupted text, not an error.
func TestStop_leakFails(t *testing.T) {
	c := newFake(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"1, 2, 3, 4, 5"},"finish_reason":"stop"}]}`))
	})
	res := c.Stop(context.Background(), "m")
	if res.OK {
		t.Error("a leaked stop string must fail")
	}
	if !strings.Contains(res.Detail, "leaked") {
		t.Errorf("the reason must say what happened, got %q", res.Detail)
	}
}

// Structured output that does not parse as the requested schema must FAIL — this is the
// README's headline promise, and M-27's shape (a top-level integer) is the one that broke.
func TestStructured_nonConformingFails(t *testing.T) {
	c := newFake(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"the answer is 366"}}]}`))
	})
	res := c.Structured(context.Background(), "m")
	if res.OK {
		t.Error("prose where an integer was demanded must fail")
	}
}

func TestStructured_conformingPasses(t *testing.T) {
	c := newFake(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"366"}}]}`))
	})
	if res := c.Structured(context.Background(), "m"); !res.OK {
		t.Errorf("a conforming integer must pass, got %q", res.Detail)
	}
}

// An HTTP error must be reported with its body, not swallowed into a bare "failed".
func TestErrorsCarryTheServersReason(t *testing.T) {
	c := newFake(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"unsupported response_format type"}}`))
	})
	res := c.Structured(context.Background(), "m")
	if res.OK || !strings.Contains(res.Detail, "unsupported response_format") {
		t.Errorf("the server's own reason must survive into the row, got %q", res.Detail)
	}
}

// No models is a fact to report, not a crash.
func TestModels_emptyIsReportedNotFailed(t *testing.T) {
	c := newFake(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[]}`))
	})
	res, ids := c.Models(context.Background())
	if !res.OK || len(ids) != 0 {
		t.Errorf("an empty model list is a valid server state, got ok=%v detail=%q", res.OK, res.Detail)
	}
}
