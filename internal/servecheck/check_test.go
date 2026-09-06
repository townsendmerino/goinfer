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

// TestStructured_truncatedDigitFails pins V-17 (docs/review-2026-09-04.md): the prompt asks for
// "366" specifically because that is M-27's shape (StopWhenComplete used to stop at the first
// complete document and return a single truncated digit). The old check only confirmed the
// output parsed as SOME json.Number — a truncated "3" parses exactly as cleanly as "366" and
// would report OK against the very regression this row exists to catch.
func TestStructured_truncatedDigitFails(t *testing.T) {
	c := newFake(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"3"}}]}`)) // M-27's truncation shape
	})
	res := c.Structured(context.Background(), "m")
	if res.OK {
		t.Error("a truncated single digit (M-27's exact shape) passed the structured-output check (V-17)")
	}
	if !strings.Contains(res.Detail, "366") {
		t.Errorf("the reason must name the expected value, got %q", res.Detail)
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

// A checker that cannot go red is not a checker (see the header). These drive Tools against
// servers that are wrong in one specific way each — the ways that actually break a harness on
// turn two rather than turn one.

// toolServer answers turn one with a tool call built from `call`, and turn two with plain text.
// nil `call` means "answer without calling the tool".
func toolServer(t *testing.T, call map[string]any) *Client {
	t.Helper()
	return newFake(t, func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Messages []map[string]any `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		turnTwo := false
		for _, m := range req.Messages {
			if m["role"] == "tool" {
				turnTwo = true
			}
		}
		msg := map[string]any{"content": "It is raining in Paris, 14C."}
		if !turnTwo {
			if call == nil {
				msg = map[string]any{"content": "I do not know.", "role": "assistant"}
				_ = json.NewEncoder(w).Encode(map[string]any{
					"choices": []map[string]any{{"message": msg, "finish_reason": "stop"}}})
				return
			}
			msg = map[string]any{"content": nil, "tool_calls": []any{call}}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": msg}}})
	})
}

func goodCall() map[string]any {
	return map[string]any{"id": "call_1", "type": "function",
		"function": map[string]any{"name": "get_weather", "arguments": `{"city":"Paris"}`}}
}

func TestTools_roundTripOK(t *testing.T) {
	res := toolServer(t, goodCall()).Tools(context.Background(), "m")
	if !res.OK {
		t.Fatalf("a correct two-turn tool round-trip must pass: %+v", res)
	}
	if !strings.Contains(res.Detail, "2 turns") {
		t.Errorf("detail should say the round-trip completed, got %q", res.Detail)
	}
}

// The id is what pairs a result with its call. Without it a harness has nothing to send back,
// and the turn-one response still looks well-formed.
func TestTools_missingIDFails(t *testing.T) {
	call := goodCall()
	delete(call, "id")
	res := toolServer(t, call).Tools(context.Background(), "m")
	if res.OK || res.Skip {
		t.Fatalf("a tool_call with no id must FAIL, got %+v", res)
	}
	if !strings.Contains(res.Detail, "id") {
		t.Errorf("the reason must name the missing id, got %q", res.Detail)
	}
}

// OpenAI's arguments field is a JSON *string*. A server that emits an object there breaks every
// client that unmarshals the field, and the response is otherwise perfectly shaped.
func TestTools_argumentsAsObjectFails(t *testing.T) {
	call := goodCall()
	call["function"] = map[string]any{"name": "get_weather", "arguments": map[string]any{"city": "Paris"}}
	res := toolServer(t, call).Tools(context.Background(), "m")
	if res.OK || res.Skip {
		t.Fatalf("arguments as an object must FAIL, got %+v", res)
	}
}

// The half that fails separately: turn one is fine and the server then rejects the tool result.
// This is the failure a harness sees as a conversation that dies after the first tool.
func TestTools_rejectedToolResultFails(t *testing.T) {
	c := newFake(t, func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Messages []map[string]any `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		for _, m := range req.Messages {
			if m["role"] == "tool" {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"error":{"message":"unsupported role: tool"}}`))
				return
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{
			{"message": map[string]any{"content": nil, "tool_calls": []any{goodCall()}}}}})
	})
	res := c.Tools(context.Background(), "m")
	if res.OK || res.Skip {
		t.Fatalf("a server that rejects the tool result must FAIL, got %+v", res)
	}
	if !strings.Contains(res.Detail, "tool result rejected") {
		t.Errorf("the reason must say the tool RESULT was rejected, got %q", res.Detail)
	}
}

// A model that answers without calling the tool is a fact about the checkpoint, not a broken
// route. Reporting it red trains an operator to ignore the row.
func TestTools_noCallIsSkipNotFail(t *testing.T) {
	res := toolServer(t, nil).Tools(context.Background(), "m")
	if res.OK {
		t.Fatal("a turn with no tool call must not report OK")
	}
	if !res.Skip {
		t.Fatalf("a model declining to call the tool must SKIP, not fail: %+v", res)
	}
}
