package servecheck

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

// audit-2026-09-10 G-12: the tools row passed turn two on any 200, and the stop row passed when the
// stop sequence never fired. These fakes drive the FAILING direction the existing ones never did.

// turnTwoServer answers turn one with goodCall(), and the tool-result turn with turnTwo.
func turnTwoServer(t *testing.T, turnTwo map[string]any) *Client {
	t.Helper()
	return newFake(t, func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Messages []map[string]any `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		msg := map[string]any{"content": nil, "tool_calls": []any{goodCall()}}
		for _, m := range req.Messages {
			if m["role"] == "tool" {
				msg = turnTwo
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": msg, "finish_reason": "stop"}}})
	})
}

// An empty answer to the tool result is a conversation that died on turn two.
func TestTools_emptyAnswerOnTurnTwoFails(t *testing.T) {
	res := turnTwoServer(t, map[string]any{"content": ""}).Tools(context.Background(), "m")
	if res.OK || res.Skip {
		t.Fatalf("an empty answer to the tool result must FAIL, got %+v", res)
	}
}

// Asking for the same tool again instead of answering is M-18's agent-livelock shape.
func TestTools_repeatedToolCallOnTurnTwoFails(t *testing.T) {
	res := turnTwoServer(t, map[string]any{"content": nil, "tool_calls": []any{goodCall()}}).Tools(context.Background(), "m")
	if res.OK || res.Skip {
		t.Fatalf("a second tool call in answer to the tool result must FAIL, got %+v", res)
	}
	if !strings.Contains(res.Detail, "again") {
		t.Errorf("the reason must say the model asked for the tool again, got %q", res.Detail)
	}
}

func stopServer(t *testing.T, content string) *Client {
	t.Helper()
	return newFake(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{
			{"message": map[string]any{"content": content}, "finish_reason": "stop"}}})
	})
}

// A reply that never reaches the stop sequence proves nothing about stop handling.
func TestStop_neverFiresFails(t *testing.T) {
	res := stopServer(t, "Sure, I can count.").Stop(context.Background(), "m")
	if res.OK {
		t.Fatalf("a reply that never reached the stop sequence must FAIL, got %+v", res)
	}
}

// The passing direction: the count reached 4, and stopped before 5.
func TestStop_firesPasses(t *testing.T) {
	res := stopServer(t, "1, 2, 3, 4, ").Stop(context.Background(), "m")
	if !res.OK {
		t.Fatalf("a count that stops before 5 must pass, got %+v", res)
	}
}
