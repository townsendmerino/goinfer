package serveapp

import (
	"encoding/json"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/loadflags"
)

// TestHarmonyTools_realModelLoop is the gpt-oss tool loop through serve's real handlers on the real model: a request with a declared
// function must come back as a tool call (finish_reason "tool_calls", the call parsed out of the model's own channel message, the
// reasoning in reasoning_content and neither in content), and replaying that call with its result must produce an answer that uses
// the result. Gated on GOINFER_SERVE_MODEL and skipped unless that model speaks Harmony — a skip is not a pass.
//
// Greedy, reasoning effort low so the turn is short on a CPU fallback; one prompt, so it shows the loop works, not how often.
func TestHarmonyTools_realModelLoop(t *testing.T) {
	path := os.Getenv("GOINFER_SERVE_MODEL")
	if path == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<gpt-oss .gguf> for the Harmony tool-loop test")
	}
	srv, err := newServer(config{
		models: modelFlag{{name: "m", path: path}}, load: loadflags.Flags{Backend: "cpu", Quant: "int4"}, kvSessions: 0,
	})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	if lm := srv.models["m"]; lm == nil || lm.tmpl == nil || lm.tmpl.Name() != "harmony" {
		t.Skip("the served model's chat template is not Harmony")
	}
	ts := httptest.NewServer(batchesTestMux(srv))
	defer ts.Close()

	tools := []map[string]any{{"type": "function", "function": map[string]any{
		"name": "get_weather", "description": "Get the current weather for a city",
		"parameters": map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string", "description": "City name"}}, "required": []string{"city"}},
	}}}
	ask := func(messages []map[string]any) (msg map[string]any, finish string) {
		t.Helper()
		resp := postJSON(t, ts.URL+"/v1/chat/completions", map[string]any{
			"model": "m", "messages": messages, "tools": tools, "max_tokens": 400, "temperature": 0, "reasoning_effort": "low",
		})
		defer resp.Body.Close()
		var out struct {
			Choices []struct {
				Message      map[string]any `json:"message"`
				FinishReason string         `json:"finish_reason"`
			} `json:"choices"`
			Error map[string]any `json:"error"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil || len(out.Choices) == 0 {
			t.Fatalf("status %d, decode: %v, error: %v", resp.StatusCode, err, out.Error)
		}
		return out.Choices[0].Message, out.Choices[0].FinishReason
	}

	// Turn 1: the model should call the function.
	user := map[string]any{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}
	msg, finish := ask([]map[string]any{user})
	t.Logf("turn 1: finish=%s content=%q reasoning=%q tool_calls=%v", finish, msg["content"], msg["reasoning_content"], msg["tool_calls"])
	calls, _ := msg["tool_calls"].([]any)
	if finish != "tool_calls" || len(calls) != 1 {
		t.Fatalf("turn 1 should be one tool call: finish=%q calls=%v", finish, msg["tool_calls"])
	}
	fn := calls[0].(map[string]any)["function"].(map[string]any)
	var args map[string]any
	if fn["name"] != "get_weather" || json.Unmarshal([]byte(fn["arguments"].(string)), &args) != nil || !strings.Contains(strings.ToLower(args["city"].(string)), "paris") {
		t.Fatalf("the call is not get_weather for Paris: %v", fn)
	}
	for _, field := range []string{"content", "reasoning_content"} {
		if s, _ := msg[field].(string); strings.Contains(s, "<|") {
			t.Errorf("%s carries Harmony markup: %q", field, s)
		}
	}
	if r, _ := msg["reasoning_content"].(string); r == "" {
		t.Error("the analysis that preceded the call should arrive as reasoning_content")
	}

	// Turn 2: replay the call and its result; the model should answer with it.
	assistant := map[string]any{"role": "assistant", "content": msg["content"], "tool_calls": calls}
	if r, _ := msg["reasoning_content"].(string); r != "" {
		assistant["reasoning_content"] = r
	}
	tool := map[string]any{"role": "tool", "tool_call_id": calls[0].(map[string]any)["id"], "name": "get_weather", "content": "It is 18 degrees Celsius and sunny in Paris."}
	msg2, finish2 := ask([]map[string]any{user, assistant, tool})
	t.Logf("turn 2: finish=%s content=%q reasoning=%q", finish2, msg2["content"], msg2["reasoning_content"])
	ans, _ := msg2["content"].(string)
	if finish2 != "stop" || !strings.Contains(ans, "18") || strings.Contains(ans, "<|") {
		t.Errorf("turn 2 should answer using the result: finish=%q content=%q", finish2, ans)
	}
}
