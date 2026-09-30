//go:build goinfer_testhooks

package serveapp

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/chatapp"
	"github.com/townsendmerino/goinfer/internal/loadflags"
)

// TestBatch_cliAndServeAgree is J5's interchange gate: ONE input file, run through `goinfer-chat --batch`'s runner and through
// serve's POST /v1/batches on the same loaded model, must come out the same, line by line — same text, same finish reason, same
// token counts — and the lines one refuses the other must refuse. The two never share a generation path (serve's goes through
// admission, sessions and its own stop-string streamer; the CLI's through decoder.Generate directly), so each side's own tests
// can only show it agrees with itself. The file format is shared code (internal/batchio); THIS is what shows the behaviour is.
//
// Every line is greedy or seeded, so equality is exact, not statistical.
func TestBatch_cliAndServeAgree(t *testing.T) {
	path := os.Getenv("GOINFER_SERVE_MODEL")
	if path == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf> for the CLI/serve batch interchange test")
	}
	srv, err := newServer(config{
		models: modelFlag{{name: "m", path: path}}, load: loadflags.Flags{Backend: "cpu", Quant: "int8int8"}, kvSessions: 0,
	})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	lm := srv.models["m"]
	if lm == nil || lm.tk == nil || lm.model == nil {
		t.Fatal("the served model is missing its tokenizer or weights")
	}

	user := func(s string) []map[string]any { return []map[string]any{{"role": "user", "content": s}} }
	type line struct {
		id   string
		body map[string]any
		// refused: both sides must put this line in their error file.
		refused bool
	}
	lines := []line{
		{id: "greedy", body: map[string]any{"messages": user("Write a haiku about Go."), "max_tokens": 40, "temperature": 0}},
		{id: "system", body: map[string]any{"messages": append([]map[string]any{{"role": "system", "content": "Answer in exactly three words."}}, user("What is the capital of France?")...), "max_tokens": 24, "temperature": 0}},
		{id: "seeded-sampled", body: map[string]any{"messages": user("Name a colour."), "max_tokens": 24, "temperature": 0.9, "top_p": 0.9, "top_k": 40, "seed": 7}},
		{id: "seeded-penalties", body: map[string]any{"messages": user("Say something about the sea."), "max_tokens": 30, "temperature": 0.7, "seed": 11, "frequency_penalty": 0.5, "presence_penalty": 0.3}},
		{id: "stop-string", body: map[string]any{"messages": user("Count from 1 to 9 separated by spaces."), "max_tokens": 40, "temperature": 0, "stop": []string{"4"}}},
		{id: "length", body: map[string]any{"messages": user("Tell me a long story."), "max_tokens": 3, "temperature": 0}},
		{id: "max-completion-tokens", body: map[string]any{"messages": user("Tell me a long story."), "max_tokens": 50, "max_completion_tokens": 5, "temperature": 0}},
		{id: "content-parts", body: map[string]any{"messages": []map[string]any{{"role": "user", "content": []map[string]any{{"type": "text", "text": "Reply with the single word "}, {"type": "text", "text": "pong."}}}}, "max_tokens": 12, "temperature": 0}},
		{id: "json-object", body: map[string]any{"messages": user("Give a JSON object with a key a."), "max_tokens": 60, "temperature": 0, "response_format": map[string]any{"type": "json_object"}}},
		{id: "json-schema", body: map[string]any{"messages": user("Give a number."), "max_tokens": 60, "temperature": 0, "response_format": map[string]any{"type": "json_schema", "json_schema": map[string]any{"name": "n", "schema": map[string]any{
			"type": "object", "properties": map[string]any{"n": map[string]any{"type": "integer"}}, "required": []string{"n"}, "additionalProperties": false}}}}},
		{id: "thinking-off", body: map[string]any{"messages": user("Say hi."), "max_tokens": 12, "temperature": 0, "chat_template_kwargs": map[string]any{"enable_thinking": false}}},
		// Thinking, on a model whose template has a think control (a no-op, and still must agree, on one that does not): on with
		// a request budget; on with so small a turn that the room rule forces the block closed; and the reasoning left in the text.
		{id: "thinking-on-budget", body: map[string]any{"messages": user("What is 17 times 3?"), "max_tokens": 160, "temperature": 0, "chat_template_kwargs": map[string]any{"enable_thinking": true}, "thinking_token_budget": 40}},
		{id: "thinking-on-room", body: map[string]any{"messages": user("What is 17 times 3?"), "max_tokens": 24, "temperature": 0, "chat_template_kwargs": map[string]any{"enable_thinking": true}}},
		{id: "thinking-format-none", body: map[string]any{"messages": user("What is 2 plus 2?"), "max_tokens": 60, "temperature": 0, "chat_template_kwargs": map[string]any{"enable_thinking": true}, "reasoning_format": "none"}},
		{id: "refused-tools", refused: true, body: map[string]any{"messages": user("hi"), "tools": []map[string]any{{"type": "function", "function": map[string]any{"name": "f"}}}}},
		{id: "refused-image", refused: true, body: map[string]any{"messages": []map[string]any{{"role": "user", "content": []map[string]any{{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,AA=="}}}}}}},
		{id: "refused-no-messages", refused: true, body: map[string]any{"messages": []map[string]any{}}},
	}
	var in bytes.Buffer
	var inputLines []string
	for _, l := range lines {
		body, _ := json.Marshal(l.body)
		row := mustJSONLine(t, map[string]any{"custom_id": l.id, "method": "POST", "url": "/v1/chat/completions", "body": json.RawMessage(body)})
		in.WriteString(row + "\n")
		inputLines = append(inputLines, row)
	}

	// --- the CLI runner
	cliOut, cliErr, err := chatapp.RunBatchForTest(lm.tk, lm.model, "m", chat.ThinkTemplate, in.Bytes())
	if err != nil {
		t.Fatalf("CLI runner: %v", err)
	}
	cli := parseBatchFile(t, cliOut)
	cliBad := parseBatchFile(t, cliErr)

	// --- serve's batch API, same file
	ts := httptest.NewServer(batchesTestMux(srv))
	defer ts.Close()
	fileID := uploadJSONLFile(t, ts, "in.jsonl", inputLines)
	resp := postJSON(t, ts.URL+"/v1/batches", map[string]any{"input_file_id": fileID, "endpoint": "/v1/chat/completions", "completion_window": "24h"})
	var created struct{ ID string }
	if err := json.NewDecoder(resp.Body).Decode(&created); err != nil {
		t.Fatalf("decode: %v", err)
	}
	resp.Body.Close()
	var raw struct {
		Status       string `json:"status"`
		OutputFileID string `json:"output_file_id"`
		ErrorFileID  string `json:"error_file_id"`
	}
	for deadline := time.Now().Add(3 * time.Minute); ; time.Sleep(100 * time.Millisecond) {
		st, err := http.Get(ts.URL + "/v1/batches/" + created.ID)
		if err != nil {
			t.Fatal(err)
		}
		derr := json.NewDecoder(st.Body).Decode(&raw)
		st.Body.Close()
		if derr != nil {
			t.Fatal(derr)
		}
		if raw.Status == "completed" {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("serve's batch did not complete (status %q)", raw.Status)
		}
	}
	serve := map[string]map[string]any{}
	for _, l := range fetchJSONLLines(t, ts, raw.OutputFileID) {
		serve[l["custom_id"].(string)] = l
	}
	serveBad := map[string]map[string]any{}
	if raw.ErrorFileID != "" {
		for _, l := range fetchJSONLLines(t, ts, raw.ErrorFileID) {
			serveBad[l["custom_id"].(string)] = l
		}
	}

	// --- compare
	for _, l := range lines {
		if l.refused {
			if _, ok := cliBad[l.id]; !ok {
				t.Errorf("%s: the CLI runner did not refuse a line serve refuses", l.id)
			}
			if _, ok := serveBad[l.id]; !ok {
				t.Errorf("%s: serve did not refuse a line the CLI runner refuses", l.id)
			}
			continue
		}
		c, s := cli[l.id], serve[l.id]
		if c == nil || s == nil {
			t.Errorf("%s: missing a result (cli=%v serve=%v; cli errors %v, serve errors %v)", l.id, c != nil, s != nil, cliBad[l.id], serveBad[l.id])
			continue
		}
		cc, sc := firstChoice(t, c), firstChoice(t, s)
		if cc.content != sc.content || cc.reasoning != sc.reasoning || cc.finish != sc.finish {
			t.Errorf("%s differs:\n  cli   : finish=%s content=%q reasoning=%q\n  serve : finish=%s content=%q reasoning=%q",
				l.id, cc.finish, cc.content, cc.reasoning, sc.finish, sc.content, sc.reasoning)
		}
		if cc.prompt != sc.prompt || cc.completion != sc.completion {
			t.Errorf("%s token counts differ: cli prompt=%d completion=%d, serve prompt=%d completion=%d", l.id, cc.prompt, cc.completion, sc.prompt, sc.completion)
		}
		t.Logf("%-22s finish=%-6s prompt=%d completion=%d reasoning=%dB content=%q", l.id, cc.finish, cc.prompt, cc.completion, len(cc.reasoning), clip(cc.content, 48))
	}
	// The one property of the set worth asserting outright: it really exercised a stop and a length finish, not two "stop"s.
	if f := firstChoice(t, cli["stop-string"]).finish; f != "stop" || strings.Contains(firstChoice(t, cli["stop-string"]).content, "4") {
		t.Errorf("stop-string line did not stop on its stop string: %+v", firstChoice(t, cli["stop-string"]))
	}
	if f := firstChoice(t, cli["length"]).finish; f != "length" {
		t.Errorf("length line finish = %q, want length", f)
	}
}

type choiceView struct {
	content, reasoning, finish string
	prompt, completion         int
}

func firstChoice(t *testing.T, line map[string]any) choiceView {
	t.Helper()
	resp, _ := line["response"].(map[string]any)
	body, _ := resp["body"].(map[string]any)
	choices, _ := body["choices"].([]any)
	if len(choices) == 0 {
		t.Fatalf("no choices in %+v", line)
	}
	ch := choices[0].(map[string]any)
	msg, _ := ch["message"].(map[string]any)
	usage, _ := body["usage"].(map[string]any)
	v := choiceView{finish: ch["finish_reason"].(string)}
	v.content, _ = msg["content"].(string)
	v.reasoning, _ = msg["reasoning_content"].(string)
	if f, ok := usage["prompt_tokens"].(float64); ok {
		v.prompt = int(f)
	}
	if f, ok := usage["completion_tokens"].(float64); ok {
		v.completion = int(f)
	}
	return v
}

func parseBatchFile(t *testing.T, data []byte) map[string]map[string]any {
	t.Helper()
	out := map[string]map[string]any{}
	for _, raw := range bytes.Split(bytes.TrimSpace(data), []byte("\n")) {
		if len(raw) == 0 {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal(raw, &m); err != nil {
			t.Fatalf("parse %q: %v", raw, err)
		}
		out[m["custom_id"].(string)] = m
	}
	return out
}

func clip(s string, n int) string {
	if len(s) > n {
		return s[:n] + "…"
	}
	return s
}
