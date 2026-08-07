package serveapp

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

// --- translation unit tests (no model) ---

func TestAnthropicText(t *testing.T) {
	if got := anthropicText(json.RawMessage(`"hello"`)); got != "hello" {
		t.Errorf("string: %q", got)
	}
	if got := anthropicText(json.RawMessage(`[{"type":"text","text":"a"},{"type":"text","text":"b"}]`)); got != "ab" {
		t.Errorf("array: %q", got)
	}
	if got := anthropicText(nil); got != "" {
		t.Errorf("empty: %q", got)
	}
}

func TestAnthropicTurns_systemArrayAndMerge(t *testing.T) {
	req := &anthropicReq{
		System: json.RawMessage(`[{"type":"text","text":"sys"}]`),
		Messages: []anthropicMessage{
			{Role: "user", Content: json.RawMessage(`"hi"`)},
			{Role: "user", Content: json.RawMessage(`"there"`)}, // adjacent same-role → merged
		},
	}
	system, turns, aerr := anthropicTurns(req)
	if aerr != nil {
		t.Fatalf("aerr: %v", aerr)
	}
	if system != "sys" {
		t.Errorf("system = %q, want sys", system)
	}
	if len(turns) != 1 || turns[0].Role != "user" || turns[0].Content != "hi\nthere" {
		t.Errorf("merge: %+v", turns)
	}
}

func TestAnthropicTurns_cacheControlAndUnknownIgnored(t *testing.T) {
	// cache_control on a block and an unknown block type are both silently ignored.
	req := &anthropicReq{Messages: []anthropicMessage{{Role: "user",
		Content: json.RawMessage(`[{"type":"text","text":"x","cache_control":{"type":"ephemeral"}},{"type":"mystery","foo":1}]`)}}}
	_, turns, aerr := anthropicTurns(req)
	if aerr != nil {
		t.Fatalf("aerr: %v", aerr)
	}
	if len(turns) != 1 || turns[0].Content != "x" {
		t.Errorf("turns = %+v, want one user turn 'x'", turns)
	}
}

// anthropicTurns now SKIPS image blocks (they're collected by anthropicImages and
// routed to the vision path); the handler 400s an image when no vision tower is
// loaded. So turn-building drops the image and keeps the text.
func TestAnthropicTurns_imageSkipped(t *testing.T) {
	req := &anthropicReq{Messages: []anthropicMessage{{Role: "user",
		Content: json.RawMessage(`[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"x"}},{"type":"text","text":"caption it"}]`)}}}
	_, turns, aerr := anthropicTurns(req)
	if aerr != nil {
		t.Fatalf("image block should be skipped, not error: %v", aerr)
	}
	if len(turns) != 1 || turns[0].Content != "caption it" {
		t.Errorf("turns = %+v, want one user turn 'caption it'", turns)
	}
}

// anthropicImages decodes base64 image blocks and rejects a non-base64 source.
func TestAnthropicImages(t *testing.T) {
	// 1x1 px PNG, base64.
	const png1x1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
	req := &anthropicReq{Messages: []anthropicMessage{{Role: "user",
		Content: json.RawMessage(`[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + png1x1 + `"}}]`)}}}
	imgs, err := anthropicImages(req)
	if err != nil || len(imgs) != 1 || len(imgs[0].data) == 0 {
		t.Fatalf("decode base64 image: imgs=%d err=%v", len(imgs), err)
	}
	// URL source → rejected (SSRF guard).
	bad := &anthropicReq{Messages: []anthropicMessage{{Role: "user",
		Content: json.RawMessage(`[{"type":"image","source":{"type":"url","url":"http://x/y.png"}}]`)}}}
	if _, err := anthropicImages(bad); err == nil {
		t.Error("url image source should be rejected")
	}
}

func TestAnthropicTurns_toolReplay(t *testing.T) {
	req := &anthropicReq{Messages: []anthropicMessage{
		{Role: "assistant", Content: json.RawMessage(
			`[{"type":"text","text":"let me check"},{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"location":"Paris"}}]`)},
		{Role: "user", Content: json.RawMessage(
			`[{"type":"tool_result","tool_use_id":"toolu_1","content":"sunny"}]`)},
	}}
	_, turns, aerr := anthropicTurns(req)
	if aerr != nil {
		t.Fatalf("aerr: %v", aerr)
	}
	if len(turns) != 2 {
		t.Fatalf("turns = %+v, want 2", turns)
	}
	a := turns[0]
	if a.Role != "assistant" || a.Content != "let me check" || len(a.ToolCalls) != 1 || a.ToolCalls[0].Name != "get_weather" {
		t.Errorf("assistant turn = %+v", a)
	}
	tr := turns[1]
	if tr.Role != "tool" || tr.Content != "sunny" || tr.ToolName != "get_weather" || tr.ToolCallID != "toolu_1" {
		t.Errorf("tool turn = %+v", tr)
	}
}

func TestAnthropicToSampling(t *testing.T) {
	f := func(v float64) *float64 { return &v }
	i := func(v int) *int { return &v }
	req := &anthropicReq{Temperature: f(0.5), TopP: f(0.9), TopK: i(40), MaxTokens: i(50), StopSequences: []string{"END", "STOP"}}
	sm := req.toSampling()
	if deref(sm.Temperature, -1) != 0.5 || deref(sm.TopP, -1) != 0.9 || deref(sm.TopK, -1) != 40 || deref(sm.MaxTokens, -1) != 50 {
		t.Errorf("sampling fields not mapped: %+v", sm)
	}
	if got := parseStop(sm.Stop); strings.Join(got, ",") != "END,STOP" {
		t.Errorf("stop_sequences → stop = %v", got)
	}
}

func TestAnthropicStopReason(t *testing.T) {
	if r, s := anthropicStopReason("length", ""); r != "max_tokens" || s != nil {
		t.Errorf("length → (%q,%v)", r, s)
	}
	if r, s := anthropicStopReason("stop", "END"); r != "stop_sequence" || s != "END" {
		t.Errorf("stop seq → (%q,%v)", r, s)
	}
	if r, s := anthropicStopReason("stop", ""); r != "end_turn" || s != nil {
		t.Errorf("eos → (%q,%v)", r, s)
	}
}

func TestAnthropicForcedTool(t *testing.T) {
	tools := []anthropicTool{{Name: "a"}, {Name: "b"}}
	ct := anthropicTools(tools)
	if anthropicForcedTool("none", "", ct) != nil {
		t.Error("none should not force")
	}
	if f := anthropicForcedTool("tool", "b", ct); f == nil || f.Name != "b" {
		t.Errorf("named tool: %+v", f)
	}
	if f := anthropicForcedTool("any", "", ct[:1]); f == nil || f.Name != "a" {
		t.Errorf("any with a lone tool should force it: %+v", f)
	}
	if anthropicForcedTool("any", "", ct) != nil {
		t.Error("any with 2 tools can't constrain to one schema → no constraint")
	}
	// auto never forces — even a lone tool stays optional (the model may answer).
	if anthropicForcedTool("auto", "", ct[:1]) != nil {
		t.Error("auto must not force a lone tool")
	}
}

// --- handler error paths (empty server, no model needed) ---

func TestAnthropicHandler_errors(t *testing.T) {
	srv, err := newServer(config{})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/messages", srv.handleMessages)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	post := func(body string) (int, string) {
		r, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatalf("POST: %v", err)
		}
		defer r.Body.Close()
		var e struct {
			Type  string `json:"type"`
			Error struct {
				Type string `json:"type"`
			} `json:"error"`
		}
		json.NewDecoder(r.Body).Decode(&e)
		if e.Type != "error" {
			t.Errorf("error body type = %q, want error (status %d)", e.Type, r.StatusCode)
		}
		return r.StatusCode, e.Error.Type
	}

	// Missing max_tokens → 400 invalid_request_error.
	if code, kind := post(`{"model":"x","messages":[{"role":"user","content":"hi"}]}`); code != 400 || kind != "invalid_request_error" {
		t.Errorf("missing max_tokens: (%d,%q), want (400,invalid_request_error)", code, kind)
	}
	// max_tokens <= 0 → 400.
	if code, kind := post(`{"model":"x","max_tokens":0,"messages":[{"role":"user","content":"hi"}]}`); code != 400 || kind != "invalid_request_error" {
		t.Errorf("max_tokens 0: (%d,%q), want (400,invalid_request_error)", code, kind)
	}
	// Unknown model (empty registry) → 404 not_found_error.
	if code, kind := post(`{"model":"nope","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`); code != 404 || kind != "not_found_error" {
		t.Errorf("unknown model: (%d,%q), want (404,not_found_error)", code, kind)
	}
}

// --- integration (gated on a real .gguf) ---

func newAnthropicTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	path := os.Getenv("GOINFER_SERVE_MODEL")
	if path == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf> for the Anthropic integration tests")
	}
	srv, err := newServer(config{models: modelFlag{{name: "test-model", path: path}}, backend: "cpu", quant: "int8int8", kvSessions: 4})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/messages", srv.handleMessages)
	mux.HandleFunc("POST /v1/messages/count_tokens", srv.handleCountTokens)
	return httptest.NewServer(mux)
}

// anthropicResp is the non-streaming response subset the tests assert on.
type anthropicResp struct {
	Type    string `json:"type"`
	Role    string `json:"role"`
	Model   string `json:"model"`
	Content []struct {
		Type  string          `json:"type"`
		Text  string          `json:"text"`
		ID    string          `json:"id"`
		Name  string          `json:"name"`
		Input json.RawMessage `json:"input"`
	} `json:"content"`
	StopReason string `json:"stop_reason"`
	Usage      struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// TestServe_anthropic_integration runs the llama.cpp blog's "basic chat
// completion" shape verbatim, plus count_tokens. Gated on GOINFER_SERVE_MODEL.
func TestServe_anthropic_integration(t *testing.T) {
	ts := newAnthropicTestServer(t)
	defer ts.Close()

	body := `{"model":"test-model","max_tokens":64,"temperature":0,
		"messages":[{"role":"user","content":"Hello, world"}]}`
	r, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer r.Body.Close()
	if r.StatusCode != 200 {
		t.Fatalf("status %d", r.StatusCode)
	}
	var out anthropicResp
	if err := json.NewDecoder(r.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if out.Type != "message" || out.Role != "assistant" || out.Model != "test-model" {
		t.Errorf("envelope wrong: %+v", out)
	}
	if len(out.Content) == 0 || out.Content[0].Type != "text" || out.Content[0].Text == "" {
		t.Errorf("no text content: %+v", out.Content)
	}
	if out.StopReason != "end_turn" && out.StopReason != "max_tokens" {
		t.Errorf("stop_reason = %q", out.StopReason)
	}
	if out.Usage.InputTokens == 0 || out.Usage.OutputTokens == 0 {
		t.Errorf("usage = %+v", out.Usage)
	}
	t.Logf("anthropic content: %q (stop=%s, usage=%+v)", out.Content[0].Text, out.StopReason, out.Usage)

	// count_tokens: same shape minus max_tokens → {"input_tokens": N>0}.
	r2, err := http.Post(ts.URL+"/v1/messages/count_tokens", "application/json",
		strings.NewReader(`{"model":"test-model","messages":[{"role":"user","content":"Hello, world"}]}`))
	if err != nil {
		t.Fatalf("count_tokens POST: %v", err)
	}
	defer r2.Body.Close()
	var ct struct {
		InputTokens int `json:"input_tokens"`
	}
	json.NewDecoder(r2.Body).Decode(&ct)
	if ct.InputTokens <= 0 {
		t.Errorf("count_tokens input_tokens = %d, want > 0", ct.InputTokens)
	}
}

// sseEventLine is one parsed Anthropic SSE event (its type + raw data json).
type sseEventLine struct {
	event string
	data  string
}

func readAnthropicSSE(t *testing.T, body *bufio.Scanner) []sseEventLine {
	t.Helper()
	var events []sseEventLine
	var cur sseEventLine
	for body.Scan() {
		line := body.Text()
		switch {
		case strings.HasPrefix(line, "event: "):
			cur.event = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			cur.data = strings.TrimPrefix(line, "data: ")
		case line == "":
			if cur.event != "" {
				events = append(events, cur)
			}
			cur = sseEventLine{}
		}
	}
	return events
}

// TestServe_anthropic_streaming asserts the exact event sequence, that there is
// no [DONE], and that concatenated text_deltas equal the non-streaming text for
// the same (temperature 0) seed. Gated on GOINFER_SERVE_MODEL.
func TestServe_anthropic_streaming(t *testing.T) {
	ts := newAnthropicTestServer(t)
	defer ts.Close()

	ask := `"messages":[{"role":"user","content":"Say hello in one short sentence."}]`

	// Non-streaming reference text.
	r0, err := http.Post(ts.URL+"/v1/messages", "application/json",
		strings.NewReader(`{"model":"test-model","max_tokens":40,"temperature":0,`+ask+`}`))
	if err != nil {
		t.Fatalf("ref POST: %v", err)
	}
	var ref anthropicResp
	json.NewDecoder(r0.Body).Decode(&ref)
	r0.Body.Close()
	refText := ""
	if len(ref.Content) > 0 {
		refText = ref.Content[0].Text
	}

	// Streaming.
	r, err := http.Post(ts.URL+"/v1/messages", "application/json",
		strings.NewReader(`{"model":"test-model","max_tokens":40,"temperature":0,"stream":true,`+ask+`}`))
	if err != nil {
		t.Fatalf("stream POST: %v", err)
	}
	defer r.Body.Close()
	if ct := r.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content-type = %q", ct)
	}
	events := readAnthropicSSE(t, bufio.NewScanner(r.Body))

	// No [DONE] terminator anywhere.
	for _, e := range events {
		if strings.Contains(e.data, "[DONE]") {
			t.Fatal("anthropic stream must not contain [DONE]")
		}
	}
	// Required ordered backbone: message_start … content_block_start …
	// content_block_stop, message_delta, message_stop (ping permitted).
	var order []string
	var streamed strings.Builder
	for _, e := range events {
		order = append(order, e.event)
		if e.event == "content_block_delta" {
			var d struct {
				Delta struct {
					Type string `json:"type"`
					Text string `json:"text"`
				} `json:"delta"`
			}
			json.Unmarshal([]byte(e.data), &d)
			if d.Delta.Type == "text_delta" {
				streamed.WriteString(d.Delta.Text)
			}
		}
	}
	want := []string{"message_start", "content_block_start", "content_block_stop", "message_delta", "message_stop"}
	if !subsequence(order, want) {
		t.Errorf("event order %v missing backbone %v", order, want)
	}
	if order[0] != "message_start" {
		t.Errorf("first event = %q, want message_start", order[0])
	}
	if order[len(order)-1] != "message_stop" {
		t.Errorf("last event = %q, want message_stop", order[len(order)-1])
	}
	if streamed.String() != refText {
		t.Errorf("streamed text %q != non-streamed %q", streamed.String(), refText)
	}
}

// TestServe_anthropic_streamAbort asserts that a client disconnect mid-stream
// cancels generation: the model's decode mutex is free again promptly after the
// client goes away. Gated on GOINFER_SERVE_MODEL.
func TestServe_anthropic_streamAbort(t *testing.T) {
	path := os.Getenv("GOINFER_SERVE_MODEL")
	if path == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf> for the Anthropic abort test")
	}
	srv, err := newServer(config{models: modelFlag{{name: "test-model", path: path}}, backend: "cpu", quant: "int8int8", kvSessions: 4})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/messages", srv.handleMessages)
	ts := httptest.NewServer(mux)
	defer ts.Close()
	lm := srv.models["test-model"]

	// A long generation we will abort after the first delta arrives.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	body := `{"model":"test-model","max_tokens":512,"temperature":0,"stream":true,
		"messages":[{"role":"user","content":"Count slowly from 1 to 200, one number per line."}]}`
	req, _ := http.NewRequestWithContext(ctx, "POST", ts.URL+"/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST: %v", err)
	}

	// Read until the first text_delta, then disconnect.
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		if strings.Contains(sc.Text(), "text_delta") {
			break
		}
	}
	cancel()
	resp.Body.Close()

	// The decode mutex (held for the duration of a generation) must free promptly
	// once the request context is cancelled.
	deadline := time.Now().Add(5 * time.Second)
	for {
		if lm.mu.TryLock() {
			lm.mu.Unlock()
			return // free — generation stopped
		}
		if time.Now().After(deadline) {
			t.Fatal("model mutex still held 5s after client abort — generation did not stop")
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// TestServe_anthropic_toolStreaming asserts the streaming tool path emits a
// tool_use content block with an input_json_delta carrying valid arguments, and
// closes with stop_reason "tool_use". tool_choice "any" forces the call (lone
// tool ⇒ constrained) so the path is deterministic. Gated on GOINFER_SERVE_MODEL.
func TestServe_anthropic_toolStreaming(t *testing.T) {
	ts := newAnthropicTestServer(t)
	defer ts.Close()

	body := `{"model":"test-model","max_tokens":120,"temperature":0,"stream":true,
		"tools":[{"name":"get_weather","description":"Get the current weather for a city.",
			"input_schema":{"type":"object","additionalProperties":false,
				"properties":{"location":{"type":"string"}},"required":["location"]}}],
		"messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],
		"tool_choice":{"type":"any"}}`
	r, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer r.Body.Close()
	events := readAnthropicSSE(t, bufio.NewScanner(r.Body))

	var sawToolStart, sawJSONDelta, sawToolUseStop bool
	var partial strings.Builder
	for _, e := range events {
		switch e.event {
		case "content_block_start":
			if strings.Contains(e.data, `"type":"tool_use"`) && strings.Contains(e.data, `"name":"get_weather"`) {
				sawToolStart = true
			}
		case "content_block_delta":
			var d struct {
				Delta struct {
					Type        string `json:"type"`
					PartialJSON string `json:"partial_json"`
				} `json:"delta"`
			}
			json.Unmarshal([]byte(e.data), &d)
			if d.Delta.Type == "input_json_delta" {
				sawJSONDelta = true
				partial.WriteString(d.Delta.PartialJSON)
			}
		case "message_delta":
			if strings.Contains(e.data, `"stop_reason":"tool_use"`) {
				sawToolUseStop = true
			}
		}
	}
	if !sawToolStart || !sawJSONDelta || !sawToolUseStop {
		t.Fatalf("missing tool stream pieces: start=%v jsonDelta=%v toolUseStop=%v", sawToolStart, sawJSONDelta, sawToolUseStop)
	}
	var args struct {
		Location string `json:"location"`
	}
	if err := json.Unmarshal([]byte(partial.String()), &args); err != nil || args.Location == "" {
		t.Errorf("input_json_delta not valid args: %q (%v)", partial.String(), err)
	}
	t.Logf("streamed tool args: %s", partial.String())
}

// subsequence reports whether want appears in order as an ordered subsequence.
func subsequence(order, want []string) bool {
	i := 0
	for _, e := range order {
		if i < len(want) && e == want[i] {
			i++
		}
	}
	return i == len(want)
}

// TestServe_anthropic_tools drives the llama.cpp blog get_weather example: a
// tool_use block out, then a replayed tool_use + tool_result producing grounded
// text. Gated on GOINFER_SERVE_MODEL.
func TestServe_anthropic_tools(t *testing.T) {
	ts := newAnthropicTestServer(t)
	defer ts.Close()

	tools := `"tools":[{"name":"get_weather","description":"Get the current weather for a city.",
		"input_schema":{"type":"object","additionalProperties":false,
			"properties":{"location":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]}},
			"required":["location"]}}]`
	// tool_choice "any" forces a call (lone tool ⇒ constrained), so the tool_use
	// output path is deterministic to assert. "auto" is exercised by the replay.
	body := `{"model":"test-model","max_tokens":120,"temperature":0,` + tools + `,
		"messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],
		"tool_choice":{"type":"any"}}`
	r, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer r.Body.Close()
	var out anthropicResp
	if err := json.NewDecoder(r.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if out.StopReason != "tool_use" {
		t.Fatalf("stop_reason = %q, want tool_use (content %+v)", out.StopReason, out.Content)
	}
	var call struct {
		id, name, location string
	}
	for _, b := range out.Content {
		if b.Type == "tool_use" {
			call.id, call.name = b.ID, b.Name
			var in struct {
				Location string `json:"location"`
			}
			json.Unmarshal(b.Input, &in)
			call.location = in.Location
		}
	}
	if call.name != "get_weather" || call.location == "" {
		t.Fatalf("tool_use block wrong: %+v", out.Content)
	}
	if !strings.HasPrefix(call.id, "toolu_") && call.id == "" {
		t.Errorf("tool_use id = %q", call.id)
	}
	t.Logf("tool_use: %s(location=%q) id=%s", call.name, call.location, call.id)

	// Replay the call + a result → expect a grounded text answer (end_turn).
	replay := `{"model":"test-model","max_tokens":80,"temperature":0,` + tools + `,
		"messages":[
			{"role":"user","content":"What is the weather in Paris? Use the tool."},
			{"role":"assistant","content":[{"type":"tool_use","id":"` + call.id + `","name":"get_weather","input":{"location":"Paris"}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"` + call.id + `","content":"18 degrees celsius and sunny"}]}
		]}`
	r2, err := http.Post(ts.URL+"/v1/messages", "application/json", strings.NewReader(replay))
	if err != nil {
		t.Fatalf("replay POST: %v", err)
	}
	defer r2.Body.Close()
	var out2 anthropicResp
	json.NewDecoder(r2.Body).Decode(&out2)
	if len(out2.Content) == 0 || out2.Content[0].Type != "text" || out2.Content[0].Text == "" {
		t.Errorf("grounded answer missing: %+v", out2.Content)
	}
	t.Logf("grounded answer: %q (stop=%s)", out2.Content[0].Text, out2.StopReason)
}
