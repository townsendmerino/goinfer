package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestParseStop(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{`"END"`, []string{"END"}},
		{`["a","b"]`, []string{"a", "b"}},
		{`""`, nil},
		{``, nil},
	}
	for _, c := range cases {
		got := parseStop(json.RawMessage(c.in))
		if strings.Join(got, ",") != strings.Join(c.want, ",") {
			t.Errorf("parseStop(%s) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestFirstStop(t *testing.T) {
	if i, which, hit := firstStop("hello END world", []string{"END"}); !hit || i != 6 || which != "END" {
		t.Errorf("got (%d,%q,%v), want (6,END,true)", i, which, hit)
	}
	if i, which, hit := firstStop("abc STOP xyz END", []string{"END", "STOP"}); !hit || i != 4 || which != "STOP" {
		t.Errorf("earliest stop: got (%d,%q,%v), want (4,STOP,true)", i, which, hit)
	}
	if _, _, hit := firstStop("no stop here", []string{"END"}); hit {
		t.Error("false positive stop")
	}
}

func TestCompleteUTF8(t *testing.T) {
	full := "héllo" // é is 2 bytes
	if completeUTF8(full) != len(full) {
		t.Errorf("complete string held back: %d/%d", completeUTF8(full), len(full))
	}
	// A trailing partial 2-byte sequence (first byte of é) must be held back.
	partial := "hello\xc3"
	if got := completeUTF8(partial); got != 5 {
		t.Errorf("partial: got %d, want 5 (hold back the lone lead byte)", got)
	}
}

func TestGrammarFor(t *testing.T) {
	if g, err := grammarFor(nil); g != nil || err != nil {
		t.Errorf("nil format: got (%v,%v)", g, err)
	}
	if g, err := grammarFor(&respFormat{Type: "text"}); g != nil || err != nil {
		t.Errorf("text: got (%v,%v)", g, err)
	}
	if g, err := grammarFor(&respFormat{Type: "json_object"}); g == nil || err != nil {
		t.Errorf("json_object: got (%v,%v)", g, err)
	}
	rf := &respFormat{Type: "json_schema"}
	rf.JSONSchema = &struct {
		Name   string          `json:"name"`
		Schema json.RawMessage `json:"schema"`
	}{Name: "p", Schema: json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}},"required":["x"],"additionalProperties":false}`)}
	if g, err := grammarFor(rf); g == nil || err != nil {
		t.Errorf("json_schema: got (%v,%v)", g, err)
	}
	if _, err := grammarFor(&respFormat{Type: "json_schema"}); err == nil {
		t.Error("json_schema without schema should error")
	}
	if _, err := grammarFor(&respFormat{Type: "bogus"}); err == nil {
		t.Error("unknown response_format type should error")
	}
}

// prepare's sampling translation (no model needed when there's no response_format).
func TestPrepare_sampling(t *testing.T) {
	lm := &loadedModel{}
	f := func(v float64) *float64 { return &v }
	i := func(v int) *int { return &v }
	gr, err := lm.prepare(sampling{
		Temperature: f(0.3), TopP: f(0.9), TopK: i(40), MaxTokens: i(128),
		FrequencyPenalty: f(0.5), PresencePenalty: f(0.2),
	}, []int{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	if gr.sp.Temperature != 0.3 || gr.sp.TopP != 0.9 || gr.sp.TopK != 40 || gr.maxTokens != 128 {
		t.Errorf("sampling mismatch: %+v (max %d)", gr.sp, gr.maxTokens)
	}
	if gr.sp.FrequencyPenalty != 0.5 || gr.sp.PresencePenalty != 0.2 {
		t.Errorf("penalties not mapped: %+v", gr.sp)
	}
	// Defaults: no fields → temperature 1, max_tokens default, top_p disabled (0).
	def, _ := lm.prepare(sampling{}, nil)
	if def.sp.Temperature != 1.0 || def.maxTokens != defaultMaxTokens || def.sp.TopP != 0 {
		t.Errorf("defaults wrong: temp %v max %d topP %v", def.sp.Temperature, def.maxTokens, def.sp.TopP)
	}
}

// prepare rejects out-of-range max_tokens up front so a bad value never reaches
// NewCache (negative → makeslice panic on the VL bare goroutine, C-19; huge → OOM
// throw that kills the server, C-18).
func TestPrepare_maxTokensBounds(t *testing.T) {
	lm := &loadedModel{}
	i := func(v int) *int { return &v }
	for _, tc := range []struct {
		name string
		mt   int
		ok   bool
	}{
		{"negative", -1000000, false},
		{"zero", 0, false},
		{"one", 1, true},
		{"ceiling", maxOutputTokensCeiling, true},
		{"over-ceiling", maxOutputTokensCeiling + 1, false},
	} {
		_, err := lm.prepare(sampling{MaxTokens: i(tc.mt)}, []int{1, 2, 3})
		if tc.ok && err != nil {
			t.Errorf("%s: max_tokens=%d unexpectedly rejected: %v", tc.name, tc.mt, err)
		}
		if !tc.ok && err == nil {
			t.Errorf("%s: max_tokens=%d should have been rejected", tc.name, tc.mt)
		}
	}
}

func TestChatChunkShape(t *testing.T) {
	fin := "stop"
	b, _ := json.Marshal(chatChunk("id1", 123, "m", delta{Content: "hi"}, &fin))
	var got map[string]any
	_ = json.Unmarshal(b, &got)
	if got["object"] != "chat.completion.chunk" {
		t.Errorf("object = %v", got["object"])
	}
	ch := got["choices"].([]any)[0].(map[string]any)
	if ch["finish_reason"] != "stop" {
		t.Errorf("finish_reason = %v", ch["finish_reason"])
	}
	if ch["delta"].(map[string]any)["content"] != "hi" {
		t.Errorf("delta.content = %v", ch["delta"])
	}
	// A nil finish_reason must serialize as JSON null, not be omitted.
	b2, _ := json.Marshal(chatChunk("id1", 123, "m", delta{Content: "x"}, nil))
	if !bytes.Contains(b2, []byte(`"finish_reason":null`)) {
		t.Errorf("nil finish_reason not null: %s", b2)
	}
}

// TestServe_multiModel is the Inc2 gate: two generative models in one process,
// requests routed on the OpenAI `model` field, /v1/models listing both, unknown
// model → 404, and concurrent cross-model requests running in parallel (per-model
// mutex). Gated on two .gguf paths so it skips in normal CI.
//
//	GOINFER_SERVE_MODEL=~/models/a.gguf GOINFER_SERVE_MODEL2=~/models/b.gguf \
//	  go test ./cmd/serve -run TestServe_multiModel -v
func TestServe_multiModel(t *testing.T) {
	p1, p2 := os.Getenv("GOINFER_SERVE_MODEL"), os.Getenv("GOINFER_SERVE_MODEL2")
	if p1 == "" || p2 == "" {
		t.Skip("set GOINFER_SERVE_MODEL + GOINFER_SERVE_MODEL2 (two .gguf) for the multi-model test")
	}
	srv, err := newServer(config{
		models:  modelFlag{{name: "m1", path: p1}, {name: "m2", path: p2}},
		backend: "cpu", quant: "int8int8", kvSessions: 2,
	})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
	mux.HandleFunc("GET /v1/models", srv.handleModels)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	// /v1/models lists both.
	resp, _ := http.Get(ts.URL + "/v1/models")
	var ml struct {
		Data []struct{ ID string } `json:"data"`
	}
	json.NewDecoder(resp.Body).Decode(&ml)
	resp.Body.Close()
	got := map[string]bool{}
	for _, m := range ml.Data {
		got[m.ID] = true
	}
	if !got["m1"] || !got["m2"] {
		t.Errorf("/v1/models = %v, want m1 + m2", ml.Data)
	}

	// chat returns the responded model id + status + latency.
	chat := func(model string) (int, string, time.Duration) {
		body := `{"model":"` + model + `","max_tokens":24,"temperature":0,"messages":[{"role":"user","content":"Write a one-line hello in Go."}]}`
		t0 := time.Now()
		r, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatalf("POST %s: %v", model, err)
		}
		defer r.Body.Close()
		var out struct {
			Model string `json:"model"`
		}
		json.NewDecoder(r.Body).Decode(&out)
		return r.StatusCode, out.Model, time.Since(t0)
	}

	// Routing: each request's response model field matches the requested model.
	for _, m := range []string{"m1", "m2"} {
		if code, model, _ := chat(m); code != 200 || model != m {
			t.Errorf("route %s: status %d model %q, want 200/%s", m, code, model, m)
		}
	}
	// Unknown model → OpenAI-shaped 404.
	if code, _, _ := chat("nope"); code != http.StatusNotFound {
		t.Errorf("unknown model: status %d, want 404", code)
	}

	// Parallelism: two concurrent requests to DIFFERENT models share no mutex, so
	// they overlap — cross-model wall time should be well under the sum of the two
	// serial times. (Soft: per-model mutexes guarantee it structurally; logged.)
	s1, s2 := func() time.Duration { _, _, d := chat("m1"); return d }(), func() time.Duration { _, _, d := chat("m2"); return d }()
	done := make(chan struct{}, 2)
	t0 := time.Now()
	go func() { chat("m1"); done <- struct{}{} }()
	go func() { chat("m2"); done <- struct{}{} }()
	<-done
	<-done
	wall := time.Since(t0)
	t.Logf("serial m1=%v m2=%v (sum %v) | concurrent cross-model wall=%v", s1, s2, s1+s2, wall)
	if wall >= s1+s2 {
		t.Errorf("cross-model requests did not overlap: wall %v >= serial sum %v", wall, s1+s2)
	}
}

// TestServe_integration exercises the real HTTP path end to end against a loaded
// model. Gated on GOINFER_SERVE_MODEL (a .gguf path) so it skips in normal CI.
//
//	GOINFER_SERVE_MODEL=~/models/gemma-4-E2B_q4_0-it.gguf go test ./cmd/serve -run Integration -v
func TestServe_integration(t *testing.T) {
	path := os.Getenv("GOINFER_SERVE_MODEL")
	if path == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf> to run the serve integration test")
	}
	srv, err := newServer(config{models: modelFlag{{name: "test-model", path: path}}, backend: "cpu", quant: "int8int8", kvSessions: 4})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
	mux.HandleFunc("GET /v1/models", srv.handleModels)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	// /v1/models
	resp, err := http.Get(ts.URL + "/v1/models")
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("/v1/models: %v code %v", err, resp.StatusCode)
	}

	// /v1/chat/completions with a JSON-Schema response_format: the content must
	// be valid JSON conforming to the schema.
	// The prompt must ask for JSON: under a constraint, a model that wants to write
	// prose can otherwise emit only schema-valid whitespace (OpenAI's json mode
	// requires "json" in the prompt for the same reason).
	body := `{"model":"test-model","max_tokens":100,"temperature":0,
		"messages":[{"role":"user","content":"Return a JSON object for a person with a name and an integer age."}],
		"response_format":{"type":"json_schema","json_schema":{"name":"person","schema":
			{"type":"object","additionalProperties":false,
			 "properties":{"name":{"type":"string"},"age":{"type":"integer"}},
			 "required":["name","age"]}}}}`
	resp, err = http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
	var out struct {
		Choices []struct {
			Message      struct{ Content string } `json:"message"`
			FinishReason string                   `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	content := out.Choices[0].Message.Content
	var person struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}
	dec := json.NewDecoder(strings.NewReader(content))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&person); err != nil {
		t.Fatalf("constrained content is not the schema's JSON: %q (%v)", content, err)
	}
	t.Logf("constrained chat content: %s", content)

	// Streaming smoke: SSE chunks + [DONE].
	resp, err = http.Post(ts.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"test-model","max_tokens":16,"temperature":0,"stream":true,
			"messages":[{"role":"user","content":"Say hi."}]}`))
	if err != nil {
		t.Fatalf("stream POST: %v", err)
	}
	defer resp.Body.Close()
	var chunks, done int
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		if strings.TrimPrefix(line, "data: ") == "[DONE]" {
			done++
			continue
		}
		chunks++
	}
	if chunks == 0 || done != 1 {
		t.Errorf("stream: chunks=%d done=%d (want chunks>0, done=1)", chunks, done)
	}
}

// TestServe_tools_integration drives a tool call end to end: one tool (so the
// call is constrained), a weather question, and we assert the response is a
// tool_calls finish with valid arguments. Gated on GOINFER_SERVE_MODEL.
func TestServe_tools_integration(t *testing.T) {
	path := os.Getenv("GOINFER_SERVE_MODEL")
	if path == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf> to run the serve tool test")
	}
	srv, err := newServer(config{models: modelFlag{{name: "test-model", path: path}}, backend: "cpu", quant: "int8int8", kvSessions: 4})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	body := `{"model":"test-model","max_tokens":120,"temperature":0,
		"messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],
		"tools":[{"type":"function","function":{"name":"get_weather",
			"description":"Get the current weather for a city.",
			"parameters":{"type":"object","additionalProperties":false,
				"properties":{"location":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]}},
				"required":["location"]}}}],
		"tool_choice":"auto"}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	var out struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				ToolCalls []struct {
					Function struct{ Name, Arguments string } `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	ch := out.Choices[0]
	if ch.FinishReason != "tool_calls" || len(ch.Message.ToolCalls) != 1 {
		t.Fatalf("expected a tool call, got finish=%q calls=%d", ch.FinishReason, len(ch.Message.ToolCalls))
	}
	tc := ch.Message.ToolCalls[0]
	if tc.Function.Name != "get_weather" {
		t.Errorf("tool name = %q, want get_weather", tc.Function.Name)
	}
	var args struct {
		Location string `json:"location"`
	}
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil || args.Location == "" {
		t.Errorf("arguments not valid / missing location: %q (%v)", tc.Function.Arguments, err)
	}
	t.Logf("tool call: %s(%s)", tc.Function.Name, tc.Function.Arguments)
}

// TestServe_grammarSpecLossless verifies the grammar-fused speculative path wired into
// drive (--spec on a constrained greedy request) produces output byte-identical to plain
// constrained decode — the losslessness guarantee, end-to-end through the HTTP layer.
// Set GOINFER_SERVE_MODEL=<.gguf>.
func TestServe_grammarSpecLossless(t *testing.T) {
	path := os.Getenv("GOINFER_SERVE_MODEL")
	if path == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf> to run the grammar-spec losslessness test")
	}
	const body = `{"model":"test-model","max_tokens":80,"temperature":0,
		"messages":[{"role":"user","content":"Return a JSON object for a person with a name and an integer age."}],
		"response_format":{"type":"json_schema","json_schema":{"name":"person","schema":
			{"type":"object","additionalProperties":false,
			 "properties":{"name":{"type":"string"},"age":{"type":"integer"}},
			 "required":["name","age"]}}}}`

	run := func(spec string) string {
		srv, err := newServer(config{models: modelFlag{{name: "test-model", path: path}}, backend: "cpu", quant: "int8int8", kvSessions: 4, spec: spec})
		if err != nil {
			t.Fatalf("newServer(spec=%q): %v", spec, err)
		}
		mux := http.NewServeMux()
		mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
		ts := httptest.NewServer(mux)
		defer ts.Close()
		resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatalf("POST(spec=%q): %v", spec, err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Fatalf("spec=%q status %d", spec, resp.StatusCode)
		}
		var out struct {
			Choices []struct {
				Message struct{ Content string } `json:"message"`
			} `json:"choices"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			t.Fatalf("decode(spec=%q): %v", spec, err)
		}
		return out.Choices[0].Message.Content
	}

	plain := run("")
	spec := run("ngram") // constrained greedy ⇒ grammar-fused spec path
	if plain != spec {
		t.Fatalf("grammar-spec not lossless:\n plain=%q\n spec =%q", plain, spec)
	}
	t.Logf("grammar-spec lossless OK: %q", spec)
}

// TestClampMaxTokens_contextWindow is the C-18 gate: max_tokens is bounded by the model's context so
// KV preallocation (NewCache is sized len(prompt)+max_tokens) can't be driven to OOM by a request at
// the server ceiling against a small-context model. It only ever shrinks.
func TestClampMaxTokens_contextWindow(t *testing.T) {
	for _, tc := range []struct {
		name                      string
		maxTokens, promptLen, ctx int
		want                      int
	}{
		{"over-ceiling clamps to room", 131072, 100, 4096, 3996},
		{"within context untouched", 500, 100, 4096, 500},
		{"exact fit untouched", 4000, 96, 4096, 4000},
		{"unknown ctx (0) untouched", 131072, 100, 0, 131072},
		{"prompt at ctx left to C-20", 500, 4096, 4096, 500},
		{"prompt over ctx left to C-20", 500, 5000, 4096, 500},
		{"room exactly equals request", 3996, 100, 4096, 3996},
	} {
		if got := clampMaxTokens(tc.maxTokens, tc.promptLen, tc.ctx); got != tc.want {
			t.Errorf("%s: clampMaxTokens(%d, %d, %d) = %d, want %d",
				tc.name, tc.maxTokens, tc.promptLen, tc.ctx, got, tc.want)
		}
	}
}

// TestContextLengthError is the C-20 gate: a prompt that fills or exceeds the model's context is
// rejected (context_length_exceeded) rather than OOM-killing the server or decoding out-of-range RoPE.
func TestContextLengthError(t *testing.T) {
	for _, tc := range []struct {
		name           string
		promptLen, ctx int
		wantErr        bool
	}{
		{"within context", 100, 4096, false},
		{"one below context", 4095, 4096, false},
		{"exactly at context (no room)", 4096, 4096, true},
		{"past context", 5000, 4096, true},
		{"1M-token body", 1_000_000, 32768, true},
		{"unknown ctx (0) never rejects", 1_000_000, 0, false},
	} {
		err := contextLengthError(tc.promptLen, tc.ctx)
		if (err != nil) != tc.wantErr {
			t.Errorf("%s: contextLengthError(%d, %d) err=%v, wantErr=%v", tc.name, tc.promptLen, tc.ctx, err, tc.wantErr)
		}
		if err != nil && !strings.Contains(err.Error(), "context_length_exceeded") {
			t.Errorf("%s: error %q should name context_length_exceeded", tc.name, err.Error())
		}
	}
}

// TestPrepare_topPAndSeed gates M-02 (top_p==0 is greedy, out-of-range rejected) and M-03 (an omitted
// seed varies, a supplied seed is honored).
func TestPrepare_topPAndSeed(t *testing.T) {
	lm := &loadedModel{}
	f := func(v float64) *float64 { return &v }
	// M-02: top_p==0 → greedy (Temperature 0), not a full-vocab draw.
	gr, err := lm.prepare(sampling{TopP: f(0)}, []int{1})
	if err != nil {
		t.Fatalf("top_p=0: %v", err)
	}
	if gr.sp.Temperature != 0 || gr.sp.TopP != 0 {
		t.Errorf("top_p=0 → Temperature %v TopP %v, want greedy (0,0)", gr.sp.Temperature, gr.sp.TopP)
	}
	// M-02: 0 < top_p < 1 sets the nucleus; out-of-range is a 400.
	if gr, _ := lm.prepare(sampling{TopP: f(0.9)}, []int{1}); gr.sp.TopP != 0.9 {
		t.Errorf("top_p=0.9 → TopP %v, want 0.9", gr.sp.TopP)
	}
	for _, bad := range []float64{-0.1, 1.5} {
		if _, err := lm.prepare(sampling{TopP: f(bad)}, []int{1}); err == nil {
			t.Errorf("top_p=%v not rejected", bad)
		}
	}
	// M-03: omitted seed varies (two calls differ w.h.p.); a supplied seed is honored.
	a, _ := lm.prepare(sampling{}, []int{1})
	b, _ := lm.prepare(sampling{}, []int{1})
	if a.sp.Seed == 0 && b.sp.Seed == 0 {
		t.Error("omitted seed is still deterministic 0 (M-03)")
	}
	sd := int64(42)
	if gr, _ := lm.prepare(sampling{Seed: &sd}, []int{1}); gr.sp.Seed != 42 {
		t.Errorf("supplied seed 42 → %d", gr.sp.Seed)
	}
}

// TestConstrainForcedTool_M05 gates M-05: a NAMED tool_choice whose family cannot be
// constrained (gemma4 has a bespoke parse-only call form, ToolCallWrapper ok=false) must
// return an error the handler renders as a 400 — not silently decode unconstrained. The
// lone-tool convenience (namedForce=false) never errors. A constrainable family (chatml)
// wires the masker and returns nil.
func TestConstrainForcedTool_M05(t *testing.T) {
	tool := &chat.Tool{Name: "get_weather", Parameters: []byte(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`)}
	// gemma4: SupportsTools true, ToolCallWrapper ok=false → unconstrainable.
	lmG := &loadedModel{tmpl: chat.Gemma4(), vocab: 32, tk: nil}
	// named force on an unconstrainable family → error (becomes 400).
	if err := constrainForcedTool(lmG, &genRequest{}, tool, true); err == nil {
		t.Error("named tool_choice on gemma4 (no constrainable form) should error, got nil (M-05)")
	}
	// lone-tool convenience on the same family → no error (optimization only).
	if err := constrainForcedTool(lmG, &genRequest{}, tool, false); err != nil {
		t.Errorf("lone-tool on gemma4 should not error, got %v", err)
	}
	// no forced tool at all → no error.
	if err := constrainForcedTool(lmG, &genRequest{}, nil, true); err != nil {
		t.Errorf("nil forced tool should not error, got %v", err)
	}
	// constrainable family (chatml): wires the masker, returns nil.
	lmC := &loadedModel{tmpl: chat.ChatML(), vocab: 32, tk: &tokenizer.Tokenizer{}}
	gr := &genRequest{}
	if err := constrainForcedTool(lmC, gr, tool, true); err != nil {
		t.Fatalf("chatml named force should succeed, got %v", err)
	}
	if gr.masker == nil || gr.sp.LogitProcessor == nil {
		t.Error("chatml named force did not wire the masker/LogitProcessor")
	}
}

// TestCompletionReqLogprobs_M06 gates M-06: /v1/completions must accept the legacy
// integer logprobs field (the standard SDK sends `logprobs: 5`) instead of failing to
// decode into a bool and 400ing with leaked Go struct/field names. The shadow *int field
// wins, so the embedded chat bool stays false (the expensive logprobs path never engages).
func TestCompletionReqLogprobs_M06(t *testing.T) {
	var req completionReq
	if err := json.Unmarshal([]byte(`{"model":"m","prompt":"hi","logprobs":5}`), &req); err != nil {
		t.Fatalf("integer logprobs must decode, got %v (M-06)", err)
	}
	if req.Logprobs == nil || *req.Logprobs != 5 {
		t.Errorf("req.Logprobs = %v, want *int 5", req.Logprobs)
	}
	if req.sampling.Logprobs {
		t.Error("embedded chat-bool Logprobs must stay false (shadowed) so the logprobs path is off")
	}
}

// TestEffectiveBudget_M04 gates M-04: finish_reason must be judged against the resident
// context-cap-clamped budget, not the requested max_tokens. When the decoder publishes a
// smaller Budget (prompt filled most of the KV cap), a generation that emits exactly Budget
// tokens is a truncation ("length"), not a clean "stop" — so effectiveBudget must return the
// clamped value. Paths that don't clamp (Budget 0, nil gen) fall back to the request.
func TestEffectiveBudget_M04(t *testing.T) {
	cases := []struct {
		name      string
		gen       *decoder.Generation
		requested int
		want      int
	}{
		{"clamped below request", &decoder.Generation{Budget: 96}, 512, 96},
		{"unset budget → request", &decoder.Generation{Budget: 0}, 512, 512},
		{"nil gen → request", nil, 512, 512},
		{"budget equals request", &decoder.Generation{Budget: 512}, 512, 512},
	}
	for _, c := range cases {
		if got := effectiveBudget(c.gen, c.requested); got != c.want {
			t.Errorf("%s: effectiveBudget = %d, want %d", c.name, got, c.want)
		}
	}
}
