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
	if i, hit := firstStop("hello END world", []string{"END"}); !hit || i != 6 {
		t.Errorf("got (%d,%v), want (6,true)", i, hit)
	}
	if i, hit := firstStop("abc STOP xyz END", []string{"END", "STOP"}); !hit || i != 4 {
		t.Errorf("earliest stop: got (%d,%v), want (4,true)", i, hit)
	}
	if _, hit := firstStop("no stop here", []string{"END"}); hit {
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
	s := &server{}
	f := func(v float64) *float64 { return &v }
	i := func(v int) *int { return &v }
	gr, err := s.prepare(sampling{
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
	def, _ := s.prepare(sampling{}, nil)
	if def.sp.Temperature != 1.0 || def.maxTokens != defaultMaxTokens || def.sp.TopP != 0 {
		t.Errorf("defaults wrong: temp %v max %d topP %v", def.sp.Temperature, def.maxTokens, def.sp.TopP)
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

// TestServe_integration exercises the real HTTP path end to end against a loaded
// model. Gated on GOINFER_SERVE_MODEL (a .gguf path) so it skips in normal CI.
//
//	GOINFER_SERVE_MODEL=~/models/gemma-4-E2B_q4_0-it.gguf go test ./cmd/serve -run Integration -v
func TestServe_integration(t *testing.T) {
	path := os.Getenv("GOINFER_SERVE_MODEL")
	if path == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf> to run the serve integration test")
	}
	srv, err := newServer(path, "cpu", "int8int8", "test-model")
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
	srv, err := newServer(path, "cpu", "int8int8", "test-model")
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
