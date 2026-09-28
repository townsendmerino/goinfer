package serveapp

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/loadflags"
)

// goinfer_confidence is refused wherever it would be dropped: on a route that does not write it back, and without a
// JSON Schema response_format (the schema is what says each field's kind). CI-runnable: the refusals precede anything
// that needs a tokenizer.
func TestConfidence_refusals(t *testing.T) {
	_, lm := tinyServed(t)
	schema := &respFormat{Type: "json_schema"}
	schema.JSONSchema = &struct {
		Name   string          `json:"name"`
		Schema json.RawMessage `json:"schema"`
	}{Name: "x", Schema: json.RawMessage(`{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}`)}
	for _, c := range []struct {
		name string
		sm   sampling
		want string
	}{
		{"a route that does not write it back", sampling{Confidence: true, ResponseFormat: schema}, "supported on"},
		{"no response_format", sampling{Confidence: true, confidenceOK: true}, "needs response_format"},
		{"json_object", sampling{Confidence: true, confidenceOK: true, ResponseFormat: &respFormat{Type: "json_object"}}, "needs response_format"},
	} {
		if _, err := lm.prepare(c.sm, []int{1, 2, 3}, false); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want an error containing %q", c.name, err, c.want)
		}
	}
}

func TestConfResult_payload(t *testing.T) {
	var c confResult
	if b, _ := json.Marshal(c.payload()); string(b) != "[]" {
		t.Errorf("no qualifying field: %s, want []", b)
	}
}

// End to end on a real model (the local Qwen 0.5B; skips without it): a json_schema request with goinfer_confidence
// gets a record per enum / boolean / integer field and none for the string; the same request without the flag has
// no goinfer_confidence key at all; streaming carries the records as one event; a tools request with the flag is a
// 400.
func TestConfidence_endToEnd(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no local checkpoint at %s (an untracked symlink): %v", path, err)
	}
	// DirectLoad: the default would transcode a 0.5 GB sidecar .giw next to the testdata symlink.
	srv, err := newServer(config{models: modelFlag{{name: "m", path: path}}, load: loadflags.Flags{Backend: "cpu", Quant: "int4", DirectLoad: true}, kvSessions: 4})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
	ts := httptest.NewServer(mux)
	defer ts.Close()
	const req = `{"model":"m","max_tokens":120,"temperature":0,%s
		"messages":[{"role":"user","content":"Return only a JSON object with the keys category (billing, technical or shipping), urgent (true or false), orders (an integer: how many order numbers the ticket mentions) and name (who signed it). Ticket: I was charged twice for order #123, please refund me. - Ann"}],
		"response_format":{"type":"json_schema","json_schema":{"name":"t","schema":
			{"type":"object","additionalProperties":false,
			 "properties":{"category":{"enum":["billing","technical","shipping"]},"urgent":{"type":"boolean"},
			               "orders":{"type":"integer"},"name":{"type":"string"}},
			 "required":["category","urgent","orders","name"]}}}}`
	post := func(body string) map[string]json.RawMessage {
		t.Helper()
		resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Fatalf("status %d", resp.StatusCode)
		}
		var out map[string]json.RawMessage
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			t.Fatal(err)
		}
		return out
	}
	with := post(strings.Replace(req, "%s", `"goinfer_confidence":true,`, 1))
	var fields []struct {
		Path, Kind   string
		Confidence   float64
		Distribution map[string]float64
		FreeTokens   int  `json:"free_tokens"`
		Calibrated   bool `json:"calibrated"`
	}
	if err := json.Unmarshal(with["goinfer_confidence"], &fields); err != nil {
		t.Fatalf("goinfer_confidence %s: %v", with["goinfer_confidence"], err)
	}
	kinds := map[string]string{}
	for _, f := range fields {
		kinds[f.Path] = f.Kind
		if !(f.Confidence > 0 && f.Confidence <= 1) || f.FreeTokens < 1 || f.Calibrated {
			t.Errorf("%s: confidence %v free %d calibrated %v", f.Path, f.Confidence, f.FreeTokens, f.Calibrated)
		}
	}
	for path, kind := range map[string]string{"category": "enum", "urgent": "boolean", "orders": "integer"} {
		if kinds[path] != kind {
			t.Errorf("field %s: kind %q, want %q (all: %v)", path, kinds[path], kind, kinds)
		}
	}
	if _, ok := kinds["name"]; ok {
		t.Error("a string field was reported; C0 parked strings")
	}
	t.Logf("content %s\nconfidence %s", with["choices"], with["goinfer_confidence"])

	without := post(strings.Replace(req, "%s", "", 1))
	if _, ok := without["goinfer_confidence"]; ok {
		t.Error("a request without goinfer_confidence got the field")
	}
	if string(without["choices"]) != string(with["choices"]) {
		t.Errorf("the flag changed the answer:\nwith    %s\nwithout %s", with["choices"], without["choices"])
	}

	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(strings.Replace(req, "%s", `"goinfer_confidence":true,"stream":true,`, 1)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	events := 0
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		if line := sc.Text(); strings.HasPrefix(line, "data: ") && strings.Contains(line, `"goinfer_confidence"`) {
			events++
		}
	}
	if events != 1 {
		t.Errorf("stream carried %d goinfer_confidence events, want 1", events)
	}

	tools := `{"model":"m","max_tokens":8,"goinfer_confidence":true,"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}}]}`
	r2, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(tools))
	if err != nil {
		t.Fatal(err)
	}
	r2.Body.Close()
	if r2.StatusCode != http.StatusBadRequest {
		t.Errorf("tools + goinfer_confidence: status %d, want 400", r2.StatusCode)
	}
}
