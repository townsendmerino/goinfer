package serveapp

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/modelload"
)

// TestKV8aQ_gate1Scopes posts recorded /v1/chat/completions bodies, in file order, to the chat route of a scripted server that has a
// real model's tokenizer and template and keeps recurrent snapshots, and writes each request's prompt ids with the KV scope the
// route hands the decoder (Keep, from loadedModel.recurrentKeep) and the system-and-tools boundary (loadedModel.prefixBoundaryOf).
// It is KV8a-Q gate 1's measurement tool (docs/measurements/kv8a-q-gate1-2026-10/README.md), not a check: it skips unless
// KV8AQ_BODIES is set.
//
//	KV8AQ_BODIES=<dir of bodies> KV8AQ_MODEL=<a .gguf, a .giw or a model directory> KV8AQ_OUT=<out.jsonl> \
//	  go test -count=1 -run '^TestKV8aQ_gate1Scopes$' -v ./internal/serveapp/
func TestKV8aQ_gate1Scopes(t *testing.T) {
	dir, model, out := os.Getenv("KV8AQ_BODIES"), os.Getenv("KV8AQ_MODEL"), os.Getenv("KV8AQ_OUT")
	if dir == "" {
		t.Skip("set KV8AQ_BODIES, KV8AQ_MODEL and KV8AQ_OUT")
	}
	tk, err := modelload.Tokenizer(model)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatalf("chat template: %v", err)
	}
	reply, err := tk.Encode("Sure.", false)
	if err != nil {
		t.Fatal(err)
	}
	s := &server{}
	lm := &loadedModel{tk: tk, name: "m", scripted: reply, scriptedRecurrent: true}
	lm.tmpl = s.cfg.tuneTemplate(tmpl)
	var prompts [][]int
	var scopes []kvScope
	lm.onScripted = func(p []int) { prompts = append(prompts, slices.Clone(p)) }
	lm.onScriptedScope = func(sc kvScope) { scopes = append(scopes, sc) }
	s.models = map[string]*loadedModel{"m": lm}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.handleChat)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	files, _ := filepath.Glob(filepath.Join(dir, "*.json"))
	sort.Strings(files)
	of, err := os.Create(out)
	if err != nil {
		t.Fatal(err)
	}
	defer of.Close()
	enc := json.NewEncoder(of)
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		var rec struct {
			Path string `json:"path"`
			Body string `json:"body"`
		}
		if err := json.Unmarshal(raw, &rec); err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		if rec.Path != "/v1/chat/completions" {
			t.Fatalf("%s: route %s, want /v1/chat/completions", f, rec.Path)
		}
		// Values stay raw: decoding them into maps would reorder a tool schema's keys, which the route renders as sent.
		var fields map[string]json.RawMessage
		if err := json.Unmarshal([]byte(rec.Body), &fields); err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		body := map[string]any{"model": "m", "max_tokens": 8}
		for k, v := range fields {
			if k != "model" && k != "max_tokens" && k != "stream" {
				body[k] = v
			}
		}
		n := len(prompts)
		postScripted(t, ts.URL+"/v1/chat/completions", body)
		if len(prompts) != n+1 || len(scopes) != n+1 {
			t.Fatalf("%s: %d prompts and %d scopes captured, want one each", f, len(prompts)-n, len(scopes)-n)
		}
		ids := prompts[n]
		pieces := make([]string, len(ids))
		for i, id := range ids {
			pieces[i] = lm.tokenText(id)
		}
		line := map[string]any{"file": f, "n": len(ids), "ids": ids, "pieces": pieces, "keep": scopes[n].Keep,
			"boundary": kv8aqBoundary(t, s, lm, rec.Body, ids)}
		if err := enc.Encode(line); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("%d bodies posted", len(files))
}

// kv8aqBoundary is the system-and-tools boundary the chat route would compute for body with a model that takes prefix snapshots
// (loadedModel.prefixBoundary less its model check, which a scripted model fails), from the same render as the route's.
func kv8aqBoundary(t *testing.T, s *server, lm *loadedModel, body string, ids []int) int {
	t.Helper()
	var req chatReq
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatal(err)
	}
	req.Tools, _ = functionTools(req.Tools)
	th, err := s.resolveThink(req.think())
	if err != nil {
		t.Fatal(err)
	}
	system, turns := messagesToTurns(req.Messages)
	tools := make([]chat.Tool, len(req.Tools))
	for i, tl := range req.Tools {
		tools[i] = chat.Tool{Name: tl.Function.Name, Description: tl.Function.Description, Parameters: tl.Function.Parameters}
	}
	if strings.TrimSpace(toolChoiceMode(req.ToolChoice)) == "none" {
		tools = nil
	}
	return lm.prefixBoundaryOf(lm.templateFor(th), system, turns, tools, ids)
}
