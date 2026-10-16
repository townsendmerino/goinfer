package serveapp

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
)

// The fast alias (fast_alias.go) at tier 0, on the scripted Mellum2.1 server: a request under "<model>-fast" runs on the same
// loaded model with thinking off and nothing else different, its output limit included; a request under the model's own name is
// unchanged.

// fastRig is a scripted Mellum2.1 server whose model "m" generates reply, with the alias on or off (-fast-alias). prompts collects
// the prompt ids of every generation, in order.
type fastRig struct {
	ts      *httptest.Server
	s       *server
	lm      *loadedModel
	log     *bytes.Buffer
	prompts *[][]int
}

func newFastRig(t *testing.T, reply string, alias bool) *fastRig {
	t.Helper()
	tk := scriptTokenizer(t)
	ids, err := tk.Encode(reply, false)
	if err != nil {
		t.Fatal(err)
	}
	s := &server{cfg: config{fastAlias: alias}}
	var prompts [][]int
	lm := &loadedModel{tk: tk, name: "m", scripted: append([]int{}, ids...)}
	lm.onScripted = func(p []int) { prompts = append(prompts, append([]int(nil), p...)) }
	lm.tmpl = s.cfg.tuneTemplate(mellum21Template(t))
	s.models = map[string]*loadedModel{"m": lm}
	var log bytes.Buffer
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/responses", s.traceRequests(&log, s.handleResponses))
	mux.HandleFunc("POST /v1/chat/completions", s.traceRequests(&log, s.handleChat))
	mux.HandleFunc("POST /v1/messages", s.traceRequests(&log, s.handleMessages))
	mux.HandleFunc("POST /v1/messages/count_tokens", s.handleCountTokens)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return &fastRig{ts: ts, s: s, lm: lm, log: &log, prompts: &prompts}
}

// fastRoutes are the generation routes, each with a body for model m that asks for up to max tokens (0: no limit given), and the
// request's spelling of "thinking off" for the same model, and where the reply's token count is in the response.
var fastRoutes = []struct {
	path   string
	body   func(model string, max int) map[string]any
	off    map[string]any // the fields that turn thinking off on this route, merged into the body
	tokens func(resp map[string]any) int
}{
	{"/v1/chat/completions",
		func(m string, max int) map[string]any {
			b := map[string]any{"model": m, "messages": []any{map[string]any{"role": "user", "content": "Hi"}}}
			if max > 0 {
				b["max_tokens"] = max
			}
			return b
		},
		map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": false}},
		func(r map[string]any) int { return int(r["usage"].(map[string]any)["completion_tokens"].(float64)) }},
	{"/v1/responses",
		func(m string, max int) map[string]any {
			b := map[string]any{"model": m, "input": "Hi"}
			if max > 0 {
				b["max_output_tokens"] = max
			}
			return b
		},
		map[string]any{"reasoning": map[string]any{"effort": "none"}},
		func(r map[string]any) int { return int(r["usage"].(map[string]any)["output_tokens"].(float64)) }},
	{"/v1/messages",
		func(m string, max int) map[string]any {
			if max == 0 {
				max = 100000
			}
			return map[string]any{"model": m, "max_tokens": max, "messages": []any{map[string]any{"role": "user", "content": "Hi"}}}
		},
		map[string]any{"thinking": map[string]any{"type": "disabled"}},
		func(r map[string]any) int { return int(r["usage"].(map[string]any)["output_tokens"].(float64)) }},
}

func (f *fastRig) post(t *testing.T, path string, body map[string]any) (map[string]any, []byte) {
	t.Helper()
	b, _ := json.Marshal(body)
	resp, err := http.Post(f.ts.URL+path, "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("%s: %d %s", path, resp.StatusCode, raw)
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	return out, raw
}

func (f *fastRig) lastPrompt(t *testing.T) string {
	t.Helper()
	ps := *f.prompts
	s, err := f.lm.tk.Decode(ps[len(ps)-1])
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func withFields(body map[string]any, extra map[string]any) map[string]any {
	out := map[string]any{}
	for k, v := range body {
		out[k] = v
	}
	for k, v := range extra {
		out[k] = v
	}
	return out
}

// The alias reaches the same loaded model and renders its prompt with thinking off, exactly as the same request does under the
// model's own name with thinking turned off in that route's own spelling; under the model's name the default thinking stands.
// count_tokens counts the alias's prompt too.
func TestFastAlias_sameModelThinkingOff(t *testing.T) {
	f := newFastRig(t, "ok", true)
	const closed = "<think>\n\n</think>\n\n"
	for _, rt := range fastRoutes {
		t.Run(rt.path, func(t *testing.T) {
			f.post(t, rt.path, rt.body("m", 0))
			def := f.lastPrompt(t)
			f.post(t, rt.path, withFields(rt.body("m", 0), rt.off))
			off := f.lastPrompt(t)
			f.post(t, rt.path, rt.body("m-fast", 0))
			fast := f.lastPrompt(t)
			if strings.HasSuffix(def, closed) || !strings.HasSuffix(off, closed) {
				t.Fatalf("VOID: the template's default thinking is not distinguishable from off:\ndefault %q\noff     %q", def[max(0, len(def)-60):], off[max(0, len(off)-60):])
			}
			if fast != off {
				t.Errorf("the alias's prompt is not the thinking-off prompt:\nalias %q\noff   %q", fast[max(0, len(fast)-80):], off[max(0, len(off)-80):])
			}
			// A client that asks for thinking on the alias does not get it.
			on := withFields(rt.body("m-fast", 0), map[string]any{})
			switch rt.path {
			case "/v1/chat/completions":
				on["chat_template_kwargs"], on["reasoning_effort"], on["thinking_token_budget"] = map[string]any{"enable_thinking": true}, "high", 512
			case "/v1/messages":
				on["thinking"] = map[string]any{"type": "enabled", "budget_tokens": 512}
			}
			f.post(t, rt.path, on)
			if got := f.lastPrompt(t); got != off {
				t.Errorf("a request that asked for thinking on the alias got it:\n%q", got[max(0, len(got)-80):])
			}
		})
	}
	count := func(model string, extra map[string]any) int {
		r, _ := f.post(t, "/v1/messages/count_tokens", withFields(map[string]any{"model": model, "messages": []any{map[string]any{"role": "user", "content": "Hi"}}}, extra))
		return int(r["input_tokens"].(float64))
	}
	if a, o, d := count("m-fast", nil), count("m", map[string]any{"thinking": map[string]any{"type": "disabled"}}), count("m", nil); a != o || a == d {
		t.Errorf("count_tokens: alias %d, thinking off %d, default %d; want alias == off != default", a, o, d)
	}
}

// No output cap: whatever limit the request gives, or the route's default, the alias generates what the main model does, so a
// reply longer than any small cap comes back whole (a summary cut short is worse than a slow one). The reply here is 3,000 tokens,
// past the 1,024 an earlier version of the alias capped at.
func TestFastAlias_doesNotCapOutput(t *testing.T) {
	const n = 3000
	f := newFastRig(t, strings.Repeat("a", n), true)
	for _, rt := range fastRoutes {
		t.Run(rt.path, func(t *testing.T) {
			for _, asked := range []int{0, 100000, 1500, 3} {
				if rt.path == "/v1/messages" && asked == 0 {
					continue // max_tokens is required on this route
				}
				a, _ := f.post(t, rt.path, rt.body("m-fast", asked))
				m, _ := f.post(t, rt.path, rt.body("m", asked))
				if rt.tokens(a) != rt.tokens(m) {
					t.Errorf("max %d: the alias generated %d tokens, the main model %d", asked, rt.tokens(a), rt.tokens(m))
				}
				if asked == 100000 && rt.tokens(a) != n {
					t.Errorf("max %d: %d tokens, want the whole %d-token reply", asked, rt.tokens(a), n)
				}
				if asked == 3 && rt.tokens(a) != 3 {
					t.Errorf("max 3: %d tokens, the request's own limit was not honored", rt.tokens(a))
				}
			}
		})
	}
}

// A request under the model's own name is the same with the alias on and off, byte for byte once the response's id and time are
// set aside, and renders the same prompt.
func TestFastAlias_mainRequestUnchanged(t *testing.T) {
	norm := regexp.MustCompile(`"(id|created|created_at|completed_at)":("[^"]*"|[0-9]+)`)
	on, off := newFastRig(t, "<think>\nhm\n</think>\n\nok", true), newFastRig(t, "<think>\nhm\n</think>\n\nok", false)
	for _, rt := range fastRoutes {
		t.Run(rt.path, func(t *testing.T) {
			for _, max := range []int{0, 64} {
				_, a := on.post(t, rt.path, rt.body("m", max))
				_, b := off.post(t, rt.path, rt.body("m", max))
				na, nb := norm.ReplaceAll(a, []byte(`"$1":0`)), norm.ReplaceAll(b, []byte(`"$1":0`))
				if !bytes.Equal(na, nb) {
					t.Errorf("max %d: the response differs with the alias on:\n%s\n%s", max, na, nb)
				}
				if on.lastPrompt(t) != off.lastPrompt(t) {
					t.Errorf("max %d: the prompt differs with the alias on", max)
				}
			}
		})
	}
}

// /v1/models: the model's own entry advertises the alias; the alias is not an entry of its own, so a listing is one
// model; off advertises nothing; a model really served under the alias's name wins, and then nothing is advertised for
// the model it would have aliased. An unknown name stays a 404, alias or not.
func TestFastAlias_modelsAndNames(t *testing.T) {
	list := func(f *fastRig) []map[string]any {
		resp, err := http.Get(f.ts.URL + "/v1/models")
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var out struct {
			Data []map[string]any `json:"data"`
		}
		_ = json.NewDecoder(resp.Body).Decode(&out)
		return out.Data
	}
	f := newFastRig(t, "ok", true)
	if d := list(f); len(d) != 1 || d[0]["id"] != "m" || d[0]["fast_model"] != "m-fast" || d[0]["fast_max_tokens"] != nil {
		t.Errorf("listing %v; want one entry, m, advertising m-fast and no cap", d)
	}
	off := newFastRig(t, "ok", false)
	if d := list(off); len(d) != 1 || d[0]["fast_model"] != nil {
		t.Errorf("with the alias off the listing advertises it: %v", d)
	}
	status := func(f *fastRig, model string) int {
		b, _ := json.Marshal(fastRoutes[0].body(model, 0))
		resp, err := http.Post(f.ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(b))
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		return resp.StatusCode
	}
	if s := status(f, "m-fast"); s != 200 {
		t.Errorf("the alias: %d", s)
	}
	if s := status(f, "nope-fast"); s != 404 {
		t.Errorf("an alias of a model not served: %d, want 404", s)
	}
	if s := status(off, "m-fast"); s != 404 {
		t.Errorf("the alias with -fast-alias=false: %d, want 404", s)
	}
	// A served model named m-fast takes the name.
	taken := newFastRig(t, "ok", true)
	other := &loadedModel{tk: taken.lm.tk, name: "m-fast", tmpl: taken.lm.tmpl, scripted: []int{}}
	taken.s.regMu.Lock()
	taken.s.models["m-fast"] = other
	taken.s.regMu.Unlock()
	if _, ok := taken.s.fastTarget("m-fast"); ok {
		t.Error("an alias shadows a model served under its name")
	}
	if a := taken.s.fastAliasOf("m"); a != "" {
		t.Errorf("m advertises %q though m-fast is a served model", a)
	}
}

// The request log marks a request that named the alias, and no other line changes.
func TestFastAlias_requestLogMarksTheAlias(t *testing.T) {
	f := newFastRig(t, "ok", true)
	f.post(t, "/v1/chat/completions", fastRoutes[0].body("m", 0))
	f.post(t, "/v1/chat/completions", fastRoutes[0].body("m-fast", 0))
	lines := strings.Split(strings.TrimSpace(f.log.String()), "\n")
	if len(lines) != 2 || strings.Contains(lines[0], "fast=") || !strings.HasSuffix(lines[1], " fast=1") || !strings.Contains(lines[1], "model=m ") {
		t.Errorf("request log:\n%s", f.log.String())
	}
}

// applyFast on each request type, whatever the client asked for: it resolves to thinking off, explicitly, with no budget, and
// leaves the output limits as the client gave them. Leaving thinking on for the alias turns it red.
func TestFastAlias_applyFastTurnsThinkingOff(t *testing.T) {
	s := &server{}
	budget, big, one := 512, 5000, 1
	on := map[string]json.RawMessage{"enable_thinking": json.RawMessage("true")}
	for name, tc := range map[string]struct {
		tr    func() thinkRequest
		limit func() []int
	}{
		"chat": {func() thinkRequest {
			r := chatReq{ChatTemplateKwargs: on, ReasoningEffort: "high", ThinkingTokenBudget: &budget}
			r.MaxTokens, r.MaxCompletionTokens = &big, &one
			r.applyFast()
			return r.think()
		}, nil},
		"responses": {func() thinkRequest {
			r := responseReq{Reasoning: &struct {
				Effort string `json:"effort"`
			}{Effort: "high"}}
			r.applyFast()
			return r.thinkRequest()
		}, nil},
		"messages": {func() thinkRequest {
			r := anthropicReq{Thinking: &struct {
				Type         string `json:"type"`
				BudgetTokens int    `json:"budget_tokens"`
			}{Type: "enabled", BudgetTokens: 512}}
			r.applyFast()
			return r.thinkRequest()
		}, nil},
	} {
		ts, err := s.resolveThink(tc.tr())
		if err != nil || !ts.explicit || ts.mode != chat.ThinkOff || ts.budget != 0 {
			t.Errorf("%s: resolved %+v (%v); want explicit thinking off and no budget", name, ts, err)
		}
	}
	c := chatReq{}
	c.MaxTokens, c.MaxCompletionTokens = &big, &one
	c.applyFast()
	if *c.MaxTokens != big || *c.MaxCompletionTokens != one {
		t.Errorf("chat limits: max_tokens %d, max_completion_tokens %d, want %d and %d untouched", *c.MaxTokens, *c.MaxCompletionTokens, big, one)
	}
	k := chatReq{ChatTemplateKwargs: on}
	k.applyFast()
	if string(on["enable_thinking"]) != "true" {
		t.Error("applyFast edited the caller's chat_template_kwargs in place")
	}
}
