package serveapp

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// A thinking model can write its tool call inside its reasoning block and end its turn there (the call is then read from
// the reasoning, settleCalls), or end its turn with no answer at all (the reply then names why, emptyAnswerOf). The shapes
// were seen on a real client: docs/measurements/integration-harness-2026-10/i3-night-2026-10-13.md, "Leads seen in the runs".

// mellum21Template is Mellum2.1's own chat template (pinned in testdata/chat_think_goldens): ChatML with Qwen3's
// thinking control, the model the empty answers were seen on.
func mellum21Template(t *testing.T) *chat.Template {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "chat_think_goldens", "mellum21_think_history.json"))
	if err != nil {
		t.Fatal(err)
	}
	var gs []struct {
		ChatTemplate string `json:"chat_template"`
	}
	if err := json.Unmarshal(raw, &gs); err != nil || len(gs) != 1 {
		t.Fatalf("golden: %v (%d entries)", err, len(gs))
	}
	tm, err := chat.Detect(chat.Meta{ChatTemplate: gs[0].ChatTemplate})
	if err != nil {
		t.Fatal(err)
	}
	if tm.Reasoning() == nil {
		t.Fatal("Mellum2.1's template carries no reasoning behaviour")
	}
	return tm
}

// printableAlphabet is every printable ASCII byte plus newline: prompts render tool schemas and ChatML markup.
func printableAlphabet() string {
	var b strings.Builder
	for c := byte(' '); c <= '~'; c++ {
		b.WriteByte(c)
	}
	b.WriteByte('\n')
	return b.String()
}

var execTool = chat.Tool{Name: "exec_command", Description: "Runs a command",
	Parameters: []byte(`{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}`)}

const callInReasoning = "<think>\nI cannot answer without reading calc.go, so I will read it.\n" +
	"<tool_call>\n{\"name\": \"exec_command\", \"arguments\": {\"cmd\": \"cat calc.go\"}}\n</tool_call>"

// The unit where the call is lost: settleCalls reads the answer text, and a call written inside the reasoning block is not in
// it. Fed through the real splitter for Mellum2.1's template, with the block left open (the turn ended inside it) and with
// the block closed after the call (the answer is then empty).
func TestSettleCalls_callWrittenInsideReasoning(t *testing.T) {
	tm := mellum21Template(t)
	for name, reply := range map[string]string{
		"block left open":         callInReasoning,
		"block closed after call": callInReasoning + "\n</think>\n\n",
	} {
		th := newThinkOut(tm, nil, thinkSettings{format: rfSplit}, nil)
		answer := th.feed(reply) + th.finish()
		if strings.TrimSpace(answer) != "" {
			t.Fatalf("%s: the splitter gave answer %q; the case needs the call inside the reasoning", name, answer)
		}
		calls, lead := settleCalls(tm, answer, []chat.Tool{execTool}, th)
		if len(calls) != 1 || calls[0].Name != "exec_command" || !strings.Contains(string(calls[0].Arguments), "cat calc.go") {
			t.Errorf("%s: calls %+v (lead %q); want the exec_command call the model wrote in its reasoning", name, calls, lead)
		}
	}
	// A call drafted in the reasoning is not taken when the model went on to answer in prose: the answer is the turn.
	th := newThinkOut(tm, nil, thinkSettings{format: rfSplit}, nil)
	answer := th.feed(callInReasoning+"\nActually I know it.\n</think>\n\nIt returns 5.") + th.finish()
	if calls, _ := settleCalls(tm, answer, []chat.Tool{execTool}, th); len(calls) != 0 {
		t.Errorf("a prose answer after a call drafted in the reasoning: calls %+v, want none", calls)
	}
}

// scriptTokenizer is a byte-level BPE over printable ASCII (one token per character, one merge so it can encode) with
// ChatML's turn markers and <think> / </think> as added tokens, as Mellum2.1's tokenizer has them.
func scriptTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	vocab := map[string]int{}
	add := func(tok string) {
		if _, ok := vocab[tok]; !ok {
			vocab[tok] = len(vocab)
		}
	}
	for _, r := range printableAlphabet() {
		switch r {
		case ' ':
			add("Ġ")
		case '\n':
			add("Ċ")
		default:
			add(string(r))
		}
	}
	add("ĊĊ")
	var added []map[string]any
	for _, sp := range []string{"<|im_start|>", "<|im_end|>", "<think>", "</think>"} {
		add(sp)
		added = append(added, map[string]any{"id": vocab[sp], "content": sp, "special": true})
	}
	raw, _ := json.Marshal(map[string]any{
		"model":         map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{"Ċ Ċ"}},
		"added_tokens":  added,
		"pre_tokenizer": map[string]any{"type": "ByteLevel", "add_prefix_space": false},
		"decoder":       map[string]any{"type": "ByteLevel"},
	})
	tk, err := tokenizer.LoadJSONBytes(raw)
	if err != nil {
		t.Fatal(err)
	}
	return tk
}

// scriptedServer serves one model, "m", on Mellum2.1's template, whose every generation is reply (the think tags as their
// single tokens, every other character its own).
func scriptedServer(t *testing.T, reply string) *httptest.Server {
	t.Helper()
	tk := scriptTokenizer(t)
	ids, err := tk.Encode(reply, false)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := tk.Decode(ids); got != reply {
		t.Fatalf("the script tokenizer does not round-trip the reply: %q", got)
	}
	s := &server{}
	lm := &loadedModel{tk: tk, name: "m", scripted: append([]int{}, ids...)} // non-nil even when empty: an end of turn at the first token
	lm.tmpl = s.cfg.tuneTemplate(mellum21Template(t))
	s.models = map[string]*loadedModel{"m": lm}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/responses", s.handleResponses)
	mux.HandleFunc("POST /v1/chat/completions", s.handleChat)
	mux.HandleFunc("POST /v1/messages", s.handleMessages)
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return ts
}

// postScripted sends body and returns the JSON object (stream false) or every SSE data object, in order (stream true).
func postScripted(t *testing.T, url string, body map[string]any) []map[string]any {
	t.Helper()
	b, _ := json.Marshal(body)
	resp, err := http.Post(url, "application/json", strings.NewReader(string(b)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("%s: %d %s", url, resp.StatusCode, raw)
	}
	var out []map[string]any
	if s, _ := body["stream"].(bool); !s {
		var m map[string]any
		if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
			t.Fatal(err)
		}
		return append(out, m)
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		d, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok || d == "[DONE]" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(d), &m); err != nil {
			t.Fatalf("bad SSE data %q: %v", d, err)
		}
		out = append(out, m)
	}
	return out
}

func requestFor(route string, tools, stream bool) map[string]any {
	const q = "What does Add(2, 3) in calc.go return? Answer with the number."
	params := json.RawMessage(execTool.Parameters)
	switch route {
	case "/v1/responses":
		b := map[string]any{"model": "m", "input": q, "stream": stream}
		if tools {
			b["tools"] = []any{map[string]any{"type": "function", "name": execTool.Name, "description": execTool.Description, "parameters": params}}
		}
		return b
	case "/v1/chat/completions":
		b := map[string]any{"model": "m", "stream": stream, "messages": []any{map[string]any{"role": "user", "content": q}}}
		if tools {
			b["tools"] = []any{map[string]any{"type": "function", "function": map[string]any{"name": execTool.Name, "description": execTool.Description, "parameters": params}}}
		}
		return b
	default: // /v1/messages
		b := map[string]any{"model": "m", "max_tokens": 512, "stream": stream, "messages": []any{map[string]any{"role": "user", "content": q}}}
		if tools {
			b["tools"] = []any{map[string]any{"name": execTool.Name, "description": execTool.Description, "input_schema": params}}
		}
		return b
	}
}

// emptyReasonOf finds the reason a route's reply gives for having no answer, and whether the reply carried a tool call.
// "" means no reason was given.
func emptyReasonOf(t *testing.T, route string, objs []map[string]any) (reason string, called bool) {
	t.Helper()
	asMap := func(v any) map[string]any { m, _ := v.(map[string]any); return m }
	for _, o := range objs {
		switch route {
		case "/v1/responses":
			r := o
			if typ, _ := o["type"].(string); typ != "" { // streamed: the final object rides response.completed
				if typ != "response.completed" {
					continue
				}
				r = asMap(o["response"])
			}
			for _, it := range r["output"].([]any) {
				if asMap(it)["type"] == "function_call" {
					called = true
				}
			}
			if r["status"] == "incomplete" {
				reason, _ = asMap(r["incomplete_details"])["reason"].(string)
			}
		case "/v1/chat/completions":
			for _, c := range o["choices"].([]any) {
				ch := asMap(c)
				if m := asMap(ch["message"]); m != nil && m["tool_calls"] != nil {
					called = true
				}
				if d := asMap(ch["delta"]); d != nil && d["tool_calls"] != nil {
					called = true
				}
				if e := asMap(ch["goinfer_empty_answer"]); e != nil {
					reason, _ = e["reason"].(string)
				}
			}
		default:
			if content, ok := o["content"].([]any); ok {
				for _, b := range content {
					if asMap(b)["type"] == "tool_use" {
						called = true
					}
				}
			}
			if cb := asMap(o["content_block"]); cb != nil && cb["type"] == "tool_use" {
				called = true
			}
			if e := asMap(o["goinfer_empty_answer"]); e != nil {
				reason, _ = e["reason"].(string)
			}
		}
	}
	return reason, called
}

var allRoutes = []string{"/v1/responses", "/v1/chat/completions", "/v1/messages"}

// A call written inside the reasoning reaches the client as a call on every route, streamed or not, with no empty-answer
// reason.
func TestToolTurn_callInsideReasoningIsReturned(t *testing.T) {
	for name, reply := range map[string]string{"open": callInReasoning, "closed": callInReasoning + "\n</think>\n\n"} {
		ts := scriptedServer(t, reply)
		for _, route := range allRoutes {
			for _, stream := range []bool{false, true} {
				reason, called := emptyReasonOf(t, route, postScripted(t, ts.URL+route, requestFor(route, true, stream)))
				if !called || reason != "" {
					t.Errorf("block %s, %s stream=%v: called=%v reason=%q; want the call and no empty-answer reason", name, route, stream, called, reason)
				}
			}
		}
	}
}

// The invariant: a reply with no answer text and no tool call names why, on every route, streamed or not, with tools or
// without. A reply with an answer names nothing.
func TestEmptyAnswer_alwaysCarriesAReason(t *testing.T) {
	cases := []struct{ name, reply, want string }{
		{"ended inside its reasoning", "<think>\nThe file is unreadable, but Add surely returns 5.\n", "ended_in_reasoning"},
		{"closed its reasoning, then ended", "<think>\nAdd returns 5.\n</think>\n\n", "reasoning_only"},
		{"ended at the first token", "", "no_output"},
		{"answered (the control)", "<think>\nAdd returns 5.\n</think>\n\n5", ""},
	}
	for _, c := range cases {
		ts := scriptedServer(t, c.reply)
		for _, route := range allRoutes {
			for _, tools := range []bool{false, true} {
				for _, stream := range []bool{false, true} {
					reason, called := emptyReasonOf(t, route, postScripted(t, ts.URL+route, requestFor(route, tools, stream)))
					if called || reason != c.want {
						t.Errorf("%s: %s tools=%v stream=%v: reason %q (called %v), want %q", c.name, route, tools, stream, reason, called, c.want)
					}
				}
			}
		}
	}
}
