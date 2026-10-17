package serveapp

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The pilot's replies that came back as text (chat/testdata/tool_calls_as_text) must leave /v1/chat/completions as a call whether the client streamed
// or not, with the same name and arguments; and the plants (prose that imitates a call) must leave it as content, again identically. The scripted
// model writes the reply byte for byte, so this is the server's own streaming path (runToolTurn: the prose streamer, then settleCalls).

type wireReply struct {
	content, finish string
	calls           []string // "name args"
}

// chatWire reads a chat/completions reply, streamed (deltas assembled by tool-call index) or whole.
func chatWire(t *testing.T, objs []map[string]any) wireReply {
	t.Helper()
	var w wireReply
	asMap := func(v any) map[string]any { m, _ := v.(map[string]any); return m }
	type acc struct{ name, args string }
	byIdx := map[int]*acc{}
	add := func(tc map[string]any, fallback int) {
		idx := fallback
		if f, ok := tc["index"].(float64); ok {
			idx = int(f)
		}
		a := byIdx[idx]
		if a == nil {
			a = &acc{}
			byIdx[idx] = a
		}
		fn := asMap(tc["function"])
		if n, _ := fn["name"].(string); n != "" {
			a.name = n
		}
		s, _ := fn["arguments"].(string)
		a.args += s
	}
	for _, o := range objs {
		for _, c := range o["choices"].([]any) {
			ch := asMap(c)
			if f, _ := ch["finish_reason"].(string); f != "" {
				w.finish = f
			}
			if m := asMap(ch["message"]); m != nil {
				s, _ := m["content"].(string)
				w.content += s
				for i, tc := range asSlice(m["tool_calls"]) {
					add(asMap(tc), i)
				}
			}
			if d := asMap(ch["delta"]); d != nil {
				s, _ := d["content"].(string)
				w.content += s
				for i, tc := range asSlice(d["tool_calls"]) {
					add(asMap(tc), i)
				}
			}
		}
	}
	for i := 0; i < len(byIdx); i++ {
		if a := byIdx[i]; a != nil {
			w.calls = append(w.calls, a.name+" "+a.args)
		}
	}
	return w
}

func asSlice(v any) []any { s, _ := v.([]any); return s }

func pilotChatRequest(stream bool) map[string]any {
	obj := json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}}}`)
	tool := func(name string) any {
		return map[string]any{"type": "function", "function": map[string]any{"name": name, "description": name, "parameters": obj}}
	}
	return map[string]any{"model": "m", "stream": stream, "max_tokens": 4096, "tools": []any{tool("task"), tool("read")},
		"messages": []any{map[string]any{"role": "user", "content": "go"}}}
}

func TestToolCallsAsText_sameStreamedOrNot(t *testing.T) {
	file := func(n string) string {
		b, err := os.ReadFile(filepath.Join("..", "..", "chat", "testdata", "tool_calls_as_text", n))
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	for _, c := range []struct {
		name, reply string
		wantCall    string // the tool the reply must come out as; "" = it must stay content
	}{
		{"task 1: raw newlines in a string", file("pilot_task01_raw_newlines.txt"), "task"},
		{"task 2: \"function\" for \"name\"", file("pilot_task02_function_key.txt"), "task"},
		{"a block that names the tool as an object", "<tool_call>\n{\"function\": {\"name\": \"read\", \"arguments\": {\"x\": \"a\nb\"}}}\n</tool_call>", "read"},
		{"prose before a repaired call", "I will look.\n<tool_call>\n{\"function\": \"read\", \"arguments\": {\"x\": \"a.go\"}}\n</tool_call>", "read"},
		{"plant: a prose example of a call", "You can read a file like this: {\"name\": \"read\", \"arguments\": {\"x\": \"a.go\"}} and go on.", ""},
		{"plant: a bare function-key call", "{\"function\": \"read\", \"arguments\": {}}", ""},
		{"plant: a repaired shape on an unoffered tool", "<tool_call>\n{\"function\": \"rm_rf\", \"arguments\": {}}\n</tool_call>", ""},
	} {
		ts := scriptedServer(t, c.reply)
		whole := chatWire(t, postScripted(t, ts.URL+"/v1/chat/completions", pilotChatRequest(false)))
		streamed := chatWire(t, postScripted(t, ts.URL+"/v1/chat/completions", pilotChatRequest(true)))
		if whole.finish != streamed.finish || strings.Join(whole.calls, "|") != strings.Join(streamed.calls, "|") || strings.TrimSpace(whole.content) != strings.TrimSpace(streamed.content) {
			t.Errorf("%s: whole %+v, streamed %+v; want the same reply either way", c.name, whole, streamed)
		}
		if c.wantCall == "" {
			if len(whole.calls) != 0 || whole.finish == "tool_calls" || strings.TrimSpace(whole.content) != strings.TrimSpace(c.reply) {
				t.Errorf("%s: %+v; want the reply back as content, no call", c.name, whole)
			}
			continue
		}
		if len(whole.calls) != 1 || !strings.HasPrefix(whole.calls[0], c.wantCall+" {") || whole.finish != "tool_calls" {
			t.Errorf("%s: %+v; want one call to %s and finish tool_calls", c.name, whole, c.wantCall)
		}
		for _, call := range whole.calls {
			args := strings.TrimPrefix(call, c.wantCall+" ")
			var m map[string]any
			if err := json.Unmarshal([]byte(args), &m); err != nil {
				t.Errorf("%s: the call's arguments are not a JSON object: %q", c.name, args)
			}
		}
	}
}
