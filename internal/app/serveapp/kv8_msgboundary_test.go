package serveapp

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"
)

// KV8's message-boundary checkpoint at the serve layer (docs/measurements/kv8-msgboundary-2026-11/README.md, M1): for a model whose snapshots
// carry its recurrent state, every chat route hands the decoder KVScope.MsgBoundary, the ids before the prompt's last turn's content, on Qwen3.5-9B's
// own template and a byte-level tokenizer.

func TestKV8MsgBoundary_isTheStartOfTheLastTurnsContent(t *testing.T) {
	const reply = "Sure, here it is."
	ts, prompts, scopes := keepServer(t, reply, true)
	tk := keepTokenizer(t)
	msgs := []map[string]any{{"role": "user", "content": "What does Add(2, 3) return?"}}
	for turn, next := range []string{"And Add(4, 5)?", "And Sub(9, 1)?", ""} {
		postScripted(t, ts.URL+"/v1/chat/completions", map[string]any{"model": "m", "messages": msgs, "max_tokens": 32})
		p, sc := (*prompts)[turn], (*scopes)[turn]
		if turn == 0 {
			if sc.MsgBoundary != 0 {
				t.Fatalf("a single turn: MsgBoundary %d, want 0 (the system section is Boundary's)", sc.MsgBoundary)
			}
		} else {
			last, _ := msgs[len(msgs)-1]["content"].(string)
			if sc.MsgBoundary <= 0 || sc.MsgBoundary >= len(p) {
				t.Fatalf("turn %d: MsgBoundary %d for a %d-id prompt, want inside it", turn+1, sc.MsgBoundary, len(p))
			}
			if head, _ := tk.Decode(p[:sc.MsgBoundary]); !strings.HasSuffix(head, "<|im_end|>\n<|im_start|>") {
				t.Errorf("turn %d: the ids before MsgBoundary end %q, want the last turn's opening marker, which the template renders as its own segment before the role", turn+1, head[max(0, len(head)-40):])
			}
			if rest, _ := tk.Decode(p[sc.MsgBoundary:]); !strings.HasPrefix(rest, "user\n"+last) {
				t.Errorf("turn %d: the ids from MsgBoundary read %q, want the last turn's role and message %q first", turn+1, rest, last)
			}
			// The case it serves: the next request keeps every earlier turn, so it repeats the ids before this turn's last message.
			if sc.MsgBoundary >= sc.Keep {
				t.Errorf("turn %d: MsgBoundary %d is not below Keep %d", turn+1, sc.MsgBoundary, sc.Keep)
			}
		}
		if next != "" {
			if turn > 0 {
				prev, ps := (*prompts)[turn-1], (*scopes)[turn-1]
				if ps.MsgBoundary > 0 && !slices.Equal(p[:ps.MsgBoundary], prev[:ps.MsgBoundary]) {
					t.Errorf("turn %d does not repeat turn %d's first %d ids", turn+1, turn, ps.MsgBoundary)
				}
			}
			msgs = append(msgs, map[string]any{"role": "assistant", "content": reply}, map[string]any{"role": "user", "content": next})
		}
	}
}

func TestKV8MsgBoundary_everyRouteSetsIt(t *testing.T) {
	ts, prompts, scopes := keepServer(t, "Sure.", true)
	two := []map[string]any{{"role": "user", "content": "What does Add(2, 3) return?"}, {"role": "assistant", "content": "Sure."}, {"role": "user", "content": "And Add(4, 5)?"}}
	for _, c := range []struct {
		route string
		body  map[string]any
	}{
		{"/v1/chat/completions", map[string]any{"model": "m", "messages": two, "max_tokens": 16}},
		{"/v1/chat/completions", map[string]any{"model": "m", "messages": two, "max_tokens": 16,
			"tools": []map[string]any{{"type": "function", "function": map[string]any{"name": execTool.Name, "description": execTool.Description,
				"parameters": json.RawMessage(execTool.Parameters)}}}}},
		{"/v1/responses", map[string]any{"model": "m", "input": two, "max_output_tokens": 16}},
		{"/v1/messages", map[string]any{"model": "m", "messages": two, "max_tokens": 16}},
	} {
		n := len(*scopes)
		postScripted(t, ts.URL+c.route, c.body)
		if len(*scopes) != n+1 {
			t.Fatalf("%s: %d generations, want 1", c.route, len(*scopes)-n)
		}
		if b, p := (*scopes)[n].MsgBoundary, (*prompts)[n]; b <= 0 || b >= len(p) {
			t.Errorf("%s (tools %v): MsgBoundary %d for a %d-id prompt", c.route, c.body["tools"] != nil, b, len(p))
		}
	}
}

func TestKV8MsgBoundary_noneWithoutRecurrentSnapshots(t *testing.T) {
	ts, _, scopes := keepServer(t, "Sure.", false)
	postScripted(t, ts.URL+"/v1/chat/completions", map[string]any{"model": "m", "max_tokens": 16, "messages": []map[string]any{
		{"role": "user", "content": "What does Add(2, 3) return?"}, {"role": "assistant", "content": "Sure."}, {"role": "user", "content": "And Add(4, 5)?"}}})
	if len(*scopes) != 1 || (*scopes)[0].MsgBoundary != 0 {
		t.Fatalf("scopes %+v: want one generation with MsgBoundary 0", *scopes)
	}
}
