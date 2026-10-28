package serveapp

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// B35 at the serve layer (docs/queue-engineering.md, B35): for a model whose snapshots carry its recurrent state, every chat route hands
// the decoder a KVScope.Keep, and the next turn's real request, as a client sends it (the reply, then a new question), repeats the
// prompt's first Keep ids, so the state kept there is the one turn 2 continues from. Run on the scripted server with Qwen3.5-9B's own
// template, whose history render rewrites the end of the prompt: a state kept at the prompt's end would serve no later turn.

// qwen35Template is Qwen3.5-9B's chat template, pinned in testdata/chat_think_goldens/think_modes.json.
func qwen35Template(t *testing.T) *chat.Template {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "chat_think_goldens", "think_modes.json"))
	if err != nil {
		t.Fatal(err)
	}
	var gs []struct {
		Checkpoint   string `json:"checkpoint"`
		ChatTemplate string `json:"chat_template"`
	}
	if err := json.Unmarshal(raw, &gs); err != nil {
		t.Fatal(err)
	}
	for _, g := range gs {
		if g.Checkpoint == "qwen3.5-9b" {
			tm, err := chat.Detect(chat.Meta{ChatTemplate: g.ChatTemplate})
			if err != nil {
				t.Fatal(err)
			}
			return tm
		}
	}
	t.Fatal("think_modes.json has no qwen3.5-9b entry")
	return nil
}

// keepTokenizer is scriptTokenizer with the whole byte-level alphabet, as a real byte-level BPE has it: the next-turn render (chat.NextTurns)
// carries a control byte that a printable-only vocabulary cannot encode.
func keepTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	vocab := map[string]int{}
	add := func(tok string) {
		if _, ok := vocab[tok]; !ok {
			vocab[tok] = len(vocab)
		}
	}
	for b := 0; b < 256; b++ {
		add(string(byteLevelRune(byte(b))))
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

// byteLevelRune is GPT-2's byte-to-unicode map: printable Latin-1 bytes stand for themselves, the rest map to 256 upward in order.
func byteLevelRune(b byte) rune {
	switch {
	case b >= '!' && b <= '~', b >= 0xA1 && b <= 0xAC, b >= 0xAE:
		return rune(b)
	}
	n := 0
	for c := 0; c < int(b); c++ {
		if !(c >= '!' && c <= '~' || c >= 0xA1 && c <= 0xAC || c >= 0xAE) {
			n++
		}
	}
	return rune(256 + n)
}

// keepServer is scriptedServer on Qwen3.5-9B's template, replying reply; each generation's prompt ids and KV scope are appended to the
// returned slices. recurrent is the model's KeepsRecurrentSnapshots.
func keepServer(t *testing.T, reply string, recurrent bool) (*httptest.Server, *[][]int, *[]kvScope) {
	t.Helper()
	tk := keepTokenizer(t)
	ids, err := tk.Encode(reply, false)
	if err != nil {
		t.Fatal(err)
	}
	s := &server{}
	lm := &loadedModel{tk: tk, name: "m", scripted: append([]int{}, ids...), scriptedRecurrent: recurrent}
	lm.tmpl = s.cfg.tuneTemplate(qwen35Template(t))
	var prompts [][]int
	var scopes []kvScope
	lm.onScripted = func(p []int) { prompts = append(prompts, slices.Clone(p)) }
	lm.onScriptedScope = func(sc kvScope) { scopes = append(scopes, sc) }
	s.models = map[string]*loadedModel{"m": lm}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/responses", s.handleResponses)
	mux.HandleFunc("POST /v1/chat/completions", s.handleChat)
	mux.HandleFunc("POST /v1/messages", s.handleMessages)
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return ts, &prompts, &scopes
}

func TestB35Keep_chatTurnsContinueFromKeep(t *testing.T) {
	const reply = "Sure, here it is."
	tails := map[bool]string{false: "<think>\n\n</think>\n\n", true: "<think>\n"}
	for _, thinking := range []bool{false, true} {
		ts, prompts, scopes := keepServer(t, reply, true)
		tk := keepTokenizer(t)
		msgs := []map[string]any{{"role": "user", "content": "What does Add(2, 3) return?"}}
		for turn, next := range []string{"And Add(4, 5)?", "And Sub(9, 1)?", ""} {
			postScripted(t, ts.URL+"/v1/chat/completions", map[string]any{"model": "m", "messages": msgs, "max_tokens": 32,
				"chat_template_kwargs": map[string]any{"enable_thinking": thinking}})
			if len(*prompts) != turn+1 || len(*scopes) != turn+1 {
				t.Fatalf("thinking=%v turn %d: %d prompts and %d scopes captured", thinking, turn+1, len(*prompts), len(*scopes))
			}
			p, k := (*prompts)[turn], (*scopes)[turn].Keep
			if k <= 0 || k >= len(p) {
				t.Fatalf("thinking=%v turn %d: Keep %d for a %d-id prompt, want inside it", thinking, turn+1, k, len(p))
			}
			if tail, _ := tk.Decode(p[k:]); tail != tails[thinking] {
				t.Errorf("thinking=%v turn %d: the prompt past Keep reads %q, want the generation tail %q", thinking, turn+1, tail, tails[thinking])
			}
			if turn > 0 {
				prev, pk := (*prompts)[turn-1], (*scopes)[turn-1].Keep
				if !slices.Equal(p[:pk], prev[:pk]) {
					t.Errorf("thinking=%v turn %d: the request does not repeat turn %d's first %d ids (its Keep)", thinking, turn+1, turn, pk)
				}
				// Why Keep exists: the template rewrites the end of the previous prompt, so its whole length is not a prefix of this one.
				if len(p) >= len(prev) && slices.Equal(p[:len(prev)], prev) {
					t.Errorf("thinking=%v turn %d: the request repeats all of turn %d's prompt; the tail is not rewritten and the case proves nothing", thinking, turn+1, turn)
				}
			}
			if next != "" {
				msgs = append(msgs, map[string]any{"role": "assistant", "content": reply}, map[string]any{"role": "user", "content": next})
			}
		}
	}
}

// The other chat routes compute it too: tools, the Responses API and the Anthropic Messages API.
func TestB35Keep_everyRouteSetsKeep(t *testing.T) {
	ts, prompts, scopes := keepServer(t, "Sure.", true)
	user := []map[string]any{{"role": "user", "content": "What does Add(2, 3) return?"}}
	for _, c := range []struct {
		route string
		body  map[string]any
	}{
		{"/v1/chat/completions", map[string]any{"model": "m", "messages": user, "max_tokens": 16}},
		{"/v1/chat/completions", map[string]any{"model": "m", "messages": user, "max_tokens": 16,
			"tools": []map[string]any{{"type": "function", "function": map[string]any{"name": execTool.Name, "description": execTool.Description,
				"parameters": json.RawMessage(execTool.Parameters)}}}}},
		{"/v1/responses", map[string]any{"model": "m", "input": "What does Add(2, 3) return?", "max_output_tokens": 16}},
		{"/v1/messages", map[string]any{"model": "m", "messages": user, "max_tokens": 16}},
	} {
		n := len(*scopes)
		postScripted(t, ts.URL+c.route, c.body)
		if len(*scopes) != n+1 {
			t.Fatalf("%s: %d generations, want 1", c.route, len(*scopes)-n)
		}
		if k, p := (*scopes)[n].Keep, (*prompts)[n]; k <= 0 || k >= len(p) {
			t.Errorf("%s (tools %v): Keep %d for a %d-id prompt", c.route, c.body["tools"] != nil, k, len(p))
		}
	}
}

// A model whose snapshots do not carry its state gets no Keep: the decoder then takes no recurrent snapshot.
func TestB35Keep_noneWithoutRecurrentSnapshots(t *testing.T) {
	ts, _, scopes := keepServer(t, "Sure.", false)
	postScripted(t, ts.URL+"/v1/chat/completions", map[string]any{"model": "m", "max_tokens": 16,
		"messages": []map[string]any{{"role": "user", "content": "What does Add(2, 3) return?"}}})
	if len(*scopes) != 1 || (*scopes)[0].Keep != 0 {
		t.Fatalf("scopes %+v: want one generation with Keep 0", *scopes)
	}
}
