package chatapp

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// thinkTokenizer is a byte-level vocab where every character the tests use is one token, plus "<think>" / "</think>" as whole
// tokens, so a reply can be fed either way a model might write it.
func thinkTokenizer(t *testing.T, alphabet string) (*tokenizer.Tokenizer, map[string]int) {
	t.Helper()
	vocab := map[string]int{}
	add := func(tok string) {
		if _, ok := vocab[tok]; !ok {
			vocab[tok] = len(vocab)
		}
	}
	for _, r := range alphabet {
		switch r {
		case ' ':
			add("Ġ")
		case '\n':
			add("Ċ")
		default:
			add(string(r))
		}
	}
	add("<think>")
	add("</think>")
	raw, _ := json.Marshal(map[string]any{
		"model":   map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{}},
		"decoder": map[string]any{"type": "ByteLevel"},
	})
	tk, err := tokenizer.LoadJSONBytes(raw)
	if err != nil {
		t.Fatal(err)
	}
	return tk, vocab
}

func thinkIDs(vocab map[string]int, text string, whole bool) []int {
	var ids []int
	for i := 0; i < len(text); {
		if whole {
			if strings.HasPrefix(text[i:], "</think>") {
				ids, i = append(ids, vocab["</think>"]), i+len("</think>")
				continue
			}
			if strings.HasPrefix(text[i:], "<think>") {
				ids, i = append(ids, vocab["<think>"]), i+len("<think>")
				continue
			}
		}
		switch c := text[i]; c {
		case ' ':
			ids = append(ids, vocab["Ġ"])
		case '\n':
			ids = append(ids, vocab["Ċ"])
		default:
			ids = append(ids, vocab[string(c)])
		}
		i++
	}
	return ids
}

// goldenTemplate detects a pinned checkpoint's real chat template, as a session does at load.
func goldenTemplate(t *testing.T, checkpoint string) *chat.Template {
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
		if g.Checkpoint == checkpoint {
			tm, err := chat.Detect(chat.Meta{ChatTemplate: g.ChatTemplate})
			if err != nil {
				t.Fatal(err)
			}
			return tm
		}
	}
	t.Fatalf("no golden for %s", checkpoint)
	return nil
}

func feed(ids []int) <-chan int {
	ch := make(chan int, len(ids))
	for _, id := range ids {
		ch <- id
	}
	close(ch)
	return ch
}

const thinkAlphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?'-<>/|\n0123456789Ã©"

// The CLI's stream loop: reasoning and answer come out separately, in both encodings of the tags, and a multi-byte character
// split across two tokens reaches the reader whole in the reasoning as well as in the answer.
func TestStreamReply_separatesReasoningFromTheAnswer(t *testing.T) {
	tk, vocab := thinkTokenizer(t, thinkAlphabet)
	s := &session{tk: tk}
	tm := goldenTemplate(t, "qwen3-4b").WithThinking(chat.ThinkTemplate)
	reply := "<think>\ncafé au lait\n</think>\n\nOK é"
	for _, whole := range []bool{true, false} {
		var frags []string
		var chunks []string
		answer, reasoning, n, truncated := s.streamReply(feed(thinkIDs(vocab, reply, whole)), tm.NewReplySplitter(nil),
			func(r string) { frags = append(frags, r) }, func(c string) { chunks = append(chunks, c) })
		if reasoning != "café au lait" || answer != "OK é" || truncated {
			t.Fatalf("whole=%v reasoning=%q answer=%q truncated=%v", whole, reasoning, answer, truncated)
		}
		if strings.Join(frags, "") != reasoning || strings.Join(chunks, "") != answer {
			t.Errorf("callbacks disagree with the return values: %q / %q", frags, chunks)
		}
		for _, f := range append(frags, chunks...) {
			if !utf8.ValidString(f) {
				t.Errorf("whole=%v: fragment %q is not valid UTF-8 (a rune was split)", whole, f)
			}
		}
		if n != len(thinkIDs(vocab, reply, whole)) {
			t.Errorf("token count %d counts the thinking too: want %d", n, len(thinkIDs(vocab, reply, whole)))
		}
	}
	// A reply cut off inside the think block has no answer and says so.
	answer, reasoning, _, truncated := s.streamReply(feed(thinkIDs(vocab, "<think>\nstill going", true)), tm.NewReplySplitter(nil), nil, nil)
	if answer != "" || reasoning != "still going" || !truncated {
		t.Errorf("truncated: answer=%q reasoning=%q truncated=%v", answer, reasoning, truncated)
	}
	// No splitter (a model with no recognised thinking control): everything is the answer, tags and all — exactly as before.
	answer, reasoning, _, truncated = s.streamReply(feed(thinkIDs(vocab, "<think>\nx\n</think>\n\nA", true)), nil, nil, nil)
	if answer != "<think>\nx\n</think>\n\nA" || reasoning != "" || truncated {
		t.Errorf("unmanaged: answer=%q reasoning=%q", answer, reasoning)
	}
}

// buildPrompt renders the session's thinking mode; a JSON grammar forces thinking off; the history carries the answer alone.
func TestBuildPrompt_thinkingModes(t *testing.T) {
	small := goldenTemplate(t, "qwen3.5-0.8b")
	nine := goldenTemplate(t, "qwen3.5-9b")
	hist := []msg{{"user", "Hi"}}
	cases := []struct {
		name string
		tm   *chat.Template
		mode chat.ThinkMode
		json bool
		tail string
	}{
		{"0.8B template default", small, chat.ThinkTemplate, false, "<|im_start|>assistant\n<think>\n\n</think>\n\n"},
		{"0.8B on", small, chat.ThinkOn, false, "<|im_start|>assistant\n<think>\n"},
		{"9B template default", nine, chat.ThinkTemplate, false, "<|im_start|>assistant\n<think>\n"},
		{"9B off", nine, chat.ThinkOff, false, "<|im_start|>assistant\n<think>\n\n</think>\n\n"},
		{"9B asis writes nothing after the tag", nine, chat.ThinkAsIs, false, "<|im_start|>assistant\n"},
		{"9B default under a JSON grammar is forced off", nine, chat.ThinkTemplate, true, "<|im_start|>assistant\n<think>\n\n</think>\n\n"},
	}
	for _, tc := range cases {
		s := &session{tmpl: tc.tm, think: tc.mode, jsonOut: tc.json, history: hist}
		prompt, turns, tm := s.buildPrompt()
		if !strings.HasSuffix(prompt, tc.tail) {
			t.Errorf("%s: prompt ends %q, want it to end %q", tc.name, prompt[max(0, len(prompt)-50):], tc.tail)
		}
		if len(turns) != 1 || tm == nil {
			t.Errorf("%s: turns=%d template=%v", tc.name, len(turns), tm)
		}
	}
	// No template: the raw fallback, no template returned.
	s := &session{think: chat.ThinkTemplate, history: hist}
	if _, _, tm := s.buildPrompt(); tm != nil {
		t.Error("a session with no template must not invent one")
	}
}

// The budget goes on a thinking reply and stays off where a processor would cost more than it buys.
func TestApplyBudget_whereItInstalls(t *testing.T) {
	tk, _ := thinkTokenizer(t, thinkAlphabet)
	nine := goldenTemplate(t, "qwen3.5-9b").WithThinking(chat.ThinkTemplate) // default on, prompt opens the block
	small := goldenTemplate(t, "qwen3.5-0.8b").WithThinking(chat.ThinkTemplate)
	install := func(s *session, tm *chat.Template) decoder.SamplingParams {
		var sp decoder.SamplingParams
		s.tk = tk
		s.applyBudget(&sp, tm, nil)
		return sp
	}
	if sp := install(&session{maxTok: 160}, nine); sp.LogitProcessor == nil || sp.LogitProcessorGate == nil {
		t.Error("a thinking model must get a gated budget processor")
	}
	if sp := install(&session{maxTok: 160}, small); sp.LogitProcessor != nil {
		t.Error("the 0.8B under its default (thinking off) can never be inside a block")
	}
	for name, s := range map[string]*session{
		"json":  {maxTok: 160, jsonOut: true},
		"draft": {maxTok: 160, draft: &decoder.Model{}},
		"ngram": {maxTok: 160, ngram: true},
		"tiny":  {maxTok: 3},
	} {
		if sp := install(s, nine); sp.LogitProcessor != nil {
			t.Errorf("%s: the budget must not install", name)
		}
	}
	// An existing processor is left alone.
	called := 0
	sp := decoder.SamplingParams{LogitProcessor: func([]int, []float32) { called++ }}
	s := &session{maxTok: 160, tk: tk}
	s.applyBudget(&sp, nine, nil)
	sp.LogitProcessor(nil, nil)
	if called != 1 || sp.LogitProcessorGate != nil {
		t.Errorf("an existing processor must be left exactly as it was (called=%d)", called)
	}
}
