package agent

import (
	"context"
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

func goldenTemplate(t *testing.T, checkpoint string) *chat.Template {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "..", "testdata", "chat_think_goldens", "think_modes.json"))
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

// The agent's stream loop: reasoning goes to its own callback (whole runes), only the answer is emitted and returned.
func TestStreamReply_keepsReasoningOutOfTheAnswer(t *testing.T) {
	tk, vocab := thinkTokenizer(t, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?'-<>/|\n0123456789Ã©")
	s := &Session{tk: tk}
	tm := goldenTemplate(t, "qwen3-4b").WithThinking(chat.ThinkTemplate)
	reply := "<think>\ncafé au lait\n</think>\n\nOK é"
	for _, whole := range []bool{true, false} {
		var reasoning, emitted strings.Builder
		var frags []string
		answer, err := s.streamReply(context.Background(), feed(thinkIDs(vocab, reply, whole)), &decoder.Generation{}, tm.NewReplySplitter(nil),
			func(r string) { reasoning.WriteString(r); frags = append(frags, r) }, func(c string) { emitted.WriteString(c) })
		if err != nil {
			t.Fatal(err)
		}
		if answer != "OK é" || emitted.String() != "OK é" || reasoning.String() != "café au lait" {
			t.Fatalf("whole=%v answer=%q emitted=%q reasoning=%q", whole, answer, emitted.String(), reasoning.String())
		}
		for _, f := range frags {
			if !utf8.ValidString(f) {
				t.Errorf("whole=%v: reasoning fragment %q is not valid UTF-8", whole, f)
			}
		}
	}
	// No splitter (a model with no recognised thinking control): everything is the answer, as before.
	answer, _ := s.streamReply(context.Background(), feed(thinkIDs(vocab, "<think>\nx\n</think>\n\nA", true)), &decoder.Generation{}, nil, nil, nil)
	if answer != "<think>\nx\n</think>\n\nA" {
		t.Errorf("unmanaged answer = %q", answer)
	}
}

// DECIDE's JSON grammar governs the first token, so its prompt must not end inside an open think block; the answer keeps the mode.
func TestTemplateFor_decideIsThinkingOff(t *testing.T) {
	nine := goldenTemplate(t, "qwen3.5-9b") // default on: the prompt ends `<think>\n`
	s := &Session{tmpl: nine, think: chat.ThinkTemplate}
	user := []chat.Turn{{Role: "user", Content: "Hi"}}
	if got := s.templateFor(false).Render("", user); !strings.HasSuffix(got, "<think>\n") || strings.HasSuffix(got, "</think>\n\n") {
		t.Errorf("the answer phase must keep the model's own mode, got ...%q", got[max(0, len(got)-30):])
	}
	if got := s.templateFor(true).Render("", user); !strings.HasSuffix(got, "<think>\n\n</think>\n\n") {
		t.Errorf("the decide phase must be rendered thinking-off, got ...%q", got[max(0, len(got)-30):])
	}
	// No template at all: nil, not a panic.
	if (&Session{think: chat.ThinkTemplate}).templateFor(true) != nil {
		t.Error("a session with no template must have no template")
	}
}

func TestResolveThinking(t *testing.T) {
	for in, want := range map[string]chat.ThinkMode{"": chat.ThinkTemplate, "template": chat.ThinkTemplate, "asis": chat.ThinkAsIs, "on": chat.ThinkOn, "off": chat.ThinkOff} {
		got, err := resolveThinking(in)
		if err != nil || got != want {
			t.Errorf("resolveThinking(%q) = %v, %v; want %v", in, got, err, want)
		}
	}
	if _, err := resolveThinking("maybe"); err == nil {
		t.Error("an unknown mode must be an error")
	}
}

// The budget goes on a thinking answer and nowhere it would cost more than it buys.
func TestApplyBudget_agent(t *testing.T) {
	tk, _ := thinkTokenizer(t, "abc")
	nine := goldenTemplate(t, "qwen3.5-9b").WithThinking(chat.ThinkTemplate)
	small := goldenTemplate(t, "qwen3.5-0.8b").WithThinking(chat.ThinkTemplate)
	s := &Session{tk: tk}
	var sp decoder.SamplingParams
	s.applyBudget(&sp, nine, nil, 512, false)
	if sp.LogitProcessor == nil || sp.LogitProcessorGate == nil {
		t.Error("a thinking answer must get a gated budget")
	}
	sp = decoder.SamplingParams{}
	s.applyBudget(&sp, nine, nil, 96, true)
	if sp.LogitProcessor != nil {
		t.Error("the constrained DECIDE phase must never get a budget")
	}
	sp = decoder.SamplingParams{}
	s.applyBudget(&sp, small, nil, 512, false)
	if sp.LogitProcessor != nil {
		t.Error("the 0.8B under its default (thinking off) has nothing to bound")
	}
	sp = decoder.SamplingParams{}
	s.applyBudget(&sp, nine, nil, 3, false)
	if sp.LogitProcessor != nil {
		t.Error("a turn under 4 tokens is left alone")
	}
	called := 0
	sp = decoder.SamplingParams{LogitProcessor: func([]int, []float32) { called++ }}
	s.applyBudget(&sp, nine, nil, 512, false)
	sp.LogitProcessor(nil, nil)
	if called != 1 || sp.LogitProcessorGate != nil {
		t.Errorf("an existing processor must be left exactly as it was (called=%d)", called)
	}
}
