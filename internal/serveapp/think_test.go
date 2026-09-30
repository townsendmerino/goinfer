package serveapp

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// thinkTokenizer is a byte-level vocab where every character the tests use is one token, plus "<think>" and "</think>" as
// whole tokens — so a reply can be fed as real special tokens OR spelled out a character at a time (a model is free to do
// either, and the splitter must not care).
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

// thinkIDs encodes text as ids; with whole, "<think>" / "</think>" become their single tokens, otherwise every character
// is its own token.
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

// templateFromGolden detects the real chat template of a checkpoint pinned in testdata/chat_think_goldens.
func templateFromGolden(t *testing.T, checkpoint string) *chat.Template {
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

type thinkRun struct {
	reasoning, content string
	finish             string
	ended              bool // the reply ended inside an unclosed block
}

// runThink drives ids through the real streamTokens with a thinkOut built from tm/ts.
func runThink(t *testing.T, tk *tokenizer.Tokenizer, ids []int, tm *chat.Template, ts thinkSettings, stops []string, maxTokens int) thinkRun {
	t.Helper()
	lm := &loadedModel{tk: tk}
	stream := make(chan int, len(ids))
	for _, id := range ids {
		stream <- id
	}
	close(stream)
	var r, c strings.Builder
	gr := genRequest{maxTokens: maxTokens, stopStrings: stops}
	gr.think = newThinkOut(tm, ts, func(s string) { r.WriteString(s) })
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	finish, _, _ := lm.streamTokens(ctx, cancel, stream, gr, nil, func(s string) { c.WriteString(s) })
	out := thinkRun{reasoning: r.String(), content: c.String(), finish: finish}
	if gr.think != nil {
		out.ended = gr.think.endedInReasoning
	}
	return out
}

const thinkAlphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?'-<>/|\n0123456789"

func TestStreamTokens_thinkSplit(t *testing.T) {
	tk, vocab := thinkTokenizer(t, thinkAlphabet)
	qwen3 := templateFromGolden(t, "qwen3-4b")
	nineB := templateFromGolden(t, "qwen3.5-9b").WithThinking(chat.ThinkTemplate) // prompt ends inside an open <think>
	small := templateFromGolden(t, "qwen3.5-0.8b").WithThinking(chat.ThinkTemplate)
	split := thinkSettings{format: rfSplit}

	cases := []struct {
		name  string
		tm    *chat.Template
		reply string
		wantR string
		wantC string
		ended bool
	}{
		{"qwen3 self-opened", qwen3, "<think>\nlet me see\n</think>\n\nThe answer is 4.", "let me see", "The answer is 4.", false},
		{"9B prompt opened the block: only the close is generated", nineB, "Okay the user wants a greeting.\n</think>\n\nHello!", "Okay the user wants a greeting.", "Hello!", false},
		{"0.8B closed prefill: the reply is all content", small, "Hello there.", "", "Hello there.", false},
		{"truncated inside the block: reasoning only, no answer", nineB, "Okay the user asks for a very long", "Okay the user asks for a very long", "", true},
		{"content that mentions the tag stays content", qwen3, "Use <think> tags to reason.", "", "Use <think> tags to reason.", false},
	}
	for _, tc := range cases {
		for _, whole := range []bool{true, false} {
			name := tc.name + map[bool]string{true: " / tag tokens", false: " / tag spelled out"}[whole]
			t.Run(name, func(t *testing.T) {
				got := runThink(t, tk, thinkIDs(vocab, tc.reply, whole), tc.tm, split, nil, 1000)
				if got.reasoning != tc.wantR || got.content != tc.wantC || got.ended != tc.ended {
					t.Fatalf("reply %q\n got reasoning=%q content=%q ended=%v\nwant reasoning=%q content=%q ended=%v",
						tc.reply, got.reasoning, got.content, got.ended, tc.wantR, tc.wantC, tc.ended)
				}
			})
		}
	}
}

// -reasoning-format none is nothing separated — the raw text, tags and all, exactly what serve sent before thinking was
// modelled — and deepseek-legacy is both at once.
func TestStreamTokens_thinkFormats(t *testing.T) {
	tk, vocab := thinkTokenizer(t, thinkAlphabet)
	tm := templateFromGolden(t, "qwen3-4b")
	reply := "<think>\nhmm\n</think>\n\nDone."
	ids := thinkIDs(vocab, reply, true)

	none := runThink(t, tk, ids, tm, thinkSettings{format: rfNone}, nil, 1000)
	if none.content != reply || none.reasoning != "" {
		t.Fatalf("none: content=%q reasoning=%q, want the raw reply and no reasoning", none.content, none.reasoning)
	}
	legacy := runThink(t, tk, ids, tm, thinkSettings{format: rfLegacy}, nil, 1000)
	if legacy.content != reply || legacy.reasoning != "hmm" {
		t.Fatalf("legacy: content=%q reasoning=%q, want raw content and reasoning %q", legacy.content, legacy.reasoning, "hmm")
	}
}

// A model with no recognised thinking control is never split, whatever the settings say.
func TestStreamTokens_thinkUnmanagedIsPassthrough(t *testing.T) {
	tk, vocab := thinkTokenizer(t, thinkAlphabet)
	for _, tm := range []*chat.Template{chat.ChatML(), nil} {
		got := runThink(t, tk, thinkIDs(vocab, "<think>\nx\n</think>\n\nAnswer", true), tm, thinkSettings{format: rfSplit}, nil, 1000)
		if got.content != "<think>\nx\n</think>\n\nAnswer" || got.reasoning != "" {
			t.Fatalf("template %v: content=%q reasoning=%q, want the raw text untouched", tm, got.content, got.reasoning)
		}
	}
}

// Stop strings are matched on the raw decoded text (think text included), BEFORE the split. A stop that appears in the
// answer ends the reply there; one that appears in the reasoning ends it there, with no answer. The second is a known
// limitation recorded in docs/tasks/task-qwen35-think-prompt-2026-09.md (hazard "stop strings are matched on raw decoded
// text"); this test pins today's behaviour so a change to it is deliberate, and is named for what it pins.
func TestStreamTokens_thinkStopStrings(t *testing.T) {
	tk, vocab := thinkTokenizer(t, thinkAlphabet)
	tm := templateFromGolden(t, "qwen3-4b")
	split := thinkSettings{format: rfSplit}

	got := runThink(t, tk, thinkIDs(vocab, "<think>\nhmm\n</think>\n\nOne. END Two.", true), tm, split, []string{"END"}, 1000)
	if got.content != "One. " || got.reasoning != "hmm" || got.finish != "stop" {
		t.Fatalf("stop in the answer: %+v", got)
	}
	got = runThink(t, tk, thinkIDs(vocab, "<think>\nthe word END appears\n</think>\n\nAnswer.", true), tm, split, []string{"END"}, 1000)
	if got.content != "" || got.finish != "stop" {
		t.Fatalf("KNOWN LIMITATION pinned: a stop string inside the reasoning ends the reply with no answer; got %+v", got)
	}
}

// ---------------------------------------------------------------------------------------------------------------------

func TestResolveThink(t *testing.T) {
	raw := func(s string) json.RawMessage { return json.RawMessage(s) }
	cases := []struct {
		name     string
		cfg      config
		tr       thinkRequest
		wantMode chat.ThinkMode
		explicit bool
		wantFmt  reasoningFormat
		wantErr  string
	}{
		{"nothing: the model's own template default, and deepseek", config{}, thinkRequest{}, chat.ThinkTemplate, false, rfSplit, ""},
		{"server asked for asis by name", config{thinking: "asis"}, thinkRequest{}, chat.ThinkAsIs, false, rfSplit, ""},
		{"server default thinking=off", config{thinking: "off"}, thinkRequest{}, chat.ThinkOff, false, rfSplit, ""},
		{"server default template", config{thinking: "template"}, thinkRequest{}, chat.ThinkTemplate, false, rfSplit, ""},
		{"server format none", config{reasoningFmt: "none"}, thinkRequest{}, chat.ThinkTemplate, false, rfNone, ""},
		{"kwargs enable_thinking true", config{}, thinkRequest{kwargs: map[string]json.RawMessage{"enable_thinking": raw("true")}}, chat.ThinkOn, true, rfSplit, ""},
		{"kwargs enable_thinking false overrides server on", config{thinking: "on"}, thinkRequest{kwargs: map[string]json.RawMessage{"enable_thinking": raw("false")}}, chat.ThinkOff, true, rfSplit, ""},
		{"kwargs win over effort", config{}, thinkRequest{kwargs: map[string]json.RawMessage{"enable_thinking": raw("false")}, reasoningEffort: "high"}, chat.ThinkOff, true, rfSplit, ""},
		{"effort none is off", config{}, thinkRequest{reasoningEffort: "none"}, chat.ThinkOff, true, rfSplit, ""},
		{"effort medium changes nothing (a bare reasoning_effort must not flip a client's prompt)", config{}, thinkRequest{reasoningEffort: "Medium"}, chat.ThinkTemplate, false, rfSplit, ""},
		{"effort high under server thinking=on stays on", config{thinking: "on"}, thinkRequest{reasoningEffort: "high"}, chat.ThinkOn, false, rfSplit, ""},
		{"anthropic enabled", config{}, thinkRequest{anthropicType: "enabled"}, chat.ThinkOn, true, rfSplit, ""},
		{"anthropic adaptive", config{}, thinkRequest{anthropicType: "adaptive"}, chat.ThinkOn, true, rfSplit, ""},
		{"anthropic disabled", config{}, thinkRequest{anthropicType: "disabled"}, chat.ThinkOff, true, rfSplit, ""},
		{"per-request format", config{}, thinkRequest{format: "deepseek-legacy"}, chat.ThinkTemplate, false, rfLegacy, ""},
		{"enable_thinking must be a bool", config{}, thinkRequest{kwargs: map[string]json.RawMessage{"enable_thinking": raw(`"yes"`)}}, 0, false, 0, "enable_thinking must be a boolean"},
		{"unknown format is a 400", config{}, thinkRequest{format: "xml"}, 0, false, 0, "reasoning_format"},
		{"unknown anthropic type is a 400", config{}, thinkRequest{anthropicType: "maybe"}, 0, false, 0, "thinking.type"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := &server{cfg: tc.cfg}
			ts, err := s.resolveThink(tc.tr)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("err = %v, want one containing %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if ts.mode != tc.wantMode || ts.explicit != tc.explicit || ts.format != tc.wantFmt {
				t.Fatalf("got mode=%v explicit=%v format=%v, want mode=%v explicit=%v format=%v", ts.mode, ts.explicit, ts.format, tc.wantMode, tc.explicit, tc.wantFmt)
			}
		})
	}
}

// The constrained-request rule: a prompt that ends inside an open think block contradicts a grammar that demands its
// first token, so constrainedTemplate renders such a request thinking-off — and leaves every other request alone.
func TestConstrainedTemplate(t *testing.T) {
	nine := templateFromGolden(t, "qwen3.5-9b")
	lm := &loadedModel{tmpl: nine.WithThinking(chat.ThinkTemplate)} // a server whose default opens the block
	user := []chat.Turn{{Role: "user", Content: "Hi"}}
	if !lm.tmpl.PromptOpensThink() {
		t.Fatal("setup: the 9B under ThinkTemplate must open the block")
	}
	ct := lm.constrainedTemplate(thinkSettings{})
	if ct.PromptOpensThink() || !strings.HasSuffix(ct.Render("", user), "<think>\n\n</think>\n\n") {
		t.Fatalf("constrained render still opens the block: %q", ct.Render("", user))
	}
	if !lm.templateFor(thinkSettings{}).PromptOpensThink() {
		t.Fatal("an unconstrained request must keep the server's mode")
	}
	// A model that never opens a block is returned unchanged.
	q3 := &loadedModel{tmpl: templateFromGolden(t, "qwen3-4b")}
	if q3.constrainedTemplate(thinkSettings{}) != q3.tmpl {
		t.Fatal("a template whose prompt does not open a block must pass through")
	}
}

func TestToolsConstrainedFromStart(t *testing.T) {
	tools := []chat.Tool{{Name: "a"}, {Name: "b"}}
	one := tools[:1]
	cases := []struct {
		name   string
		forced *chat.Tool
		named  bool
		union  string
		tools  []chat.Tool
		want   bool
	}{
		{"forced function", &tools[0], true, "", tools, true},
		{"lone tool", &one[0], false, "auto", one, true},
		{"required union", nil, false, "required", tools, true},
		{"auto union is lazy, reasoning before the opener is legal", nil, false, "auto", tools, false},
		{"named tool that matched nothing", nil, true, "", tools, false},
	}
	for _, tc := range cases {
		if got := toolsConstrainedFromStart(tc.forced, tc.named, tc.union, tc.tools); got != tc.want {
			t.Errorf("%s: got %v, want %v", tc.name, got, tc.want)
		}
	}
}

// The default, end to end through the template a server would load: with no -thinking flag each checkpoint renders exactly
// what HuggingFace renders from its own template with enable_thinking unset — 0.8B closed block, 9B open block, Qwen3
// nothing, Gemma 4 its closed scaffold — and `-thinking asis` still renders the pre-thinking bytes (no block at all).
func TestDefaultThinkingIsTheTemplatesOwn(t *testing.T) {
	user := []chat.Turn{{Role: "user", Content: "Hi"}}
	cases := []struct {
		ckpt     string
		wantTail string // the end of the default prompt
	}{
		{"qwen3.5-0.8b", "<|im_start|>assistant\n<think>\n\n</think>\n\n"},
		{"qwen3.5-9b", "<|im_start|>assistant\n<think>\n"},
		{"qwen3-4b", "<|im_start|>assistant\n"},
		{"gemma-4-26b-a4b-it", "<|turn>model\n<|channel>thought\n<channel|>"},
	}
	for _, tc := range cases {
		base := templateFromGolden(t, tc.ckpt)
		def := base.WithThinking(config{}.thinkDefault()).Render("", user)
		if !strings.HasSuffix(def, tc.wantTail) {
			t.Errorf("%s: default prompt ends %q, want it to end %q", tc.ckpt, def[max(0, len(def)-50):], tc.wantTail)
		}
		asis := base.WithThinking(config{thinking: "asis"}.thinkDefault()).Render("", user)
		if strings.Contains(asis, "<think>") {
			t.Errorf("%s: -thinking asis must not write a think block, got %q", tc.ckpt, asis[max(0, len(asis)-50):])
		}
	}
}
