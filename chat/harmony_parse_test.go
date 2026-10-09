package chat

import (
	"math/rand"
	"strings"
	"testing"
)

// A real gpt-oss-20b-MXFP4 reply, temperature 0, captured 2026-09-30 from goinfer-chat before this parser existed
// (docs/measurements/harmony-parser-2026-09-30/raw-gptoss20b-before-parser.log): the decoded stream as a client would have
// received it in `content`. It is NOT hand-written — the parser is held to what the model actually emits, including the
// narrow no-break spaces it puts in "17 × 3 = 51" and the two trailing spaces before a newline.
const realReply = "<|channel|>analysis<|message|>The user asks: \"What is 17 times 3? Then write one short sentence about the sea.\"\n\nWe need to answer the multiplication: 17 times 3 equals 51. Then write one short sentence about the sea. Provide short sentence about the sea. Probably something like \"The sea is vast and full of mysteries.\" Provide short sentence. Ensure short.<|end|><|start|>assistant<|channel|>final<|message|>17\u202f×\u202f3\u202f=\u202f51.  \nThe sea stretches endlessly, its waves whispering ancient secrets."
const realAnalysis = "The user asks: \"What is 17 times 3? Then write one short sentence about the sea.\"\n\nWe need to answer the multiplication: 17 times 3 equals 51. Then write one short sentence about the sea. Provide short sentence about the sea. Probably something like \"The sea is vast and full of mysteries.\" Provide short sentence. Ensure short."
const realFinal = "17\u202f×\u202f3\u202f=\u202f51.  \nThe sea stretches endlessly, its waves whispering ancient secrets."

type harmonyOut struct {
	reasoning, content string
	inReasoning        bool
	calls              []harmonyCall
}

// splitHarmony feeds chunks to a fresh parser, then flushes.
func splitHarmony(chunks ...string) harmonyOut {
	s := newHarmonySplitter()
	var r, c strings.Builder
	for _, ch := range chunks {
		a, b := s.Push(ch)
		r.WriteString(a)
		c.WriteString(b)
	}
	in := s.InReasoning()
	a, b := s.Flush()
	r.WriteString(a)
	c.WriteString(b)
	return harmonyOut{r.String(), c.String(), in, s.calls}
}

func TestHarmonySplitter_realGptOssReply(t *testing.T) {
	got := splitHarmony(realReply)
	if got.reasoning != realAnalysis {
		t.Errorf("reasoning:\n got %q\nwant %q", got.reasoning, realAnalysis)
	}
	if got.content != realFinal {
		t.Errorf("content:\n got %q\nwant %q", got.content, realFinal)
	}
	if got.inReasoning || len(got.calls) != 0 {
		t.Errorf("a finished reply is not in reasoning and made no calls: %+v", got)
	}
	if strings.Contains(got.content+got.reasoning, "<|") {
		t.Error("a marker leaked into a stream the client reads")
	}
}

// The table is the routing contract. Every row's input is what a reply's decoded text looks like (the turn stop is never in it).
func TestHarmonySplitter_routing(t *testing.T) {
	type want struct {
		reasoning, content string
		inReasoning        bool
		calls              []harmonyCall
	}
	cases := []struct {
		name, in string
		want     want
	}{
		{"analysis then final", "<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>final<|message|>answer",
			want{reasoning: "think", content: "answer"}},
		{"final only", "<|channel|>final<|message|>just this", want{content: "just this"}},
		{"a reply that ends after <|return|> text", "<|channel|>final<|message|>done<|return|>", want{content: "done"}},
		{"several analysis messages are joined by a blank line", "<|channel|>analysis<|message|>one<|end|><|start|>assistant<|channel|>analysis<|message|>two<|end|><|start|>assistant<|channel|>final<|message|>x",
			want{reasoning: "one\n\ntwo", content: "x"}},
		{"a commentary preamble is content, separated from the answer", "<|channel|>analysis<|message|>t<|end|><|start|>assistant<|channel|>commentary<|message|>Let me check.<|end|><|start|>assistant<|channel|>final<|message|>It is 18C.",
			want{reasoning: "t", content: "Let me check.\n\nIt is 18C."}},
		{"a function call is kept, and shown in neither stream", "<|channel|>analysis<|message|>need weather<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Paris\"}<|call|>",
			want{reasoning: "need weather", inReasoning: true, calls: []harmonyCall{{Channel: "commentary", Recipient: "functions.get_weather", Constrain: "json", Args: `{"city":"Paris"}`}}}},
		{"the recipient may sit in the role position", "<|start|>assistant to=functions.f<|channel|>commentary json<|message|>{}<|call|>",
			want{calls: []harmonyCall{{Channel: "commentary", Recipient: "functions.f", Args: "{}"}}}},
		{"an unknown channel is shown, not hidden", "<|channel|>thoughts<|message|>hmm", want{content: "hmm"}},
		{"no markers at all is content, verbatim", "  plain text, no harmony\n", want{content: "  plain text, no harmony\n"}},
		{"a body may mention the markers that do not end a message", "<|channel|>final<|message|>the <|channel|> and <|start|> tokens", want{content: "the <|channel|> and <|start|> tokens"}},
		{"a stray terminator with no message open is dropped", "<|end|><|channel|>final<|message|>x", want{content: "x"}},
		{"whitespace between messages is scaffolding", "<|channel|>analysis<|message|>a<|end|>\n<|start|>assistant<|channel|>final<|message|>b", want{reasoning: "a", content: "b"}},
		{"stray text before a message is content (the space before the marker is scaffolding)", "Hello <|channel|>final<|message|>x", want{content: "Hello\n\nx"}},
		{"trailing whitespace of a reply that just ends is kept", "plain \n", want{content: "plain \n"}},

		// Truncation (max_tokens): no answer, and the parser says so.
		{"cut off inside the analysis", "<|channel|>analysis<|message|>still thinking and then the", want{reasoning: "still thinking and then the", inReasoning: true}},
		{"cut off between analysis and answer", "<|channel|>analysis<|message|>done<|end|><|start|>assistant", want{reasoning: "done", inReasoning: true}},
		{"cut off inside the final header", "<|channel|>analysis<|message|>done<|end|><|start|>assistant<|channel|>fin", want{reasoning: "done", inReasoning: true}},
		{"cut off inside the answer is an answer", "<|channel|>analysis<|message|>d<|end|><|start|>assistant<|channel|>final<|message|>The sea is", want{reasoning: "d", content: "The sea is"}},
		{"cut off inside the very first header: nothing to show", "<|channel|>ana", want{}},
		{"empty", "", want{}},

		// What is held at the end of the reply is text, not lost.
		{"a partial terminator at the end of a body is text", "<|channel|>final<|message|>abc<|en", want{content: "abc<|en"}},
		{"a header with no <|message|> in reach is shown", "<|channel|>" + strings.Repeat("x", 600), want{content: "<|channel|>" + strings.Repeat("x", 600)}},
	}
	for _, tc := range cases {
		got := splitHarmony(tc.in)
		w := tc.want
		if got.reasoning != w.reasoning || got.content != w.content || got.inReasoning != w.inReasoning || len(got.calls) != len(w.calls) {
			t.Errorf("%s:\n in   %q\n got  reasoning=%q content=%q inReasoning=%v calls=%v\n want reasoning=%q content=%q inReasoning=%v calls=%v",
				tc.name, tc.in, got.reasoning, got.content, got.inReasoning, got.calls, w.reasoning, w.content, w.inReasoning, w.calls)
			continue
		}
		for i := range w.calls {
			if got.calls[i] != w.calls[i] {
				t.Errorf("%s: call %d = %+v, want %+v", tc.name, i, got.calls[i], w.calls[i])
			}
		}
	}
}

// Chunk-independence: the output must not depend on where the decoder happened to cut the reply. Every two-way split of every
// fixture, every three-way split of the small ones, and random partitions down to single bytes.
func TestHarmonySplitter_chunkIndependent(t *testing.T) {
	fixtures := []string{
		realReply,
		"<|channel|>analysis<|message|>one<|end|><|start|>assistant<|channel|>analysis<|message|>two<|end|><|start|>assistant<|channel|>commentary<|message|>pre<|end|><|start|>assistant<|channel|>final<|message|>fin<|return|>",
		"<|channel|>analysis<|message|>need<|end|><|start|>assistant<|channel|>commentary to=functions.f <|constrain|>json<|message|>{\"a\":1}<|call|>",
		"Hello <|channel|>final<|message|>x <|end|>\n<|start|>assistant<|channel|>final<|message|>y",
		"  stray <b> text <|sta without end",
		"<|end|><|return|><|call|><|channel|>final<|message|>a<|en",
		"<|channel|>final<|message|>multi-byte: é 日本語 🙂 <|end|>",
	}
	same := func(a, b harmonyOut) bool {
		if a.reasoning != b.reasoning || a.content != b.content || a.inReasoning != b.inReasoning || len(a.calls) != len(b.calls) {
			return false
		}
		for i := range a.calls {
			if a.calls[i] != b.calls[i] {
				return false
			}
		}
		return true
	}
	rng := rand.New(rand.NewSource(1))
	for _, f := range fixtures {
		whole := splitHarmony(f)
		for i := 0; i <= len(f); i++ {
			if got := splitHarmony(f[:i], f[i:]); !same(got, whole) {
				t.Fatalf("split at %d of %q:\n got  %+v\n want %+v", i, f, got, whole)
			}
		}
		if len(f) <= 140 {
			for i := 0; i <= len(f); i++ {
				for j := i; j <= len(f); j++ {
					if got := splitHarmony(f[:i], f[i:j], f[j:]); !same(got, whole) {
						t.Fatalf("split at %d,%d of %q:\n got  %+v\n want %+v", i, j, f, got, whole)
					}
				}
			}
		}
		for range 300 {
			var parts []string
			for rest := f; rest != ""; {
				k := 1 + rng.Intn(min(len(rest), 12))
				parts, rest = append(parts, rest[:k]), rest[k:]
			}
			if got := splitHarmony(parts...); !same(got, whole) {
				t.Fatalf("partition %q of %q:\n got  %+v\n want %+v", parts, f, got, whole)
			}
		}
		// And byte by byte.
		var bytewise []string
		for i := 0; i < len(f); i++ {
			bytewise = append(bytewise, f[i:i+1])
		}
		if got := splitHarmony(bytewise...); !same(got, whole) {
			t.Fatalf("byte-by-byte of %q:\n got  %+v\n want %+v", f, got, whole)
		}
	}
}

// Through the public type every decode loop holds: Harmony templates get the Harmony parser, everything else is unchanged, and
// Finish reports a reply cut off while thinking as truncated (serve turns that into finish_reason "length").
func TestReplySplitter_harmony(t *testing.T) {
	rs := Harmony().NewReplySplitter(nil)
	if rs == nil {
		t.Fatal("harmony must get a reply splitter: its output is channel messages, not plain text")
	}
	var r, c strings.Builder
	for i := 0; i < len(realReply); i += 7 {
		a, b := rs.Push(realReply[i:min(i+7, len(realReply))])
		r.WriteString(a)
		c.WriteString(b)
	}
	a, b, truncated := rs.Finish()
	r.WriteString(a)
	c.WriteString(b)
	if r.String() != realAnalysis || c.String() != realFinal || truncated {
		t.Errorf("via ReplySplitter: reasoning=%q content=%q truncated=%v", r.String(), c.String(), truncated)
	}

	cut := Harmony().NewReplySplitter(nil)
	cut.Push("<|channel|>analysis<|message|>the answer needs more thought and")
	if _, ans, trunc := cut.Finish(); !trunc || ans != "" {
		t.Errorf("a reply cut off in its analysis: truncated=%v answer=%q, want truncated with no answer", trunc, ans)
	}

	// Reasoning reaches the caller on rune boundaries: a 3-byte rune split across two pushes arrives whole.
	rs = Harmony().NewReplySplitter(nil)
	snow := "\u2603"
	p1, _ := rs.Push("<|channel|>analysis<|message|>a" + snow[:1])
	p2, _ := rs.Push(snow[1:] + "b")
	if p1 != "a" || p2 != snow+"b" {
		t.Errorf("reasoning split a rune: %q then %q", p1, p2)
	}

	if ChatML().NewReplySplitter(nil) != nil {
		t.Error("a template with no reasoning spec must still get no splitter")
	}
}
