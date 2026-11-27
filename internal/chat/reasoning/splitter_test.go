package reasoning

import (
	"math/rand/v2"
	"strings"
	"testing"
)

func splitAll(open, close string, forced bool, chunks []string) (reasoning, content string, inReasoning bool) {
	s := NewThinkSplitter(open, close, forced)
	var r, c strings.Builder
	for _, ch := range chunks {
		a, b := s.Push(ch)
		r.WriteString(a)
		c.WriteString(b)
	}
	inReasoning = s.InReasoning()
	a, b := s.Flush()
	r.WriteString(a)
	c.WriteString(b)
	return r.String(), c.String(), inReasoning
}

var thinkCases = []struct {
	name       string
	open, cls  string
	forced     bool
	reply      string
	wantR      string
	wantC      string
	wantInReas bool
}{
	{"qwen3 self-opened", "<think>", "</think>", false, "<think>\nlet me see\n</think>\n\nThe answer is 4.", "let me see", "The answer is 4.", false},
	{"9b forced-open, only the close is generated", "<think>", "</think>", true, "Okay, the user wants.\n</think>\n\nHello!", "Okay, the user wants.", "Hello!", false},
	{"0.8B closed prefill: the reply is all content", "<think>", "</think>", false, "Hello there.", "", "Hello there.", false},
	{"empty reasoning block", "<think>", "</think>", false, "<think>\n\n</think>\n\nDone.", "", "Done.", false},
	{"truncated inside the block", "<think>", "</think>", true, "Okay, the user asks for a long", "Okay, the user asks for a long", "", true},
	{"truncated, trailing newlines are trimmed", "<think>", "</think>", false, "<think>\nhmm\n\n", "hmm", "", true},
	{"an open tag inside content is literal and never re-enters", "<think>", "</think>", false, "<think>\nr\n</think>\n\nUse <think> tags.", "r", "Use <think> tags.", false},
	{"a reply that merely mentions the tag mid-text stays content", "<think>", "</think>", false, "The <think> tag opens reasoning.", "", "The <think> tag opens reasoning.", false},
	{"leading whitespace before the open tag", "<think>", "</think>", false, "  \n<think>\nr\n</think>\nA", "r", "A", false},
	{"model re-opens a block the prompt opened", "<think>", "</think>", true, "<think>\nr\n</think>\n\nA", "r", "A", false},
	{"content keeps its own leading space and inner newlines", "<think>", "</think>", false, "<think>\nr\n</think>\n\nA\n\nB ", "r", "A\n\nB ", false},
	{"reasoning keeps inner newlines", "<think>", "</think>", false, "<think>\nline1\n\nline2\n</think>\n\nA", "line1\n\nline2", "A", false},
	{"a close tag after the block closed is dropped", "<think>", "</think>", false, "<think>\nr\n</think>5</think> maybe? Or just 5.", "r", "5 maybe? Or just 5.", false},
	{"a close tag with no block before it is dropped", "<think>", "</think>", false, "work it out</think>\n\nDone.", "", "work it out\n\nDone.", false},
	{"two close tags in the answer", "<think>", "</think>", true, "r</think>A</think></think>B", "r", "AB", false},
	{"a partial close tag in the answer is literal text", "<think>", "</think>", false, "<think>\nr\n</think>\n\na </thin b", "r", "a </thin b", false},
	{"an answer ending on a partial close tag flushes it", "<think>", "</think>", false, "<think>\nr\n</think>\n\na </thi", "r", "a </thi", false},
	{"gemma4: a bare close after the answer is dropped", "<|channel>thought\n", "<channel|>", false, "<|channel>thought\nw\n<channel|>Final.<channel|>", "w", "Final.", false},
	{"partial open delimiter then plain text is content", "<think>", "</think>", false, "<thi is not a tag", "", "<thi is not a tag", false},
	{"a reply ending on a partial open delimiter flushes as content", "<think>", "</think>", false, "<thin", "", "<thin", false},
	{"whitespace-only reply flushes as content", "<think>", "</think>", false, " \n", "", " \n", false},
	{"gemma4 channel", "<|channel>thought\n", "<channel|>", false, "<|channel>thought\nwork it out\n<channel|>Final.", "work it out", "Final.", false},
	{"gemma4 closed scaffold: content direct", "<|channel>thought\n", "<channel|>", false, "Final.", "", "Final.", false},
	{"gemma4 empty thought", "<|channel>thought\n", "<channel|>", false, "<|channel>thought\n<channel|>Final.", "", "Final.", false},
}

func chunkings(s string) [][]string {
	var out [][]string
	out = append(out, []string{s})
	for i := 1; i < len(s); i++ {
		out = append(out, []string{s[:i], s[i:]})
	}
	bytewise := make([]string, 0, len(s))
	for i := 0; i < len(s); i++ {
		bytewise = append(bytewise, s[i:i+1])
	}
	out = append(out, bytewise)
	rng := rand.New(rand.NewPCG(uint64(len(s))*7919+1, 0))
	for range 40 {
		var parts []string
		for rest := s; rest != ""; {
			n := min(1+rng.IntN(6), len(rest))
			parts = append(parts, rest[:n])
			rest = rest[n:]
		}
		out = append(out, parts)
	}
	return out
}

func TestThinkSplitter_table(t *testing.T) {
	for _, tc := range thinkCases {
		t.Run(tc.name, func(t *testing.T) {
			r, c, in := splitAll(tc.open, tc.cls, tc.forced, []string{tc.reply})
			if r != tc.wantR || c != tc.wantC || in != tc.wantInReas {
				t.Fatalf("whole reply %q\n got reasoning=%q content=%q inReasoning=%v\nwant reasoning=%q content=%q inReasoning=%v",
					tc.reply, r, c, in, tc.wantR, tc.wantC, tc.wantInReas)
			}
		})
	}
}

func TestThinkSplitter_chunkIndependent(t *testing.T) {
	n := 0
	for _, tc := range thinkCases {
		wantR, wantC, wantIn := splitAll(tc.open, tc.cls, tc.forced, []string{tc.reply})
		for _, chunks := range chunkings(tc.reply) {
			r, c, in := splitAll(tc.open, tc.cls, tc.forced, chunks)
			n++
			if r != wantR || c != wantC || in != wantIn {
				t.Fatalf("%s: chunks %q\n got reasoning=%q content=%q in=%v\nwant reasoning=%q content=%q in=%v",
					tc.name, chunks, r, c, in, wantR, wantC, wantIn)
			}
		}
	}
	if n < 1000 {
		t.Fatalf("only %d chunkings exercised", n)
	}
}

func TestThinkSplitter_noDelimiterLeaks(t *testing.T) {
	for _, tc := range thinkCases {
		if tc.wantInReas || strings.Contains(tc.wantC, tc.open) {
			continue
		}
		r, c, _ := splitAll(tc.open, tc.cls, tc.forced, []string{tc.reply})
		for _, s := range []string{r, c} {
			if strings.Contains(s, tc.open) || strings.Contains(s, tc.cls) {
				t.Errorf("%s: delimiter leaked into %q", tc.name, s)
			}
		}
	}
}

func TestThinkSplitter_streamsContentAfterClose(t *testing.T) {
	s := NewThinkSplitter("<think>", "</think>", false)
	if r, c := s.Push("<think>\nab"); r != "ab" || c != "" {
		t.Fatalf("got %q %q", r, c)
	}
	if r, c := s.Push("c</think>\n\nHel"); r != "c" || c != "Hel" {
		t.Fatalf("got %q %q", r, c)
	}
	if r, c := s.Push("lo"); r != "" || c != "lo" {
		t.Fatalf("got %q %q", r, c)
	}
}
