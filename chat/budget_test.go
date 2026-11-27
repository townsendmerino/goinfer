package chat

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestReplySplitter_reasoningOnRuneBoundaries(t *testing.T) {
	tm := mustTemplate(t)
	rs := tm.NewReplySplitter(nil)
	if rs == nil {
		t.Fatal("a managed template must give a ReplySplitter")
	}
	reply := "<think>\ncafé été\n</think>\n\nOK é"
	var reasoning, answer string
	for i := 0; i < len(reply); i++ {
		r, a := rs.Push(reply[i : i+1])
		if r != "" && !validUTF8(r) {
			t.Fatalf("reasoning fragment %q is not valid UTF-8 (a rune was split)", r)
		}
		reasoning += r
		answer += a
	}
	r, a, truncated := rs.Finish()
	reasoning, answer = reasoning+r, answer+a
	if reasoning != "café été" || answer != "OK é" || truncated {
		t.Fatalf("reasoning=%q answer=%q truncated=%v", reasoning, answer, truncated)
	}
	rs = tm.NewReplySplitter(nil)
	rs.Push("<think>\nab\xc3")
	r, a, truncated = rs.Finish()
	if !truncated || a != "" || r == "" {
		t.Fatalf("truncated mid-rune: reasoning=%q answer=%q truncated=%v", r, a, truncated)
	}
	if ChatML().NewReplySplitter(nil) != nil {
		t.Fatal("a template with no recognised thinking control must not split")
	}
}

func validUTF8(s string) bool {
	for _, r := range s {
		if r == '\uFFFD' {
			return false
		}
	}
	return true
}

func mustTemplate(t *testing.T) *Template {
	t.Helper()
	for _, g := range loadThinkGoldens(t) {
		if g.Checkpoint == "qwen3-4b" {
			tm, err := Detect(Meta{ChatTemplate: g.ChatTemplate})
			if err != nil {
				t.Fatal(err)
			}
			return tm.WithThinking(ThinkTemplate)
		}
	}
	t.Fatal("no qwen3-4b golden")
	return nil
}

type mapLookup map[string]int

func (m mapLookup) TokenID(s string) (int, bool) { v, ok := m[s]; return v, ok }

func TestNewReasoningBudgetFor_harmony(t *testing.T) {
	ids := mapLookup{"<|channel|>": 5, "analysis": 35, "<|message|>": 8, "<|end|>": 7, "<|start|>": 6, "assistant": 73, "final": 17}
	for _, mode := range []ThinkMode{ThinkAsIs, ThinkTemplate, ThinkOn, ThinkOff} {
		tm := Harmony().WithThinking(mode)
		if !tm.ThinkingPossible() {
			t.Errorf("mode %v: gpt-oss always reasons, so thinking is always possible", mode)
		}
		b := tm.NewReasoningBudgetFor(ids, nil, 40)
		if b == nil {
			t.Fatalf("mode %v: no budget built", mode)
		}
		g := []int{5, 35, 8}
		for len(g) < 34 {
			g = append(g, 200)
		}
		if got := b.ForcedHarmony(g); got != 7 {
			t.Errorf("mode %v: at the trigger the first forced token is %d, want <|end|> (7)", mode, got)
		}
		if got := b.ForcedHarmony(append(slices.Clone(g), 7, 6, 73, 5)); got != 17 {
			t.Errorf("mode %v: the fifth forced token is %d, want final (17)", mode, got)
		}
	}
	for missing := range ids {
		partial := mapLookup{}
		for k, v := range ids {
			if k != missing {
				partial[k] = v
			}
		}
		if b := Harmony().NewReasoningBudgetFor(partial, nil, 40); b != nil {
			t.Errorf("a tokenizer without %q still got a budget", missing)
		}
	}
	if b := Harmony().NewReasoningBudgetFor(ids, nil, 8); b != nil {
		t.Error("limit 8 cannot hold the header and the forced sequence")
	}
}

func TestHarmonyBudget_realVocabulary(t *testing.T) {
	path := filepath.Join(os.Getenv("HOME"), "models", "gpt-oss-20b-MXFP4.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no %s", path)
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	b := Harmony().NewReasoningBudgetFor(tk, nil, 100)
	if b == nil {
		t.Fatal("no budget from gpt-oss's own tokenizer")
	}
	enc := func(segs ...Segment) []int {
		ids, err := tk.EncodeSegments(segs, false)
		if err != nil {
			t.Fatal(err)
		}
		return ids
	}
	sp := func(s string) Segment { return Segment{Text: s, Special: true} }
	ct := func(s string) Segment { return Segment{Text: s} }
	if want := enc(sp(hmEnd), sp(hmStart), ct("assistant"), sp(hmChannel), ct("final"), sp(hmMessage)); !equalInts(b.Force(), want) {
		t.Errorf("forced sequence %v, the tokenizer encodes the same text as %v", b.Force(), want)
	}
	if want := enc(sp(hmChannel), ct("analysis"), sp(hmMessage)); !equalInts(b.Head(), want) {
		t.Errorf("analysis header %v, the tokenizer encodes the same text as %v", b.Head(), want)
	}
	c := Harmony().NewCallBudgetFor(tk, nil, 100)
	if c == nil {
		t.Fatal("no call budget from gpt-oss's own tokenizer")
	}
	id := func(s string) int {
		v, ok := tk.TokenID(s)
		if !ok {
			t.Fatalf("no single token for %q", s)
		}
		return v
	}
	head := []int{id(hmEnd), id(hmStart), id("assistant"), id(hmChannel)}
	if len(c.Force()) <= len(head) || !equalInts(c.Force()[:len(head)], head) || decoded(t, tk, c.Force()[len(head):]) != harmonyCallRecipient {
		t.Errorf("call budget forced %v, want %v then the text %q", c.Force(), head, harmonyCallRecipient)
	}
	if want := enc(sp(hmEnd), sp(hmStart), ct("assistant"), sp(hmChannel), ct(harmonyCallRecipient)); !equalInts(c.Force(), want) {
		t.Errorf("call budget forced %v; whole header encodes to %v", c.Force(), want)
	}
}

type segLookup struct{ mapLookup }

func (s segLookup) EncodeSegments(segs []Segment, _ bool) ([]int, error) {
	var out []int
	for _, sg := range segs {
		if sg.Text == "commentary to=functions." {
			out = append(out, 40, 41, 42, 43)
			continue
		}
		id, ok := s.TokenID(sg.Text)
		if !ok {
			return nil, fmt.Errorf("no token for %q", sg.Text)
		}
		out = append(out, id)
	}
	return out, nil
}

func TestNewCallBudgetFor(t *testing.T) {
	ids := segLookup{mapLookup{"<|channel|>": 5, "analysis": 35, "<|message|>": 8, "<|end|>": 7, "<|start|>": 6, "assistant": 73, "final": 17}}
	b := Harmony().NewCallBudgetFor(ids, nil, 40)
	if b == nil {
		t.Fatal("no call budget for harmony")
	}
	if want := []int{7, 6, 73, 5, 40, 41, 42, 43}; !equalInts(b.Force(), want) {
		t.Errorf("forced %v, want the commentary call header %v", b.Force(), want)
	}
	if plain := Harmony().NewReasoningBudgetFor(ids, nil, 40); !equalInts(b.Head(), plain.Head()) || b.EndID() != plain.EndID() {
		t.Errorf("head %v end %d, want the plain budget's %v %d", b.Head(), b.EndID(), plain.Head(), plain.EndID())
	}
	g := []int{5, 35, 8}
	for len(g) < 32 {
		g = append(g, 200)
	}
	if got := b.ForcedHarmony(g); got != 7 {
		t.Errorf("at the trigger the first forced token is %d, want <|end|> (7)", got)
	}
	qwen, qids := mustTemplate(t), segLookup{mapLookup{"<think>": 51, "</think>": 52}}
	got, want := qwen.NewCallBudgetFor(qids, nil, 40), qwen.NewReasoningBudgetFor(qids, nil, 40)
	if want == nil || !reflect.DeepEqual(got, want) {
		t.Errorf("Qwen3: call budget %+v, plain budget %+v; want the same, non-nil", got, want)
	}
}

func decoded(t *testing.T, tk *tokenizer.Tokenizer, ids []int) string {
	t.Helper()
	s, err := tk.Decode(ids)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
