package chat

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestBudgetRoom(t *testing.T) {
	cases := []struct {
		max, want int
		ok        bool
	}{{3, 0, false}, {4, 2, true}, {8, 6, true}, {160, 120, true}, {4096, 3072, true}}
	for _, tc := range cases {
		got, ok := BudgetRoom(tc.max)
		if ok != tc.ok || (ok && got != tc.want) {
			t.Errorf("BudgetRoom(%d) = (%d, %v), want (%d, %v)", tc.max, got, ok, tc.want, tc.ok)
		}
	}
}

// due/Process/Gate: the processor as a pure function of the generated ids.
func TestReasoningBudget_due(t *testing.T) {
	const open, closeID = 10, 11
	prompt := NewReasoningBudget(open, closeID, true, 3) // the prompt ended inside an open block
	self := NewReasoningBudget(open, closeID, false, 3)  // the model opens the block itself
	cases := []struct {
		name string
		b    *ReasoningBudget
		gen  []int
		want bool
	}{
		{"prompt-opened: nothing generated yet", prompt, nil, false},
		{"prompt-opened: one token under the limit", prompt, []int{1, 2}, false},
		{"prompt-opened: exactly at the limit forces", prompt, []int{1, 2, 3}, true},
		{"prompt-opened: over the limit still forces (a missed step)", prompt, []int{1, 2, 3, 4}, true},
		{"prompt-opened: already closed is never forced", prompt, []int{1, closeID, 3, 4, 5}, false},
		{"prompt-opened: closed exactly at the limit", prompt, []int{1, 2, closeID}, false},
		{"self-opened: no open token, no budget", self, []int{1, 2, 3, 4, 5}, false},
		{"self-opened: counts from after the open token", self, []int{7, open, 1, 2}, false},
		{"self-opened: at the limit forces", self, []int{7, open, 1, 2, 3}, true},
		{"self-opened: closed before the limit", self, []int{open, 1, closeID, 2, 3, 4}, false},
		{"self-opened: a block that opened, closed and answered is done", self, []int{open, 1, 2, 3, closeID, 9, 9, 9, 9}, false},
	}
	for _, tc := range cases {
		if got := tc.b.due(tc.gen); got != tc.want {
			t.Errorf("%s: due(%v) = %v, want %v", tc.name, tc.gen, got, tc.want)
		}
		if got := tc.b.Gate(tc.gen); got != tc.want {
			t.Errorf("%s: Gate(%v) = %v, want %v", tc.name, tc.gen, got, tc.want)
		}
		logits := make([]float32, 16)
		for i := range logits {
			logits[i] = float32(i) * 0.1
		}
		tc.b.Process(tc.gen, logits)
		if tc.want {
			for i, v := range logits {
				if i == closeID && v != 0 {
					t.Errorf("%s: the close token's logit was changed to %v, want 0", tc.name, v)
				}
				if i != closeID && !math.IsInf(float64(v), -1) {
					t.Errorf("%s: logit %d = %v, want -Inf (only the close token may survive)", tc.name, i, v)
				}
			}
		} else if logits[3] != 0.3 {
			t.Errorf("%s: logits were touched while the budget was not due", tc.name)
		}
	}
}

// ReplySplitter: the splitter plus a reasoning stream that only ever reports complete runes.
func TestReplySplitter_reasoningOnRuneBoundaries(t *testing.T) {
	tm := mustTemplate(t)
	rs := tm.NewReplySplitter(nil)
	if rs == nil {
		t.Fatal("a managed template must give a ReplySplitter")
	}
	reply := "<think>\ncafé été\n</think>\n\nOK é"
	var reasoning, answer string
	for i := 0; i < len(reply); i++ { // one byte at a time: every multi-byte rune is split
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
	// truncated inside the block, mid-rune: what is there is still reported, and Finish says so.
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
		if r == '�' {
			return false
		}
	}
	return true
}

// mustTemplate is the real Qwen3 template from the pinned goldens, detected the way serve detects it.
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

// ---- Harmony (gpt-oss) ----------------------------------------------------------------------------------------------------

func harmonyTestBudget(limit int) *ReasoningBudget {
	// head <|channel|> analysis <|message|> = 10 11 12; <|end|> = 7; force <|end|> <|start|> assistant <|channel|> final <|message|>.
	return NewHarmonyBudget([]int{10, 11, 12}, 7, []int{7, 6, 8, 10, 9, 12}, limit)
}

// The trigger is len(force) tokens before limit, so the forced sequence ENDS at limit and the answer keeps the rest of the turn.
// Each step of the sequence is forced in turn, and the model is free again after it.
func TestHarmonyBudget_forcesTheWholeSequenceEndingAtLimit(t *testing.T) {
	b := harmonyTestBudget(20) // trigger at generated index 14
	gen := func(n int) []int {
		g := []int{10, 11, 12}
		for len(g) < n {
			g = append(g, 100+len(g)) // body tokens (none is <|end|>)
		}
		return g[:n]
	}
	force := []int{7, 6, 8, 10, 9, 12}
	for n := 0; n <= 24; n++ {
		g := gen(n)
		if n >= 14 && n < 20 { // what the processor forced is what is in the stream
			g = append(gen(14), force[:n-14]...)
		} else if n >= 20 {
			g = append(append(gen(14), force...), make([]int, n-20)...)
		}
		want := -1
		if n >= 14 && n < 20 {
			want = force[n-14]
		}
		if got := b.forcedHarmony(g); got != want {
			t.Errorf("after %d generated tokens: forces %d, want %d", n, got, want)
		}
		if b.Gate(g) != (want >= 0) {
			t.Errorf("after %d tokens: Gate = %v but forced token is %d", n, b.Gate(g), want)
		}
	}

	// Process masks every logit but the forced one.
	g := append(gen(14), force[:2]...) // two forced tokens written: the third is next
	logits := make([]float32, 16)
	for i := range logits {
		logits[i] = float32(i)
	}
	b.Process(g, logits)
	for i, v := range logits {
		if (i == 8) != (v == 0) || (i != 8 && !math.IsInf(float64(v), -1)) {
			t.Fatalf("Process left logit %d = %v; only token 8 (assistant) may survive", i, v)
		}
	}
	// A step that is not due leaves the logits alone.
	logits = []float32{1, 2, 3}
	b.Process(gen(5), logits)
	if logits[0] != 1 || logits[2] != 3 {
		t.Errorf("Process changed logits before the budget was due: %v", logits)
	}
	// A forced id outside the vocabulary the logits cover is ignored rather than indexing past them.
	b.Process(gen(14), make([]float32, 5))
}

// Nothing is forced unless the reply is inside an analysis message the model has not closed itself.
func TestHarmonyBudget_onlyWhenTheAnalysisIsStillOpen(t *testing.T) {
	b := harmonyTestBudget(20)
	body := func(first int, endAt int) []int {
		g := []int{first, 11, 12}
		for len(g) < 14 {
			g = append(g, 100+len(g))
		}
		if endAt >= 0 {
			g[endAt] = 7
		}
		return g
	}
	if got := b.forcedHarmony(body(10, -1)); got != 7 {
		t.Fatalf("setup: an open analysis at the trigger must force <|end|>, got %d", got)
	}
	if got := b.forcedHarmony(body(99, -1)); got != -1 {
		t.Errorf("a reply that did not open with the analysis header was forced (%d): its first message is not what the budget bounds", got)
	}
	if got := b.forcedHarmony(body(10, 9)); got != -1 {
		t.Errorf("a model that closed its own analysis before the trigger was forced (%d)", got)
	}
	if got := b.forcedHarmony(body(10, 13)); got != -1 {
		t.Errorf("an <|end|> in the last token before the trigger is still the model's own close (%d)", got)
	}
	if got := b.forcedHarmony([]int{10, 11}); got != -1 {
		t.Errorf("a reply shorter than the header was forced (%d)", got)
	}
}

// A turn too small to hold the header, the forced sequence and one reasoning token gets no budget.
func TestHarmonyBudget_noRoom(t *testing.T) {
	// header 3 + force 6 + 1 reasoning token = 10 is the least limit that can work.
	if b := harmonyTestBudget(9); b != nil {
		t.Error("limit 9 leaves no reasoning token: want no budget")
	}
	if b := harmonyTestBudget(10); b == nil {
		t.Error("limit 10 leaves one reasoning token: want a budget")
	}
	if b := NewHarmonyBudget([]int{1}, 2, nil, 100); b != nil {
		t.Error("an empty forced sequence is not a budget")
	}
}

type mapLookup map[string]int

func (m mapLookup) TokenID(s string) (int, bool) { v, ok := m[s]; return v, ok }

// Through the template: gpt-oss thinks whatever the mode (its prompt has no off form), so the budget applies under every
// -thinking mode; a tokenizer missing one of the channel words gets none rather than a budget that forces a wrong id.
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
		for len(g) < 34 { // trigger = 40 - 6
			g = append(g, 200)
		}
		if got := b.forcedHarmony(g); got != 7 {
			t.Errorf("mode %v: at the trigger the first forced token is %d, want <|end|> (7)", mode, got)
		}
		if got := b.forcedHarmony(append(append([]int(nil), g...), 7, 6, 73, 5)); got != 17 {
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

// With gpt-oss's REAL vocabulary: every word of the protocol is one token, and what the budget forces is exactly the ids the
// tokenizer makes of the text — so a forced sequence is what the model itself would have written, not a lookalike. Skipped where
// the GGUF is not present (a skip is not a pass: CI does not run this).
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
		t.Fatal("no budget from gpt-oss's own tokenizer: a protocol word is not a single token")
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
	if want := enc(sp(hmEnd), sp(hmStart), ct("assistant"), sp(hmChannel), ct("final"), sp(hmMessage)); !equalInts(b.force, want) {
		t.Errorf("forced sequence %v, the tokenizer encodes the same text as %v", b.force, want)
	}
	if want := enc(sp(hmChannel), ct("analysis"), sp(hmMessage)); !equalInts(b.head, want) {
		t.Errorf("analysis header %v, the tokenizer encodes the same text as %v", b.head, want)
	}
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
