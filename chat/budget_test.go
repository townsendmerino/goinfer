package chat

import (
	"math"
	"testing"
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
