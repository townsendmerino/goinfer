package reasoning

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

func TestBudget_due(t *testing.T) {
	const open, closeID = 10, 11
	prompt := NewBudget(open, closeID, true, 3)
	self := NewBudget(open, closeID, false, 3)
	cases := []struct {
		name string
		b    *Budget
		gen  []int
		want bool
	}{
		{"prompt-opened: nothing generated yet", prompt, nil, false},
		{"prompt-opened: one token under the limit", prompt, []int{1, 2}, false},
		{"prompt-opened: exactly at the limit forces", prompt, []int{1, 2, 3}, true},
		{"prompt-opened: over the limit still forces", prompt, []int{1, 2, 3, 4}, true},
		{"prompt-opened: already closed is never forced", prompt, []int{1, closeID, 3, 4, 5}, false},
		{"prompt-opened: closed exactly at the limit", prompt, []int{1, 2, closeID}, false},
		{"self-opened: no open token, no budget", self, []int{1, 2, 3, 4, 5}, false},
		{"self-opened: counts from after the open token", self, []int{7, open, 1, 2}, false},
		{"self-opened: at the limit forces", self, []int{7, open, 1, 2, 3}, true},
		{"self-opened: closed before the limit", self, []int{open, 1, closeID, 2, 3, 4}, false},
		{"self-opened: opened, closed and answered is done", self, []int{open, 1, 2, 3, closeID, 9, 9, 9, 9}, false},
	}
	for _, tc := range cases {
		if got := tc.b.Due(tc.gen); got != tc.want {
			t.Errorf("%s: Due(%v) = %v, want %v", tc.name, tc.gen, got, tc.want)
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
					t.Errorf("%s: close token logit = %v, want 0", tc.name, v)
				}
				if i != closeID && !math.IsInf(float64(v), -1) {
					t.Errorf("%s: logit %d = %v, want -Inf", tc.name, i, v)
				}
			}
		} else if logits[3] != 0.3 {
			t.Errorf("%s: logits touched while not due", tc.name)
		}
	}
}

func harmonyTestBudget(limit int) *Budget {
	return NewHarmonyBudget([]int{10, 11, 12}, 7, []int{7, 6, 8, 10, 9, 12}, limit)
}

func TestHarmonyBudget_forcesWholeSequenceEndingAtLimit(t *testing.T) {
	b := harmonyTestBudget(20)
	gen := func(n int) []int {
		g := []int{10, 11, 12}
		for len(g) < n {
			g = append(g, 100+len(g))
		}
		return g[:n]
	}
	force := []int{7, 6, 8, 10, 9, 12}
	for n := 0; n <= 24; n++ {
		g := gen(n)
		if n >= 14 && n < 20 {
			g = append(gen(14), force[:n-14]...)
		} else if n >= 20 {
			g = append(append(gen(14), force...), make([]int, n-20)...)
		}
		want := -1
		if n >= 14 && n < 20 {
			want = force[n-14]
		}
		if got := b.ForcedHarmony(g); got != want {
			t.Errorf("after %d tokens: forces %d, want %d", n, got, want)
		}
		if b.Gate(g) != (want >= 0) {
			t.Errorf("after %d tokens: Gate = %v, forced = %d", n, b.Gate(g), want)
		}
	}
	g := append(gen(14), force[:2]...)
	logits := make([]float32, 16)
	for i := range logits {
		logits[i] = float32(i)
	}
	b.Process(g, logits)
	for i, v := range logits {
		if (i == 8) != (v == 0) || (i != 8 && !math.IsInf(float64(v), -1)) {
			t.Fatalf("Process left logit %d = %v", i, v)
		}
	}
	logits = []float32{1, 2, 3}
	b.Process(gen(5), logits)
	if logits[0] != 1 || logits[2] != 3 {
		t.Errorf("Process changed logits before due: %v", logits)
	}
	b.Process(gen(14), make([]float32, 5))
}

func TestHarmonyBudget_onlyWhenAnalysisStillOpen(t *testing.T) {
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
	if got := b.ForcedHarmony(body(10, -1)); got != 7 {
		t.Fatalf("open analysis must force <|end|>, got %d", got)
	}
	if got := b.ForcedHarmony(body(99, -1)); got != -1 {
		t.Errorf("non-analysis header forced %d", got)
	}
	if got := b.ForcedHarmony(body(10, 9)); got != -1 {
		t.Errorf("already-closed analysis forced %d", got)
	}
	if got := b.ForcedHarmony(body(10, 13)); got != -1 {
		t.Errorf("analysis closed at trigger-1 forced %d", got)
	}
	if got := b.ForcedHarmony([]int{10, 11}); got != -1 {
		t.Errorf("short reply forced %d", got)
	}
}

func TestHarmonyBudget_noRoom(t *testing.T) {
	if b := harmonyTestBudget(9); b != nil {
		t.Error("limit 9 leaves no reasoning token: want nil")
	}
	if b := harmonyTestBudget(10); b == nil {
		t.Error("limit 10 leaves one reasoning token: want non-nil")
	}
	if b := NewHarmonyBudget([]int{1}, 2, nil, 100); b != nil {
		t.Error("empty forced sequence must return nil")
	}
}
