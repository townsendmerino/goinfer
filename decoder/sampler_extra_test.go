package decoder

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

// Tests for the completeness sampling controls: min-p, logit bias, the
// repetition/presence/frequency penalties, and logprob reporting.

func TestSampler_minP(t *testing.T) {
	// softmax(temp 1) of {5,4,0,-5}: id0≈0.727, id1≈0.268, id2≈0.005, ~0.
	logits := []float32{5, 4, 0, -5}
	// min-p 0.5 → threshold 0.5·0.727 = 0.363: only id0 survives.
	s := NewSampler(SamplingParams{Temperature: 1.0, MinP: 0.5, Seed: 1})
	for range 2000 {
		if got, _ := s.Sample(logits); got != 0 {
			t.Fatalf("min-p 0.5 drew %d, want only id0", got)
		}
	}
	// min-p 0.1 → threshold 0.0727: {id0, id1} survive, id2 excluded.
	s = NewSampler(SamplingParams{Temperature: 1.0, MinP: 0.1, Seed: 2})
	allowed := map[int]bool{0: true, 1: true}
	for range 5000 {
		if got, _ := s.Sample(logits); !allowed[got] {
			t.Fatalf("min-p 0.1 drew %d outside {0,1}", got)
		}
	}
}

func TestSampler_logitBias(t *testing.T) {
	logits := []float32{1, 2, 3} // greedy argmax = 2
	if got, _ := NewSampler(SamplingParams{}).Sample(logits); got != 2 {
		t.Fatalf("baseline greedy = %d, want 2", got)
	}
	// Ban the argmax → falls to id1.
	s := NewSampler(SamplingParams{LogitBias: map[int]float32{2: -100}})
	if got, _ := s.Sample(logits); got != 1 {
		t.Errorf("bias{2:-100} greedy = %d, want 1", got)
	}
	// Force id0.
	s = NewSampler(SamplingParams{LogitBias: map[int]float32{0: 100}})
	if got, _ := s.Sample(logits); got != 0 {
		t.Errorf("bias{0:+100} greedy = %d, want 0", got)
	}
	// Caller's slice is untouched.
	if logits[2] != 3 {
		t.Errorf("logit bias mutated the caller's slice: logits[2] = %v", logits[2])
	}
}

func TestSampler_penalties(t *testing.T) {
	logits := []float32{3.0, 2.9} // argmax 0 by a hair
	cases := []struct {
		name    string
		sp      SamplingParams
		observe []int
	}{
		{"repeat", SamplingParams{RepeatPenalty: 2.0, RepeatLastN: 64}, []int{0}}, // 3.0/2 = 1.5 < 2.9
		{"presence", SamplingParams{PresencePenalty: 0.5}, []int{0}},              // 3.0-0.5 = 2.5 < 2.9
		{"frequency", SamplingParams{FrequencyPenalty: 0.1}, []int{0, 0, 0}},      // 3.0-0.3 = 2.7 < 2.9
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := NewSampler(c.sp)
			s.Observe(c.observe...)
			if got, _ := s.Sample(logits); got != 1 {
				t.Errorf("%s penalty: greedy = %d, want 1 (id0 penalized below id1)", c.name, got)
			}
			// Without the seeded history the penalty has nothing to act on → argmax 0.
			fresh := NewSampler(c.sp)
			if got, _ := fresh.Sample(logits); got != 0 {
				t.Errorf("%s penalty, no history: greedy = %d, want 0", c.name, got)
			}
		})
	}
}

func TestSampler_repeatLastNWindow(t *testing.T) {
	logits := []float32{3.0, 2.9}
	// id0 is old (outside the window of 1) → not penalized → argmax stays 0.
	s := NewSampler(SamplingParams{RepeatPenalty: 2.0, RepeatLastN: 1})
	s.Observe(0, 1) // window of 1 = just [1]; id0 falls outside
	if got, _ := s.Sample(logits); got != 0 {
		t.Errorf("repeat-last-n=1: got %d, want 0 (id0 outside the window)", got)
	}
}

func TestSampler_logprobs(t *testing.T) {
	logits := []float32{2.0, 1.0, 0.0, -1.0}
	s := NewSampler(SamplingParams{Logprobs: true, TopLogprobs: 3}) // greedy
	info, err := s.SampleWithInfo(logits)
	if err != nil {
		t.Fatal(err)
	}
	if info.ID != 0 {
		t.Fatalf("greedy id = %d, want 0", info.ID)
	}
	// Logprob is over the full softmax at temperature 1 (greedy reports at 1).
	want := math.Log(softmaxStable(logits, 1)[0])
	if math.Abs(info.Logprob-want) > 1e-9 {
		t.Errorf("logprob = %v, want %v", info.Logprob, want)
	}
	// Top is prob-descending and headed by the argmax.
	if len(info.Top) != 3 {
		t.Fatalf("len(Top) = %d, want 3", len(info.Top))
	}
	if info.Top[0].ID != 0 {
		t.Errorf("Top[0].ID = %d, want 0 (argmax)", info.Top[0].ID)
	}
	for i := 1; i < len(info.Top); i++ {
		if info.Top[i].Logprob > info.Top[i-1].Logprob {
			t.Errorf("Top not descending at %d: %v > %v", i, info.Top[i].Logprob, info.Top[i-1].Logprob)
		}
	}
}

// referenceTopLogprobs is the pre-P-13 full-sort implementation, kept only here as
// an independent oracle: build every (id, prob) pair and sort all of them, rather
// than computeLogprobs' topKByLogit shortcut.
func referenceTopLogprobs(probs []float64, topN int) []TokenLogprob {
	type ip struct {
		id int
		p  float64
	}
	ips := make([]ip, len(probs))
	for i, p := range probs {
		ips[i] = ip{i, p}
	}
	sort.SliceStable(ips, func(a, b int) bool {
		if ips[a].p != ips[b].p {
			return ips[a].p > ips[b].p
		}
		return ips[a].id < ips[b].id // matches topKByLogit's smaller-id-wins tie-break
	})
	if topN > len(ips) {
		topN = len(ips)
	}
	out := make([]TokenLogprob, topN)
	for i := range out {
		out[i] = TokenLogprob{ID: ips[i].id, Logprob: math.Log(ips[i].p)}
	}
	return out
}

// TestComputeLogprobs_matchesFullSort is the P-13 gate: computeLogprobs' topKByLogit
// shortcut (O(V·log N)) must return exactly the same (id, logprob) pairs, in the same
// order, as a full O(V·log V) sort — across random logits, a range of topN including
// topN >= vocab, and a deliberate tie.
func TestComputeLogprobs_matchesFullSort(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := range 50 {
		n := 4 + rng.Intn(200)
		logits := make([]float32, n)
		for i := range logits {
			logits[i] = float32(rng.NormFloat64() * 5)
		}
		for _, topN := range []int{1, 3, n / 2, n, n + 5} {
			probs := softmaxStable(logits, 1)
			_, got := computeLogprobs(logits, 0, 1, topN)
			want := referenceTopLogprobs(probs, topN)
			if len(got) != len(want) {
				t.Fatalf("trial %d topN=%d: len(got)=%d, want %d", trial, topN, len(got), len(want))
			}
			for i := range want {
				if got[i].ID != want[i].ID {
					t.Fatalf("trial %d topN=%d idx %d: ID = %d, want %d (n=%d)", trial, topN, i, got[i].ID, want[i].ID, n)
				}
				if math.Abs(got[i].Logprob-want[i].Logprob) > 1e-12 {
					t.Fatalf("trial %d topN=%d idx %d: Logprob = %v, want %v", trial, topN, i, got[i].Logprob, want[i].Logprob)
				}
			}
		}
	}

	// A deliberate tie: ids 1 and 3 share the highest logit. Both the reference and
	// the fix must resolve it toward the smaller id.
	logits := []float32{0, 5, 1, 5, 2}
	_, got := computeLogprobs(logits, 0, 1, 2)
	if got[0].ID != 1 || got[1].ID != 3 {
		t.Fatalf("tie-break: got IDs [%d %d], want [1 3] (smaller id wins a logit tie)", got[0].ID, got[1].ID)
	}
}

// TestApplyPenalties_incrementalMatchesRebuild is the P-15 gate: applyPenalties'
// unbounded-window fast path (RepeatLastN ≤ 0) feeds applyPenaltiesFromCounts the
// incrementally maintained s.histCounts instead of rebuilding a counts map from
// s.history every call. This drives a mixed sequence of Observe (prompt-seed
// shape) and Sample (per-token draw shape) calls — both history-mutation sites —
// and after every step compares the fast path's logit output against the slow
// reference (applyPenaltiesOver called directly on a fresh rescan of s.history)
// bit for bit, with all three penalty kinds active so a miscount of ANY kind
// would show up numerically.
func TestApplyPenalties_incrementalMatchesRebuild(t *testing.T) {
	sp := SamplingParams{RepeatPenalty: 1.3, PresencePenalty: 0.4, FrequencyPenalty: 0.15}
	s := NewSampler(sp) // RepeatLastN unset -> unbounded -> the fast path under test
	base := []float32{1, -2, 3, 0.5, -1, 2, 0, -0.5}

	step := func(mutate func()) {
		mutate()
		fast := append([]float32(nil), base...)
		s.applyPenalties(fast)

		ref := NewSampler(sp)
		ref.history = append([]int(nil), s.history...) // same history, built by direct rescan below
		slow := append([]float32(nil), base...)
		ref.applyPenaltiesOver(slow, ref.history) // the pre-P-15 always-correct path

		for i := range fast {
			if fast[i] != slow[i] {
				t.Fatalf("history=%v: fast[%d]=%v, slow[%d]=%v (incremental histCounts diverged from a full rescan)", s.history, i, fast[i], i, slow[i])
			}
		}
	}

	step(func() { s.Observe(2, 5, 2, 0) }) // prompt-seed shape: multiple ids in one call
	step(func() { s.Observe(7) })
	if _, err := s.Sample(base); err != nil { // per-token draw shape: recordHistory via Sample
		t.Fatal(err)
	}
	step(func() {}) // no new mutation — re-checks the state Sample just recorded
	step(func() { s.Observe(2, 2, 2) })
}
