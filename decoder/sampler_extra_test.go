package decoder

import (
	"math"
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
