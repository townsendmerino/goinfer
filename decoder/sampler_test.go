package decoder

import (
	"math"
	"testing"
)

// The sampler has no HF RNG to match, so these validate its mechanics
// directly: greedy reduction, top-k/top-p support restriction, seed
// reproducibility, and that the empirical draw distribution matches the
// intended softmax (law of large numbers).

func TestSampler_greedy(t *testing.T) {
	logits := []float32{0.1, 3.0, -1, 2.9, 0.5}
	// Temperature 0 is greedy regardless of top-k/top-p.
	for _, sp := range []SamplingParams{
		{Temperature: 0},
		{Temperature: 0, TopK: 3},
		{Temperature: 0, TopP: 0.9},
	} {
		got, err := NewSampler(sp).Sample(logits)
		if err != nil {
			t.Fatalf("Sample(%+v): %v", sp, err)
		}
		if got != 1 {
			t.Errorf("Sample(%+v) = %d, want argmax 1", sp, got)
		}
	}
}

func TestSampler_topK1IsArgmax(t *testing.T) {
	logits := []float32{0.1, 3.0, -1, 2.9, 0.5}
	s := NewSampler(SamplingParams{Temperature: 1.0, TopK: 1, Seed: 7})
	for i := range 100 {
		if got, _ := s.Sample(logits); got != 1 {
			t.Fatalf("top-k=1 draw %d = %d, want argmax 1", i, got)
		}
	}
}

func TestSampler_reproducible(t *testing.T) {
	logits := []float32{1, 2, 3, 2, 1, 0, -1}
	seq := func(seed int64) []int {
		s := NewSampler(SamplingParams{Temperature: 1.2, TopK: 5, Seed: seed})
		out := make([]int, 64)
		for i := range out {
			out[i], _ = s.Sample(logits)
		}
		return out
	}
	a, b := seq(42), seq(42)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("same seed diverged at %d: %d vs %d", i, a[i], b[i])
		}
	}
	// A different seed should (almost surely) differ somewhere.
	c := seq(43)
	same := true
	for i := range a {
		if a[i] != c[i] {
			same = false
			break
		}
	}
	if same {
		t.Errorf("seeds 42 and 43 produced identical sequences")
	}
}

func TestSampler_topKsupport(t *testing.T) {
	logits := []float32{5, 4, 3, 2, 1, 0, -1, -2}
	s := NewSampler(SamplingParams{Temperature: 2.0, TopK: 3, Seed: 1})
	allowed := map[int]bool{0: true, 1: true, 2: true} // the 3 highest logits
	for range 5000 {
		got, _ := s.Sample(logits)
		if !allowed[got] {
			t.Fatalf("top-k=3 drew id %d outside the top-3 support", got)
		}
	}
}

func TestSampler_topPsupport(t *testing.T) {
	// softmax(temp 1) of these: id0≈0.64, id1≈0.23, rest small. top-p=0.8
	// nucleus is {0,1} (0.64 then 0.87 ≥ 0.8).
	logits := []float32{2, 1, -2, -3, -4}
	s := NewSampler(SamplingParams{Temperature: 1.0, TopP: 0.8, Seed: 3})
	allowed := map[int]bool{0: true, 1: true}
	for range 5000 {
		got, _ := s.Sample(logits)
		if !allowed[got] {
			t.Fatalf("top-p=0.8 drew id %d outside the nucleus {0,1}", got)
		}
	}
}

func TestSampler_distributionMatchesSoftmax(t *testing.T) {
	logits := []float32{2.0, 1.0, 0.0, -1.0}
	const temp = 1.3
	want := softmaxStable(logits, temp)
	s := NewSampler(SamplingParams{Temperature: temp, Seed: 12345})
	const N = 200000
	counts := make([]int, len(logits))
	for range N {
		id, _ := s.Sample(logits)
		counts[id]++
	}
	for i := range counts {
		got := float64(counts[i]) / N
		if d := math.Abs(got - want[i]); d > 0.01 {
			t.Errorf("id %d freq = %.4f, want softmax %.4f (Δ%.4f)", i, got, want[i], d)
		}
	}
}
