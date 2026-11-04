package whisper

import (
	"math"
	"math/rand/v2"
	"slices"
	"sort"
	"testing"
)

// G-S14h3 of docs/tasks/task-multimodal-support-2026-10.md: the fallback loop and sampling, by invariants (a sampled attempt's random stream is torch's and cannot be reproduced here).

func samplingClip(t *testing.T) (*Transcriber, []float32) {
	t.Helper()
	tr, g := tinyTS(t)
	c := g.Cases["short8s"]
	return tr, tsClip(c.Samples, c.A, c.B)
}

func TestFallback_attemptsInOrderAndTheLastIsKept(t *testing.T) {
	tr, x := samplingClip(t)
	// a log-probability threshold nothing reaches: every window needs a fallback until the last temperature
	r, err := tr.TranscribeWith(x, TranscribeOptions{Temperatures: []float64{0, 0.2, 0.4}, LogprobThreshold: f64(10), Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Windows) == 0 {
		t.Fatal("no windows")
	}
	for _, w := range r.Windows {
		if !slices.Equal(w.Attempts, []float64{0, 0.2, 0.4}) {
			t.Errorf("window at %d attempted at %v, want [0 0.2 0.4]", w.Seek, w.Attempts)
		}
	}
	// a threshold everything clears: one attempt per window
	r, err = tr.TranscribeWith(x, TranscribeOptions{Temperatures: []float64{0, 0.2, 0.4}, LogprobThreshold: f64(-1000), Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range r.Windows {
		if len(w.Attempts) != 1 {
			t.Errorf("window at %d attempted %d times under a threshold it clears", w.Seek, len(w.Attempts))
		}
	}
	// the greedy first attempt is what a single-temperature run gives
	greedy, _ := tr.TranscribeWith(x, TranscribeOptions{})
	if !slices.Equal(r.Tokens, greedy.Tokens) {
		t.Errorf("an accepted first attempt differs from the greedy run")
	}
}

func TestSampling_seedTemperatureAndStructure(t *testing.T) {
	tr, x := samplingClip(t)
	opts := func(seed uint64, temp float64) TranscribeOptions {
		return TranscribeOptions{Temperatures: []float64{temp}, Seed: seed}
	}
	a, _ := tr.TranscribeWith(x, opts(1, 0.4))
	b, _ := tr.TranscribeWith(x, opts(1, 0.4))
	c, _ := tr.TranscribeWith(x, opts(2, 0.4))
	if !slices.Equal(a.Tokens, b.Tokens) {
		t.Error("the same seed gave different results")
	}
	if slices.Equal(a.Tokens, c.Tokens) {
		t.Error("two seeds gave the same result at temperature 0.4 over 60 tokens")
	}
	// structure: every segment opens on a timestamp, timestamps never decrease within a window's tokens, <|notimestamps|> never appears
	gc := tr.Gen
	tsBegin := gc.NoTimestamps + 1
	for _, r := range []*LongResult{a, c} {
		for _, s := range r.Segments {
			if s.Tokens[0] < tsBegin {
				t.Errorf("segment %v does not open on a timestamp", s.Tokens)
			}
			last := 0
			for _, tk := range s.Tokens {
				if tk == gc.NoTimestamps {
					t.Errorf("<|notimestamps|> in %v", s.Tokens)
				}
				if tk >= tsBegin {
					if tk < last {
						t.Errorf("timestamps decrease in %v", s.Tokens)
					}
					last = tk
				}
			}
		}
	}
	// at a vanishing temperature the draw is the argmax: the greedy result
	greedy, _ := tr.TranscribeWith(x, TranscribeOptions{})
	cold, _ := tr.TranscribeWith(x, opts(5, 1e-6))
	if !slices.Equal(cold.Tokens, greedy.Tokens) {
		t.Errorf("temperature 1e-6 differs from greedy")
	}
}

func TestSampling_topK(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	l := make([]float32, 300)
	for i := range l {
		l[i] = float32(rng.NormFloat64())
	}
	l[17] = float32(math.Inf(-1)) // an already-suppressed entry
	orig := slices.Clone(l)
	sampleRescale(l, 0.7)
	var kept []int
	for i, v := range l {
		if !math.IsInf(float64(v), -1) {
			kept = append(kept, i)
			if v != orig[i] {
				t.Fatalf("a kept score changed: %v -> %v", orig[i], v)
			}
		}
	}
	if len(kept) != topK {
		t.Fatalf("%d entries kept, want %d", len(kept), topK)
	}
	idx := make([]int, 0, len(orig))
	for i := range orig {
		idx = append(idx, i)
	}
	sort.SliceStable(idx, func(a, b int) bool { return orig[idx[a]] > orig[idx[b]] })
	want := slices.Clone(idx[:topK])
	slices.Sort(want)
	if !slices.Equal(kept, want) {
		t.Errorf("the kept set is not the top %d", topK)
	}
	for range 2000 {
		if i := sampleFrom(l, 0.7, rng); !slices.Contains(kept, i) {
			t.Fatalf("sampled %d outside the kept set", i)
		}
	}
}
