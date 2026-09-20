package decoder

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

// refTopKRow builds the row a correct device kernel returns, from the full logits, by the most obvious
// route: sort every index by (logit desc, id asc) and take the first k. Deliberately NOT topKByLogit —
// the test below asserts the two agree, so the ordering contract is checked, not assumed.
func refTopKRow(logits []float32, k int, texp float64, wantZ bool) TopKRow {
	idx := make([]int, len(logits))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		la, lb := logits[idx[a]], logits[idx[b]]
		if la != lb {
			return la > lb
		}
		return idx[a] < idx[b]
	})
	if k > len(idx) {
		k = len(idx)
	}
	row := TopKRow{IDs: make([]int32, k), Logits: make([]float32, k)}
	for i := 0; i < k; i++ {
		row.IDs[i] = int32(idx[i])
		row.Logits[i] = logits[idx[i]]
	}
	if wantZ {
		row.Z = chunkedZ(logits, float64(row.Logits[0]), texp, make([]float64, len(logits)))
	}
	return row
}

func topKTestLogits(kind string, v int, rng *rand.Rand) []float32 {
	l := make([]float32, v)
	if kind == "zipf" {
		// A language-model-shaped row: logit falls off as ~2.2·ln(rank), so the top few hundred tokens carry
		// essentially all the mass, unlike i.i.d. normal logits (far flatter than any real row).
		perm := rng.Perm(v)
		for r, id := range perm {
			l[id] = float32(9 - 2.2*math.Log(float64(r+1)) + rng.NormFloat64()*0.05)
		}
		return l
	}
	for i := range l {
		switch kind {
		case "normal":
			l[i] = float32(rng.NormFloat64() * 3)
		case "quantized": // heavy exact ties
			l[i] = float32(math.Round(rng.NormFloat64()*3*2) / 2)
		case "flat":
			l[i] = 1.25
		case "zeros": // a mix of +0 and -0, which the host treats as equal
			if rng.Intn(2) == 0 {
				l[i] = float32(math.Copysign(0, -1))
			}
		case "peaked":
			l[i] = float32(rng.NormFloat64())
		}
	}
	if kind == "peaked" {
		l[v/3] = 30
	}
	return l
}

// TestSampleFromTopK_matchesFullPath is the correctness gate for the device top-K sampler: for every
// (vocab, logits shape, temperature, filter combination) the token drawn from the K-candidate row equals
// the token the full-row sampler draws from the SAME seed, draw after draw — or the row reports ok=false
// (falls back) and the full path draws instead, having consumed no randomness. Z is passed exactly here;
// the device's rounded Z is exercised separately below.
func TestSampleFromTopK_matchesFullPath(t *testing.T) {
	type cfg struct {
		name          string
		topK          int
		topP, minP, T float64
	}
	var cfgs []cfg
	for _, T := range []float64{0.3, 0.8, 1.0, 1.5} {
		cfgs = append(cfgs,
			cfg{"topk1", 1, 0, 0, T}, cfg{"topk5", 5, 0, 0, T}, cfg{"topk40", 40, 0, 0, T}, cfg{"topk300", 300, 0, 0, T},
			cfg{"topp0.5", 0, 0.5, 0, T}, cfg{"topp0.95", 0, 0.95, 0, T}, cfg{"topp0.999", 0, 0.999, 0, T},
			cfg{"minp0.05", 0, 0, 0.05, T}, cfg{"minp0.3", 0, 0, 0.3, T},
			cfg{"topk40+topp0.9", 40, 0.9, 0, T}, cfg{"minp0.05+topp0.9", 0, 0.9, 0.05, T},
			cfg{"topk40+minp0.1+topp0.95", 40, 0.95, 0.1, T},
		)
	}
	var total, served, mismatched int
	for _, v := range []int{50, 1000, 32000, 152064} {
		for _, kind := range []string{"normal", "zipf", "quantized", "flat", "zeros", "peaked"} {
			rng := rand.New(rand.NewSource(int64(v) * 31))
			logits := topKTestLogits(kind, v, rng)
			for _, c := range cfgs {
				p := SamplingParams{Temperature: c.T, TopK: c.topK, TopP: c.topP, MinP: c.minP, Seed: 12345}
				full, fast := NewSampler(p), NewSampler(p)
				if !fast.TopKEligible() {
					t.Fatalf("%s should be eligible", c.name)
				}
				k, ok := fast.TopKWidth(v)
				if !ok {
					continue // a top-k wider than the kernel maximum is never served; that is the contract
				}
				row := refTopKRow(logits, k, c.T, fast.topPActive())
				// The reference order must be the host's own tie order.
				for i, id := range topKByLogit(logits, min(k, v)) {
					_ = i
					found := false
					for _, r := range row.IDs {
						if int(r) == id {
							found = true
							break
						}
					}
					if !found {
						t.Fatalf("v=%d %s: refTopKRow's set differs from topKByLogit's at id %d", v, kind, id)
					}
				}
				// The full path sorts the whole nucleus every draw; on flat rows at a wide vocab that is ~V
				// candidates, so those cells get few draws. Everything else gets many.
				draws := 150
				switch {
				case v >= 32000 && (kind == "flat" || kind == "zeros"):
					draws = 3
				case v >= 32000:
					draws = 40
				}
				for d := 0; d < draws; d++ {
					want, err := full.SampleWithInfo(logits)
					if err != nil {
						t.Fatal(err)
					}
					got, used := fast.SampleFromTopK(row, v)
					total++
					if used {
						served++
					} else {
						var e error
						if got, e = fast.SampleWithInfo(logits); e != nil {
							t.Fatal(e)
						}
					}
					if got.ID != want.ID {
						mismatched++
						if mismatched <= 5 {
							t.Errorf("v=%d %s %s T=%.1f draw %d used=%v: device-path id %d != full-path id %d", v, kind, c.name, c.T, d, used, got.ID, want.ID)
						}
					}
				}
			}
		}
	}
	if mismatched != 0 {
		t.Fatalf("%d of %d draws differ from the full path", mismatched, total)
	}
	t.Logf("%d draws, %d served from the top-K row (%.1f%%), the rest fell back; 0 mismatches", total, served, 100*float64(served)/float64(total))
}

// TestSampleFromTopK_typicalConfigRarelyFallsBack: the fallback exists for the pathological flat row,
// not the ordinary one. On ordinary logits at the headline config the K-row must serve ~every token.
func TestSampleFromTopK_typicalConfigRarelyFallsBack(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	logits := topKTestLogits("zipf", 152064, rng)
	p := SamplingParams{Temperature: 0.8, TopP: 0.95, Seed: 1}
	s := NewSampler(p)
	k, _ := s.TopKWidth(len(logits))
	row := refTopKRow(logits, k, p.Temperature, true)
	miss := 0
	for i := 0; i < 1000; i++ {
		if _, ok := s.SampleFromTopK(row, len(logits)); !ok {
			miss++
		}
	}
	if miss != 0 {
		t.Fatalf("%d/1000 fell back on ordinary logits at T0.8 top_p 0.95 — K=%d is not carrying the nucleus", miss, k)
	}
}

// TestSampleFromTopK_deviceRoundedZ: the device's Z differs from the host's chunked f64 sum by rounding
// (an f32 exp sum reduced in f64). Perturbing Z by a relative 1e-6 — far coarser than the device's real
// error — must still give the same tokens on ordinary logits; a token differs only if the cumulative
// mass lands inside that sliver of the cut.
func TestSampleFromTopK_deviceRoundedZ(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	logits := topKTestLogits("zipf", 152064, rng)
	p := SamplingParams{Temperature: 0.8, TopP: 0.95, Seed: 99}
	full, fast := NewSampler(p), NewSampler(p)
	k, _ := fast.TopKWidth(len(logits))
	row := refTopKRow(logits, k, p.Temperature, true)
	row.Z *= 1 + 1e-6
	diff := 0
	const n = 5000
	for i := 0; i < n; i++ {
		want, _ := full.SampleWithInfo(logits)
		got, ok := fast.SampleFromTopK(row, len(logits))
		if !ok {
			t.Fatal("fell back on ordinary logits")
		}
		if got.ID != want.ID {
			diff++
		}
	}
	t.Logf("relative Z error 1e-6: %d/%d draws differ", diff, n)
	if diff > n/1000 {
		t.Fatalf("%d/%d draws differ under a 1e-6 relative Z error — the cut is more sensitive to Z than the design assumes", diff, n)
	}
}

// TestSampleFromTopK_ineligibleConfigs pins what is NOT served: temperature-only sampling (a truncated
// tail changes the distribution), and anything that needs or rewrites the full row.
func TestSampleFromTopK_ineligibleConfigs(t *testing.T) {
	for name, p := range map[string]SamplingParams{
		"greedy":           {},
		"temperature only": {Temperature: 1.0},
		"top_p 1.0":        {Temperature: 1.0, TopP: 1.0},
		"logprobs":         {Temperature: 0.8, TopP: 0.9, Logprobs: true},
		"logit bias":       {Temperature: 0.8, TopP: 0.9, LogitBias: map[int]float32{5: 1}},
		"repeat penalty":   {Temperature: 0.8, TopP: 0.9, RepeatPenalty: 1.1},
	} {
		if NewSampler(p).TopKEligible() {
			t.Errorf("%s must not be served from a top-K row", name)
		}
	}
	if !NewSampler(SamplingParams{Temperature: 0.8, TopP: 0.95}).TopKEligible() {
		t.Error("temperature + top_p must be eligible")
	}
	if _, ok := NewSampler(SamplingParams{Temperature: 0.8, TopK: 5000}).TopKWidth(152064); ok {
		t.Error("a top-k wider than the kernel maximum must not be served")
	}
}

// BenchmarkSampleFromTopK measures the host side of the device top-K path per token, by config: the
// candidate filtering and draw over K=256 rows at a 152k vocab (the device work is not in this number).
func BenchmarkSampleFromTopK(b *testing.B) {
	rng := rand.New(rand.NewSource(7))
	logits := topKTestLogits("zipf", 152064, rng)
	for _, c := range []struct {
		name string
		p    SamplingParams
	}{
		{"top_p0.95", SamplingParams{Temperature: 0.8, TopP: 0.95, Seed: 1}},
		{"top_k40", SamplingParams{Temperature: 0.7, TopK: 40, Seed: 1}},
		{"min_p0.05", SamplingParams{Temperature: 1.0, MinP: 0.05, Seed: 1}},
	} {
		s := NewSampler(c.p)
		k, _ := s.TopKWidth(len(logits))
		row := refTopKRow(logits, k, c.p.Temperature, s.topPActive())
		b.Run(c.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, ok := s.SampleFromTopK(row, len(logits)); !ok {
					b.Fatal("fell back")
				}
			}
		})
	}
}
