package decoder

import (
	"math"
	"math/rand"
	"testing"
)

// PRE-REGISTERED ACCEPTANCE RULES (written before the first run; docs/tasks/red-october.md R7b).
//
// The new temperature-only draw does not reproduce the old stream, so there is no reference stream to diff
// against; correctness is statistical, and that is a weaker instrument than token identity. The rules below
// are therefore fixed here, not tuned to the result:
//
//  1. Goodness of fit: draws from `gumbelDraw` against the EXACT softmax(l/T), chi-square, bins with expected
//     count < 20 merged into one. PASS iff the normal-approximated z = (chi2 - df)/sqrt(2 df) lies in
//     (-4.5, 4.5) — two-sided, so a sampler that is too regular fails as well as one that is skewed.
//     Seeds are fixed, so the outcome is deterministic; 4.5 sigma is ~3e-6 per side.
//  2. Equivalence to the old sampler: a two-sample chi-square (homogeneity) between `gumbelDraw` and the
//     legacy inverse-CDF `sampleChunked` on the same logits, same rule and threshold.
//  3. Tail: on a Zipf-shaped 32k vocabulary the empirical mass of tokens beyond rank 256 and each of the top
//     ten frequencies lie within 4.5 binomial sigma of their exact values.
//  4. The noise transform is checked against an f64 reference over a grid and random points (abs error
//     <= 4e-6, the f32 rounding of a value up to 23), is monotone non-increasing in h, and spans the range the
//     header of sampler_gumbel.go claims ([-3.2,-3.0] .. [22.8,23.0]).
//  5. The parallel implementation equals a sequential one on every row tried (argmax is exact — there is no
//     tolerance here, and no chunk-count dependence).
//  6. Invariances: same seed => same stream; logprobs on/off => same tokens; -Inf never drawn; flat logits
//     draw uniformly.
//
// A failure of 1-3 is a finding to investigate, not a threshold to relax.

func softmaxRef(logits []float32, T float64) []float64 {
	mx := math.Inf(-1)
	for _, v := range logits {
		mx = math.Max(mx, float64(v))
	}
	p := make([]float64, len(logits))
	var z float64
	for i, v := range logits {
		p[i] = math.Exp((float64(v) - mx) / T)
		z += p[i]
	}
	for i := range p {
		p[i] /= z
	}
	return p
}

func gumbelTestLogits(rng *rand.Rand, n int, scale float64) []float32 {
	l := make([]float32, n)
	for i := range l {
		l[i] = float32(rng.NormFloat64() * scale)
	}
	return l
}

// chiSquareZ returns the normal-approximated z of the chi-square statistic of `counts` against `probs`
// (n draws), merging bins whose expected count is < 20 into one.
func chiSquareZ(counts []int, probs []float64, n int) (z float64, df int) {
	var chi2, restObs, restExp float64
	bins := 0
	for i, p := range probs {
		exp := p * float64(n)
		if exp < 20 {
			restObs += float64(counts[i])
			restExp += exp
			continue
		}
		d := float64(counts[i]) - exp
		chi2 += d * d / exp
		bins++
	}
	if restExp >= 20 {
		d := restObs - restExp
		chi2 += d * d / restExp
		bins++
	}
	df = bins - 1
	return (chi2 - float64(df)) / math.Sqrt(2*float64(df)), df
}

func drawCounts(t *testing.T, logits []float32, T float64, seed int64, n int) []int {
	t.Helper()
	s := NewSampler(SamplingParams{Temperature: T, Seed: seed})
	c := make([]int, len(logits))
	for i := 0; i < n; i++ {
		info, err := s.SampleWithInfo(logits)
		if err != nil {
			t.Fatal(err)
		}
		c[info.ID]++
	}
	return c
}

// Rule 1.
func TestGumbelDraw_goodnessOfFit(t *testing.T) {
	const n = 1_000_000
	rng := rand.New(rand.NewSource(1))
	for _, T := range []float64{0.6, 1.0, 1.7} {
		logits := gumbelTestLogits(rng, 40, 2)
		z, df := chiSquareZ(drawCounts(t, logits, T, 7, n), softmaxRef(logits, T), n)
		t.Logf("T=%.1f df=%d z=%+.2f", T, df, z)
		if z <= -4.5 || z >= 4.5 {
			t.Errorf("T=%.1f: chi-square z=%+.2f (df=%d) outside (-4.5, 4.5) — the draw is not softmax(l/T)", T, z, df)
		}
	}
}

// Rule 2: the new draw and the legacy inverse-CDF draw sample the same distribution.
func TestGumbelDraw_equivalentToLegacyDistribution(t *testing.T) {
	const n = 1_000_000
	rng := rand.New(rand.NewSource(2))
	logits := gumbelTestLogits(rng, 200, 1.5)
	T := 1.0
	newC := drawCounts(t, logits, T, 11, n)
	old := NewSampler(SamplingParams{Temperature: T, Seed: 11})
	oldC := make([]int, len(logits))
	for i := 0; i < n; i++ {
		oldC[old.sampleChunked(logits, T, old.rng.Float64())]++
	}
	// two-sample chi-square with equal n: sum (a-b)^2/(a+b) ~ chi2(df), bins with a+b < 40 merged.
	var chi2, restA, restB float64
	bins := 0
	for i := range logits {
		a, b := float64(newC[i]), float64(oldC[i])
		if a+b < 40 {
			restA += a
			restB += b
			continue
		}
		chi2 += (a - b) * (a - b) / (a + b)
		bins++
	}
	if restA+restB >= 40 {
		chi2 += (restA - restB) * (restA - restB) / (restA + restB)
		bins++
	}
	df := bins - 1
	z := (chi2 - float64(df)) / math.Sqrt(2*float64(df))
	t.Logf("two-sample vs legacy: df=%d z=%+.2f", df, z)
	if z <= -4.5 || z >= 4.5 {
		t.Errorf("two-sample chi-square z=%+.2f (df=%d): the new draw's distribution differs from the legacy sampler's", z, df)
	}
}

// Rule 3.
func TestGumbelDraw_largeVocabTailAndHead(t *testing.T) {
	const v, n = 32768, 40000
	rng := rand.New(rand.NewSource(3))
	logits := make([]float32, v)
	for r, id := range rng.Perm(v) {
		logits[id] = float32(9 - 1.6*math.Log(float64(r+1)))
	}
	p := softmaxRef(logits, 1.0)
	c := drawCounts(t, logits, 1.0, 5, n)

	order := make([]int, v)
	for i := range order {
		order[i] = i
	}
	// rank by exact probability
	for i := 0; i < 256+10; i++ { // partial selection of the top 266 is enough
		best := i
		for j := i + 1; j < v; j++ {
			if p[order[j]] > p[order[best]] {
				best = j
			}
		}
		order[i], order[best] = order[best], order[i]
	}
	zOf := func(obs int, prob float64) float64 {
		mu := prob * n
		return (float64(obs) - mu) / math.Sqrt(n*prob*(1-prob))
	}
	for i := 0; i < 10; i++ {
		id := order[i]
		if z := zOf(c[id], p[id]); math.Abs(z) >= 4.5 {
			t.Errorf("rank %d token %d: observed %d, expected %.1f, z=%+.2f", i, id, c[id], p[id]*n, z)
		}
	}
	inHead := make(map[int]bool)
	var headMass float64
	var headObs int
	for i := 0; i < 256; i++ {
		inHead[order[i]] = true
		headMass += p[order[i]]
		headObs += c[order[i]]
	}
	tailMass := 1 - headMass
	tailObs := n - headObs
	z := zOf(tailObs, tailMass)
	t.Logf("tail (rank>256): exact mass %.4f, observed %.4f, z=%+.2f", tailMass, float64(tailObs)/n, z)
	if math.Abs(z) >= 4.5 {
		t.Errorf("tail mass z=%+.2f: the draw over/under-samples the tail", z)
	}
}

// Rule 4.
func TestGumbelNoise_transformAndRange(t *testing.T) {
	ref := func(h uint32) float64 {
		w := (float64(h) + 0.5) / 4294967296.0
		return -math.Log(-math.Log1p(-w))
	}
	if g := gumbelNoise(0); g < 22.8 || g > 23.0 {
		t.Errorf("G(h=0) = %v, want ~22.9", g)
	}
	if g := gumbelNoise(math.MaxUint32); g < -3.2 || g > -3.0 {
		t.Errorf("G(h=max) = %v, want ~-3.1", g)
	}
	rng := rand.New(rand.NewSource(4))
	prev := float32(math.Inf(1))
	var worst float64
	for i := 0; i < 2_000_000; i++ {
		var h uint32
		if i < 100_000 { // an ordered grid for the monotonicity check
			h = uint32(uint64(i) * (1 << 32) / 100_000)
		} else {
			h = rng.Uint32()
		}
		g := gumbelNoise(h)
		if e := math.Abs(float64(g) - ref(h)); e > worst {
			worst = e
		}
		if i < 100_000 {
			if g > prev {
				t.Fatalf("G not monotone non-increasing in h at h=%d: %v after %v", h, g, prev)
			}
			prev = g
		}
	}
	t.Logf("max |G_f32 - G_f64| over 2e6 points = %.2e", worst)
	if worst > 4e-6 {
		t.Errorf("noise transform error %.2e exceeds the f32 rounding bound 4e-6", worst)
	}
}

// Rule 5.
func TestGumbelDraw_parallelEqualsSequential(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	for _, v := range []int{7, 63, 4099, 32768, 151936} {
		logits := gumbelTestLogits(rng, v, 3)
		for _, T := range []float64{0.3, 1.0, 2.5} {
			s := NewSampler(SamplingParams{Temperature: T, Seed: 99})
			for d := 0; d < 8; d++ {
				seed, draw := s.gseed, s.gdraw
				got := s.gumbelDraw(logits, T)
				_, want := gumbelArgmaxRange(logits, 0, v, float32(1/T), [2]uint32{uint32(seed), uint32(seed >> 32)}, draw)
				if got != want {
					t.Fatalf("v=%d T=%.1f draw %d: parallel %d != sequential %d", v, T, d, got, want)
				}
			}
		}
	}
}

// Rule 6.
func TestGumbelDraw_invariances(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	logits := gumbelTestLogits(rng, 5000, 2)
	stream := func(p SamplingParams, n int) []int {
		s := NewSampler(p)
		out := make([]int, n)
		for i := range out {
			info, err := s.SampleWithInfo(logits)
			if err != nil {
				t.Fatal(err)
			}
			out[i] = info.ID
		}
		return out
	}
	base := SamplingParams{Temperature: 0.9, Seed: 21}
	a, b := stream(base, 300), stream(base, 300)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("same seed, different stream at %d", i)
		}
	}
	other := base
	other.Seed = 22
	c := stream(other, 300)
	same := 0
	for i := range a {
		if a[i] == c[i] {
			same++
		}
	}
	if same > 150 {
		t.Errorf("seeds 21 and 22 agree on %d/300 draws — the seed is not reaching the generator", same)
	}
	lp := base
	lp.Logprobs, lp.TopLogprobs = true, 3
	d := stream(lp, 300)
	for i := range a {
		if a[i] != d[i] {
			t.Fatalf("asking for logprobs changed the token at draw %d (%d vs %d): a request's tokens must not depend on it", i, a[i], d[i])
		}
	}
}

func TestGumbelDraw_edgeRows(t *testing.T) {
	s := NewSampler(SamplingParams{Temperature: 1.0, Seed: 3})
	// -Inf is never drawn.
	l := make([]float32, 64)
	for i := range l {
		l[i] = float32(math.Inf(-1))
	}
	l[17], l[40] = 0, 0.5
	for i := 0; i < 2000; i++ {
		if id := s.gumbelDraw(l, 1.0); id != 17 && id != 40 {
			t.Fatalf("drew masked token %d", id)
		}
	}
	// A fully masked row falls back to the argmax instead of panicking.
	for i := range l {
		l[i] = float32(math.Inf(-1))
	}
	if id := s.gumbelDraw(l, 1.0); id < 0 || id >= len(l) {
		t.Fatalf("fully masked row returned %d", id)
	}
	// Flat logits draw uniformly.
	flat := make([]float32, 50)
	c := drawCounts(t, flat, 1.0, 9, 500_000)
	probs := make([]float64, 50)
	for i := range probs {
		probs[i] = 1.0 / 50
	}
	if z, df := chiSquareZ(c, probs, 500_000); math.Abs(z) >= 4.5 {
		t.Errorf("flat logits: z=%+.2f (df=%d), not uniform", z, df)
	}
	// An absurdly small temperature is the argmax, not Inf arithmetic.
	if id := NewSampler(SamplingParams{Temperature: 1e-40, Seed: 1}).gumbelDraw([]float32{1, 9, 3}, 1e-40); id != 1 {
		t.Errorf("T=1e-40 drew %d, want the argmax 1", id)
	}
}
