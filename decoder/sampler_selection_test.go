package decoder

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"testing"
)

// refTopFilter is the canonical, obviously-correct reference for topFilterLogits: a full
// stable sort with the tie-break (ascending id) and summation-order (descending) contracts
// applied. It is the GATE TARGET (amendment 1) — the optimized bounded path must reproduce
// it bit-for-bit. It is test-only and does NOT ship on the hot path; gating the optimized
// path against the OLD (unstable-sort) path would be meaningless, since no partial-selection
// implementation can reproduce an unspecified tied order.
func refTopFilter(logits []float32, temperature float64, topK int, topP, minP float64) []indexedProb {
	texp := temperature
	if texp <= 0 {
		texp = 1
	}
	maxL := float64(logits[0])
	for _, v := range logits[1:] {
		if float64(v) > maxL {
			maxL = float64(v)
		}
	}
	var Z float64
	for _, v := range logits {
		Z += math.Exp((float64(v) - maxL) / texp)
	}
	ips := make([]indexedProb, len(logits))
	for i, v := range logits {
		ips[i] = indexedProb{id: i, p: math.Exp((float64(v) - maxL) / texp)} // e (unnormalized)
	}
	sort.SliceStable(ips, func(a, b int) bool {
		if ips[a].p != ips[b].p {
			return ips[a].p > ips[b].p
		}
		return ips[a].id < ips[b].id // tie-break: ascending id
	})
	if topK > 0 && topK < len(ips) {
		ips = ips[:topK]
	}
	if minP > 0 && len(ips) > 0 {
		thresh := minP * ips[0].p
		cut := len(ips)
		for i := range ips {
			if ips[i].p < thresh {
				cut = i
				break
			}
		}
		if cut < 1 {
			cut = 1
		}
		ips = ips[:cut]
	}
	if topP > 0 && topP < 1 {
		target := topP * Z
		var cum float64
		cut := len(ips)
		for i := range ips {
			cum += ips[i].p
			if cum >= target {
				cut = i + 1
				break
			}
		}
		if cut < 1 {
			cut = 1
		}
		ips = ips[:cut]
	}
	nz := ips[:0]
	for _, ip := range ips {
		if ip.p > 0 {
			nz = append(nz, ip)
		}
	}
	ips = nz
	var sum float64
	for i := range ips {
		sum += ips[i].p
	}
	for i := range ips {
		ips[i].p /= sum
	}
	return ips
}

func randLogits(n int, r *rand.Rand) []float32 {
	l := make([]float32, n)
	for i := range l {
		// Peaked-ish: mostly small, a few large — the realistic decode shape.
		l[i] = float32(r.NormFloat64() * 4)
	}
	return l
}

// randLogitsWithTies deliberately injects duplicate logit values so the tie-break contract
// is actually exercised (random floats almost never collide).
func randLogitsWithTies(n int, r *rand.Rand) []float32 {
	l := randLogits(n, r)
	vals := []float32{l[0], l[1%n], l[2%n]}
	for i := range l {
		if r.Intn(3) == 0 {
			l[i] = vals[r.Intn(len(vals))]
		}
	}
	return l
}

// TestTopFilterLogits_MatchesReference is the exactness gate (amendment 1): across a wide
// seed sweep and two vocabulary sizes, the optimized bounded selection must equal the
// reference bit-for-bit — same retained ids in the same order, same renormalized probs.
func TestTopFilterLogits_MatchesReference(t *testing.T) {
	const seeds = 400
	vocabs := []int{152064, 262144} // qwen2.5 / gemma3 — the reporter's pair
	// Wider params on a small vocab (fast) to cover the parameter space densely, plus the
	// two real vocab sizes at the reported config.
	type cfg struct {
		topK       int
		topP, minP float64
	}
	cfgs := []cfg{
		{0, 0.95, 0}, {0, 0.9, 0}, {0, 0.5, 0}, {0, 0.999, 0}, {0, 1.0, 0},
		{40, 0, 0}, {1, 0, 0}, {200, 0, 0},
		{0, 0, 0.05}, {0, 0, 0.1}, {0, 0, 0.5},
		{40, 0.95, 0}, {50, 0, 0.05}, {40, 0.95, 0.02}, {0, 0.95, 0.05},
	}
	temps := []float64{0.8, 0.01, 1.0, 2.0}

	// The seed sweep is run in PARALLEL SHARDS. Every case still runs — the shards partition the
	// same seed range and assert the same things — but wall time divides by the core count.
	//
	// Why it matters: CI runs `go test -race`, and this sweep is pure computation with no
	// goroutines and no shared state, so the race detector finds nothing here while costing ~10×.
	// At 400 seeds × 15 cfgs × 4 temps that pushed the whole decoder package past the 600 s default
	// timeout on CI's slower runner (locally 63 s un-raced) — main went red on a TIMEOUT, not a
	// failure. Sharding is the coverage-neutral fix; reducing the sweep would have traded away the
	// exactness gate that justifies the optimization, which is the wrong thing to trade.
	// Seed selection: every seed normally, an evenly-STRIDED subset under -race (the detector has
	// nothing to find in this pure-compute sweep — see sampler_sweep_race_test.go). Striding rather
	// than truncating keeps the selection spread across the whole range, so both logit shapes
	// (tie-heavy / tie-free, which alternate on seed parity) stay represented. All 15 configs and
	// all 4 temperatures run for every selected seed either way.
	var seedList []int
	for s := 0; s < seeds; s += sweepSeedStride {
		seedList = append(seedList, s)
	}

	// The shards are nested inside one group subtest: parallel subtests only run once their
	// PARENT returns, so without the group the totals below would be read before any case ran.
	const shards = 8
	var mu sync.Mutex
	total := 0
	t.Run("seed-sweep", func(t *testing.T) {
		for sh := 0; sh < shards; sh++ {
			t.Run(fmt.Sprintf("shard-%d", sh), func(t *testing.T) {
				t.Parallel()
				n := 0
				for i := sh; i < len(seedList); i += shards {
					s := seedList[i]
					r := rand.New(rand.NewSource(int64(s) + 1))
					// small vocab: dense parameter coverage, with ties.
					var logits []float32
					if s%2 == 0 {
						logits = randLogitsWithTies(4096, r)
					} else {
						logits = randLogits(4096, r)
					}
					for _, c := range cfgs {
						for _, temp := range temps {
							assertSameFilter(t, logits, temp, c.topK, c.topP, c.minP, s)
							n++
						}
					}
				}
				mu.Lock()
				total += n
				mu.Unlock()
			})
		}
	})

	// Real vocab sizes at the reported config, a handful of seeds (these are big).
	for _, V := range vocabs {
		for s := 0; s < 3; s++ {
			r := rand.New(rand.NewSource(int64(s) + 7))
			logits := randLogitsWithTies(V, r)
			assertSameFilter(t, logits, 0.8, 0, 0.95, 0, s)
			assertSameFilter(t, logits, 0.8, 40, 0.95, 0, s)
			assertSameFilter(t, logits, 0.8, 0, 0, 0.05, s)
			total += 3
		}
	}
	// Report the MODE as well as the count: a reader of CI output must be able to tell the full
	// sweep from the strided one without inferring it from the number.
	ties, noTies := 0, 0
	for _, s := range seedList {
		if s%2 == 0 {
			ties++
		} else {
			noTies++
		}
	}
	if ties == 0 || noTies == 0 {
		t.Fatalf("seed stride %d selects only one logit shape (%d tie-heavy / %d tie-free) — the "+
			"subset no longer spans the space it claims to; the stride must be odd", sweepSeedStride, ties, noTies)
	}
	t.Logf("bit-for-bit identity confirmed over %d (logits,params) cases across %d seeds "+
		"(%d tie-heavy / %d tie-free) and vocab sizes %v — sweep: %s",
		total, len(seedList), ties, noTies, append(vocabs, 4096), sweepMode)
}

func assertSameFilter(t *testing.T, logits []float32, temp float64, topK int, topP, minP float64, seed int) {
	t.Helper()
	got := topFilterLogits(logits, temp, topK, topP, minP)
	want := refTopFilter(logits, temp, topK, topP, minP)
	if len(got) != len(want) {
		t.Fatalf("seed %d temp=%v k=%d p=%v minp=%v: len=%d, want %d", seed, temp, topK, topP, minP, len(got), len(want))
	}
	for i := range want {
		if got[i].id != want[i].id {
			t.Fatalf("seed %d temp=%v k=%d p=%v minp=%v: [%d] id=%d, want %d", seed, temp, topK, topP, minP, i, got[i].id, want[i].id)
		}
		if got[i].p != want[i].p { // bit-for-bit: identical inputs, identical order → identical float64
			t.Fatalf("seed %d temp=%v k=%d p=%v minp=%v: [%d] id=%d p=%v, want %v (bit mismatch)",
				seed, temp, topK, topP, minP, i, got[i].id, got[i].p, want[i].p)
		}
	}
}

// TestSample_DrawIdentity confirms the whole draw (selection + CDF walk) is identical
// between the optimized path and the reference, for a fixed RNG seed — the parity contract
// (same token given the same RNG draw).
func TestSample_DrawIdentity(t *testing.T) {
	r := rand.New(rand.NewSource(99))
	logits := randLogitsWithTies(8192, r)
	for _, temp := range []float64{0.8, 0.01, 1.5} {
		for _, c := range []struct {
			k    int
			p, m float64
		}{{0, 0.95, 0}, {40, 0, 0}, {0, 0, 0.05}, {40, 0.95, 0.02}} {
			for seed := int64(0); seed < 64; seed++ {
				optSampler := &Sampler{rng: rand.New(rand.NewSource(seed))}
				refDraw := drawFromRef(refTopFilter(logits, temp, c.k, c.p, c.m), rand.New(rand.NewSource(seed)))
				optDraw := optSampler.drawFiltered(topFilterLogits(logits, temp, c.k, c.p, c.m))
				if optDraw != refDraw {
					t.Fatalf("temp=%v k=%d p=%v m=%v seed=%d: opt drew %d, ref drew %d", temp, c.k, c.p, c.m, seed, optDraw, refDraw)
				}
			}
		}
	}
}

// drawFromRef walks the reference's retained CDF with the same inverse-CDF rule drawFiltered uses.
func drawFromRef(ips []indexedProb, rng *rand.Rand) int {
	rv := rng.Float64()
	var cum float64
	for _, ip := range ips {
		cum += ip.p
		if rv < cum {
			return ip.id
		}
	}
	return ips[len(ips)-1].id
}

// TestSamplingThroughputGate asserts the top-p/top-k cliff is gone: sampling at
// temperature+top_p must run within a bounded factor of the TEMPERATURE-ONLY baseline
// (amendment 4 — gated against temp-only, not greedy). A full-vocab sort regression shows
// up as ~7× (the reported 100→15 tok/s); the gate factor sits below that and above the
// real post-fix ratio.
func TestSamplingThroughputGate(t *testing.T) {
	if testing.Short() {
		t.Skip("throughput gate: skipped under -short")
	}
	const factor = 3.0 // temp+top_p wall time must be ≤ 3× temp-only; the old full sort was ~7×
	for _, V := range []int{152064, 262144} {
		r := rand.New(rand.NewSource(1))
		logits := randLogits(V, r)

		base := benchSample(logits, SamplingParams{Temperature: 0.8})
		topp := benchSample(logits, SamplingParams{Temperature: 0.8, TopP: 0.95})
		ratio := float64(topp) / float64(base)
		t.Logf("V=%d: temp-only %d ns/op, temp+top_p %d ns/op → %.2f×", V, base, topp, ratio)
		if ratio > factor {
			t.Errorf("V=%d: temp+top_p is %.2f× temp-only (gate %.1f×) — full-vocab selection has regressed", V, ratio, factor)
		}
	}
}

func benchSample(logits []float32, p SamplingParams) int64 {
	res := testing.Benchmark(func(b *testing.B) {
		s := NewSampler(p)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = s.SampleWithInfo(logits)
		}
	})
	return res.NsPerOp()
}

// --- before/after selection cost, reported in the commit message ---

func benchFilter(b *testing.B, V int, useRef bool) {
	r := rand.New(rand.NewSource(1))
	logits := randLogits(V, r)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if useRef {
			_ = refTopFilter(logits, 0.8, 0, 0.95, 0)
		} else {
			_ = topFilterLogits(logits, 0.8, 0, 0.95, 0)
		}
	}
}

func BenchmarkFilterRef152k(b *testing.B) { benchFilter(b, 152064, true) }
func BenchmarkFilterNew152k(b *testing.B) { benchFilter(b, 152064, false) }
func BenchmarkFilterRef262k(b *testing.B) { benchFilter(b, 262144, true) }
func BenchmarkFilterNew262k(b *testing.B) { benchFilter(b, 262144, false) }
