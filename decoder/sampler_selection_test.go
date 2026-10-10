package decoder

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
)

// refTopFilter is the obviously-correct reference for topFilterLogits: a full stable sort with the tie-break (ascending
// id) and summation-order (descending) contracts applied. The optimized bounded path must reproduce it bit for bit. It
// is test-only. The old unstable-sort path is no gate target: no partial-selection implementation can reproduce an
// unspecified tied order.
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

// TestTopFilterLogits_MatchesReference is the exactness gate: across a wide seed sweep and two vocabulary sizes, the
// optimized bounded selection must equal the reference bit for bit (same retained ids in the same order, same
// renormalized probs).
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

	// The seed sweep runs in PARALLEL SHARDS: they partition the same seed range and assert the same things, so every case
	// still runs while wall time divides by the core count. Sharding rather than shrinking the sweep is deliberate: it is
	// pure computation, so -race finds nothing in it, and a smaller sweep would trade away the exactness gate that
	// justifies the optimization.
	//
	// Seed selection: every seed normally, an evenly strided subset under -race (sampler_sweep_race_test.go). Striding
	// rather than truncating keeps the selection spread across the range, so both logit shapes (tie-heavy / tie-free,
	// alternating on seed parity) stay represented; all 15 configs and all 4 temperatures run for every selected seed
	// either way. The check right after the loop is the guard the stride needs: a build that declares a stride must be
	// seen to apply it (a go fix rewrite of the loop once dropped it silently; docs/completed/task-ci-speed-2026-09.md, C8).
	var seedList []int
	for s := range seeds {
		if s%sweepSeedStride == 0 { // not `s += sweepSeedStride`: go fix rewrites that to `range seeds` where the stride is the constant 1, and drops it from the -race build too
			seedList = append(seedList, s)
		}
	}
	if sweepSeedStride > 1 && len(seedList) >= seeds {
		t.Fatalf("sweepSeedStride=%d but %d of %d seeds selected — the stride is declared and not applied", sweepSeedStride, len(seedList), seeds)
	}
	if sweepSeedStride == 1 && len(seedList) != seeds {
		t.Fatalf("full sweep declared (stride 1) but only %d of %d seeds selected", len(seedList), seeds)
	}

	// The shards are nested inside one group subtest: parallel subtests only run once their
	// PARENT returns, so without the group the totals below would be read before any case ran.
	const shards = 8
	var mu sync.Mutex
	total := 0
	t.Run("seed-sweep", func(t *testing.T) {
		for sh := range shards {
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
		for s := range 3 {
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
	got := topFilterLogits(logits, temp, topK, topP, minP, make([]float64, len(logits)), make([]int, len(logits)), make([]indexedProb, len(logits)))
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
			for seed := range int64(64) {
				optSampler := &Sampler{rng: rand.New(rand.NewSource(seed))}
				refDraw := drawFromRef(refTopFilter(logits, temp, c.k, c.p, c.m), rand.New(rand.NewSource(seed)))
				optDraw := optSampler.drawFiltered(topFilterLogits(logits, temp, c.k, c.p, c.m, optSampler.vocabBufN(len(logits)), optSampler.candBufN(len(logits)), optSampler.ipsBufN(len(logits))))
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

// TestSamplingThroughputGate asserts the top-p/top-k cliff is gone: sampling at temperature+top_p must run within a
// bounded factor of the TEMPERATURE-ONLY baseline (not greedy). A full-vocab sort regression shows up as ~7x; the
// gate factor sits below that and above the real ratio.
func TestSamplingThroughputGate(t *testing.T) {
	if testing.Short() {
		t.Skip("throughput gate: skipped under -short")
	}
	// A wall-clock ratio under the race detector measures the detector, not the sampler: the instrumentation does not
	// scale the two arms equally, so the gate would be permanently red with nothing regressed. ci.yml runs this test
	// WITHOUT -race on every push, which is where a real regression shows. Timing gates belong in an un-instrumented run.
	if raceEnabled {
		t.Skip("throughput gate: skipped under -race (the detector distorts wall clock; ci.yml runs " +
			"this test without -race on every push)")
	}
	// The denominator is the LEGACY chunked inverse-CDF draw (sampleChunked, kept unchanged in sampler_chunked_ref_test.go),
	// not production temperature-only sampling: a gate whose baseline moves measures two things at once, and the bound has
	// twice had to be re-anchored when the denominator got faster while the filtered path did not regress. Re-anchor with
	// a measurement, not a guess. The synthetic logits also overstate top_p's cost (randLogits is near-Gaussian, so a
	// 0.95 nucleus keeps a huge candidate set where real peaked logits keep a handful), so the ratio says nothing about
	// end-to-end speed. Record: docs/code-notes/decoder.md#TestSamplingThroughputGate.bound.
	const factor = 5.0
	for _, V := range []int{152064, 262144} {
		r := rand.New(rand.NewSource(1))
		logits := randLogits(V, r)

		base := benchLegacyTempOnly(logits, 0.8)
		topp := benchSample(logits, SamplingParams{Temperature: 0.8, TopP: 0.95})
		ratio := float64(topp) / float64(base)
		t.Logf("V=%d: temp-only %d ns/op, temp+top_p %d ns/op → %.2f×", V, base, topp, ratio)
		if ratio > factor {
			t.Errorf("V=%d: temp+top_p is %.2f× temp-only (gate %.1f×) — full-vocab selection has regressed", V, ratio, factor)
		}
	}
}

// benchSample times one sampling configuration, taking the BEST of three runs rather than one. testing.Benchmark's
// NsPerOp is a mean over b.N, and a mean tracks scheduling jitter upward (a floor, no ceiling), so the minimum is the
// least contaminated estimator of a floored quantity. It does not make the ratio machine-independent: x86 and arm64
// settle at different ratios on the same code, so the bound is effectively set by whichever machine runs hottest
// (arm64, at a thin margin).
func benchSample(logits []float32, p SamplingParams) int64 {
	best := int64(0)
	for range 3 {
		res := testing.Benchmark(func(b *testing.B) {
			s := NewSampler(p)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = s.SampleWithInfo(logits)
			}
		})
		if ns := res.NsPerOp(); best == 0 || ns < best {
			best = ns
		}
	}
	return best
}

// benchLegacyTempOnly is benchSample for the LEGACY temperature-only draw (sampleChunked), the yardstick
// TestSamplingThroughputGate's bar was set against.
func benchLegacyTempOnly(logits []float32, T float64) int64 {
	best := int64(0)
	for range 3 {
		res := testing.Benchmark(func(b *testing.B) {
			s := NewSampler(SamplingParams{Temperature: T})
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = s.sampleChunked(logits, T, s.rng.Float64())
			}
		})
		if ns := res.NsPerOp(); best == 0 || ns < best {
			best = ns
		}
	}
	return best
}

// --- before/after selection cost, reported in the commit message ---

func benchFilter(b *testing.B, V int, useRef bool) {
	r := rand.New(rand.NewSource(1))
	logits := randLogits(V, r)
	scratch := make([]float64, V) // reused across iterations, matching the real per-stream Sampler.vocabBuf
	candScratch := make([]int, V)
	ipsScratch := make([]indexedProb, V)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if useRef {
			_ = refTopFilter(logits, 0.8, 0, 0.95, 0)
		} else {
			_ = topFilterLogits(logits, 0.8, 0, 0.95, 0, scratch, candScratch, ipsScratch)
		}
	}
}

func BenchmarkFilterRef152k(b *testing.B) { benchFilter(b, 152064, true) }

// 32k is phi3-mini's vocab and is in the population on purpose: a scratch-reuse change validated only on the two
// large vocabs (152k, 262k) regressed this one.
func BenchmarkFilterRef32k(b *testing.B) { benchFilter(b, 32064, true) }
func BenchmarkFilterNew32k(b *testing.B) { benchFilter(b, 32064, false) }

// benchFilterFreshScratch is the pre-reuse shape: a fresh full-vocab buffer per call, as sampleChunked/chunkedZ
// allocated before the scratch was reused. Paired against benchFilter's reused scratch it isolates the scratch reuse
// itself; the Ref/New pair compares two different algorithms and cannot answer that.
func benchFilterFreshScratch(b *testing.B, V int) {
	r := rand.New(rand.NewSource(1))
	logits := randLogits(V, r)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = topFilterLogits(logits, 0.8, 0, 0.95, 0, make([]float64, V), make([]int, V), make([]indexedProb, V))
	}
}

func BenchmarkFilterFresh32k(b *testing.B)  { benchFilterFreshScratch(b, 32064) }
func BenchmarkFilterFresh152k(b *testing.B) { benchFilterFreshScratch(b, 152064) }
func BenchmarkFilterFresh262k(b *testing.B) { benchFilterFreshScratch(b, 262144) }

func BenchmarkFilterNew152k(b *testing.B) { benchFilter(b, 152064, false) }
func BenchmarkFilterRef262k(b *testing.B) { benchFilter(b, 262144, true) }
func BenchmarkFilterNew262k(b *testing.B) { benchFilter(b, 262144, false) }

// TestSweepCoverage_fullSweepRunsSomewhere is the gate on the gate. The exactness sweep strides under -race and both
// root CI jobs run -race, so the full 24,018-case sweep runs only because ci.yml carries an explicit non-race step for
// it: a coupling between a build tag and a YAML file, invisible from either side. Delete the step and the gate
// silently shrinks to a subset in every job, with nothing red. This reads ci.yml and fails if the step is gone; a
// string check, not a YAML parse, because what matters is that some step runs the test without -race.
func TestSweepCoverage_fullSweepRunsSomewhere(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", ".github", "workflows", "ci.yml"))
	if err != nil {
		t.Skipf("no ci.yml to check (%v) — this gate only applies in the repo", err)
	}
	ci := string(b)
	// Every sweep that strides itself under -race must be named on a non-race `go test` step: the exactness sweep (seed
	// stride) and the device top-K draw sweep (draw stride, TestSampleFromTopK_matchesFullPath in sampler_topk_test.go).
	// Add to this list when another sweep takes the same shape; a name missing here is a gate that silently shrank.
	strided := []string{"TestTopFilterLogits_MatchesReference", "TestSampleFromTopK_matchesFullPath"}
	found := map[string]bool{}
	for line := range strings.SplitSeq(ci, "\n") {
		s := strings.TrimSpace(line)
		if !strings.HasPrefix(s, "run:") || !strings.Contains(s, "go test") {
			continue
		}
		if strings.Contains(s, "-race") || !strings.Contains(s, "./decoder/") {
			continue
		}
		for _, name := range strided {
			if strings.Contains(s, name) {
				found[name] = true
			}
		}
	}
	for _, name := range strided {
		if !found[name] {
			t.Errorf("no CI step runs %s WITHOUT -race. Both root jobs use -race, where this sweep runs a "+
				"strided subset — so its exhaustive form is currently running nowhere. Add it to the "+
				"'sampler gates (no -race)' step (and its zero-match guard) in .github/workflows/ci.yml.", name)
		}
	}
}
