package decoder

import (
	"context"
	"math/rand/v2"
	"os"
	"strconv"
	"testing"
)

// decodeAtDepthOnce runs b.N decode steps from a cache already prefilled to
// depth, reporting tok/s. Factored out of BenchmarkDecodeAtDepth so R13's
// grouped-vs-ungrouped arms can share one prefill and interleave from it —
// prefill is the expensive setup (batched, O(depth)) and is per-model/depth,
// not per-arm, so re-running it per arm would be wasted time AND would risk
// the two arms landing on different machine states (this repo's own "paired
// differencing, arms adjacent in time" discipline, rules §6.4).
func decodeAtDepthOnce(b *testing.B, m *Model, cache *KVCache, seed int) {
	b.Helper()
	sampler := NewSampler(SamplingParams{Temperature: 0})
	logits, err := m.forward(seed, cache)
	if err != nil {
		b.Fatalf("seed forward: %v", err)
	}
	next, err := sampler.Sample(logits)
	if err != nil {
		b.Fatalf("seed sample: %v", err)
	}

	for b.Loop() {
		logits, err := m.forward(next, cache)
		if err != nil {
			b.Fatalf("forward: %v", err)
		}
		if next, err = sampler.Sample(logits); err != nil {
			b.Fatalf("sample: %v", err)
		}
	}
	b.StopTimer()

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

// BenchmarkDecodeAtDepth prefills to a target KV depth, then times b.N decode
// steps FROM there — unlike BenchmarkDecode (fixed short prompt, depth grows
// only as b.N ramps), this isolates decode cost AT a chosen context length, the
// axis P1 (the KV re-gather/V re-transpose in attendBatchedHeads) scales with.
//
// GOINFER_BENCH_DEPTH sets the prefill depth (default 2048). Same interleave
// discipline as BenchmarkDecode: do not compare two runs taken at different
// times; interleave arms in one session, discard the first sample.
//
// R13: two sub-benchmarks, "grouped_off"/"grouped_on" (GOINFER_ATTN_GROUPED),
// so `go test -bench` interleaves them within this one process — the
// served-decode-level sibling of aikit's own kernel A/B
// (BenchmarkMatmulQKAVGroupAB), which only measured the kernel in isolation.
func BenchmarkDecodeAtDepth(b *testing.B) {
	m, err := loadBenchModel()
	if err != nil {
		b.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	depth := 2048
	if d, err := strconv.Atoi(os.Getenv("GOINFER_BENCH_DEPTH")); err == nil {
		depth = d
	}
	tok := 785 // any valid id; content is decode-timing-irrelevant
	// GOINFER_BENCH_FILL=1 reaches depth by appending seeded random K/V rows to every layer instead of prefilling: decode attention's
	// work at a depth does not depend on the cached values, and a 3,900-row CPU prefill costs minutes a process. A dense model only
	// (every layer stores NumKVHeads x HeadDim a position).
	fill := os.Getenv("GOINFER_BENCH_FILL") == "1"
	// Otherwise reach depth via the BATCHED prefill path (forwardLayersN, K=depth): O(depth) via one wide matmul sweep, not
	// O(depth²) via depth sequential single-token forwards. It is also what production prefill does, so it is the
	// representative setup, not just the cheap one.
	if !fill && !m.canBatchN(depth) {
		b.Skipf("model does not support batched prefill (canBatchN(%d)=false); depth setup would be O(depth²)", depth)
	}
	ids := make([]int, depth)
	for i := range ids {
		ids[i] = tok
	}

	for _, arm := range []struct {
		name string
		env  string
	}{
		{"grouped_off", "0"},
		{"grouped_on", "1"},
	} {
		b.Run(arm.name, func(b *testing.B) {
			setKnob(b, m, knobAttnGrouped, arm.env)
			cache := m.NewCache(depth + b.N + 8)
			if fill {
				fillCacheForBench(b, m, cache, depth)
			} else if _, err := m.forwardLayersN(context.Background(), ids, cache, false); err != nil {
				b.Fatalf("batched prefill to depth %d: %v", depth, err)
			}
			decodeAtDepthOnce(b, m, cache, tok)
			// After the timed loop: its ResetTimer drops metrics reported before it.
			b.ReportMetric(float64(cache.rowPad), "rowpad")
			b.ReportMetric(float64(cache.f16InPlaceReads), "f16inplace")
		})
	}
}

// fillCacheForBench appends depth positions of seeded random K/V rows (uniform in [-1, 1)) to every layer of a dense model's cache.
func fillCacheForBench(b *testing.B, m *Model, cache *KVCache, depth int) {
	b.Helper()
	a := m.w.Arch
	if a.MLA != nil || cache.hasRecurrentState() {
		b.Skip("GOINFER_BENCH_FILL covers dense attention caches only")
	}
	kvDim := a.NumKVHeads * a.HeadDim
	rng := rand.New(rand.NewPCG(7, 11))
	k, v := make([]float32, kvDim), make([]float32, kvDim)
	for range depth {
		for l := range a.NumLayers {
			for i := range k {
				k[i], v[i] = 2*rng.Float32()-1, 2*rng.Float32()-1
			}
			cache.Append(l, k, v)
		}
	}
	if cache.Pos() != depth {
		b.Fatalf("filled cache holds %d positions, want %d", cache.Pos(), depth)
	}
}
