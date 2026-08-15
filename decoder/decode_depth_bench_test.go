package decoder

import (
	"os"
	"strconv"
	"testing"
)

// BenchmarkDecodeAtDepth prefills to a target KV depth, then times b.N decode
// steps FROM there — unlike BenchmarkDecode (fixed short prompt, depth grows
// only as b.N ramps), this isolates decode cost AT a chosen context length, the
// axis P1 (the KV re-gather/V re-transpose in attendBatchedHeads) scales with.
//
// GOINFER_BENCH_DEPTH sets the prefill depth (default 2048). Same interleave
// discipline as BenchmarkDecode: do not compare two runs taken at different
// times; interleave arms in one session, discard the first sample.
func BenchmarkDecodeAtDepth(b *testing.B) {
	m, err := loadBenchModel()
	if err != nil {
		b.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	depth := 2048
	if d, err := strconv.Atoi(os.Getenv("GOINFER_BENCH_DEPTH")); err == nil {
		depth = d
	}
	cache := m.NewCache(depth + b.N + 8)
	tok := 785 // any valid id; content is decode-timing-irrelevant
	// Reach depth via the BATCHED prefill path (forwardLayersN, K=depth) — O(depth)
	// via one wide matmul sweep, not O(depth²) via depth sequential single-token
	// forwards (the naive setup timed out at depth 2048: ~248s, almost all setup).
	// This is also what production prefill actually does, so it is the
	// representative setup, not just the cheap one.
	if !m.canBatchN(depth) {
		b.Skipf("model does not support batched prefill (canBatchN(%d)=false); depth setup would be O(depth²)", depth)
	}
	ids := make([]int, depth)
	for i := range ids {
		ids[i] = tok
	}
	if _, err := m.forwardLayersN(ids, cache); err != nil {
		b.Fatalf("batched prefill to depth %d: %v", depth, err)
	}
	sampler := NewSampler(SamplingParams{Temperature: 0})
	logits, err := m.forward(tok, cache)
	if err != nil {
		b.Fatalf("seed forward: %v", err)
	}
	next, err := sampler.Sample(logits)
	if err != nil {
		b.Fatalf("seed sample: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
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
