package decoder

import (
	"context"
	"os"
	"strconv"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// BenchmarkPrefillLong times an end-to-end prefill of a long prompt via the
// batched M=K path (forwardLayersN) so a CPU profile shows where prefill time
// actually goes — specifically whether the per-position scalar attendQuery
// (QKᵀ + scores·V, the O(L²) term) is a hotspot vs the SIMD weight matmuls.
//
//	GOINFER_PREQUANT_GGUF=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
//	GOINFER_PREFILL_LEN=2048 go test ./decoder -run '^$' \
//	  -bench BenchmarkPrefillLong -benchtime 8x -cpuprofile prefill.cpu
func BenchmarkPrefillLong(b *testing.B) {
	m, err := loadBenchModel()
	if err != nil {
		b.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	linalg.SetParallelThreshold(DefaultDecodeParallelThreshold)

	L := 2048
	if s := os.Getenv("GOINFER_PREFILL_LEN"); s != "" {
		if v, err := strconv.Atoi(s); err == nil {
			L = v
		}
	}
	// prefillLogits auto-routes: batched forwardLayersN for canBatchN archs, else
	// the sequential per-token path (Gemma4 / MoE). Both are worth profiling.
	b.Logf("canBatchN(%d)=%v", L, m.canBatchN(L))
	// Synthetic but valid token ids — this benchmarks prefill SHAPE/cost, not
	// output correctness (bit-exactness is gated by the parity tests).
	vocab := m.w.arch.VocabSize
	prompt := make([]int, L)
	for i := range prompt {
		prompt[i] = (i*131 + 7) % vocab
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache := m.NewCache(L + 4)
		if _, err := m.prefillLogits(context.Background(), prompt, cache); err != nil {
			b.Fatalf("prefill: %v", err)
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(L)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}
