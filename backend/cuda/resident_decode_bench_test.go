//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// BenchmarkResidentDecode times greedy resident decode (ForwardArgmax, the serve fast path) on CUDA
// after prefilling GOINFER_BENCH_DEPTH positions: one op is one decoded token. It is the
// speed-gate instrument for the q4k lane (goinfer against goinfer,
// separate processes per quant). Env: GOINFER_BENCH_MODEL (a .gguf), GOINFER_BENCH_QUANT,
// GOINFER_BENCH_ACT_GROUP (0 or 32), GOINFER_BENCH_DEPTH (default 128), GOINFER_BENCH_NO_FUSE (non-empty: fused kernels off), GOINFER_BENCH_FA_HD96 (non-empty: the hd-96 lane's
// opt-in). Prefill is outside the timer.
func BenchmarkResidentDecode(b *testing.B) {
	path := os.Getenv("GOINFER_BENCH_MODEL")
	if path == "" {
		b.Skip("set GOINFER_BENCH_MODEL")
	}
	depth, group := 128, 0
	if v, err := strconv.Atoi(os.Getenv("GOINFER_BENCH_DEPTH")); err == nil {
		depth = v
	}
	if v, err := strconv.Atoi(os.Getenv("GOINFER_BENCH_ACT_GROUP")); err == nil {
		group = v
	}
	opts := decoder.Options{Backend: "cuda", Quant: os.Getenv("GOINFER_BENCH_QUANT"),
		ActQuantGroup: group, ResidentContext: depth + 1024}
	opts.FlashDecodeHD96 = os.Getenv("GOINFER_BENCH_FA_HD96") != "" // the hd-96 lane's opt-in (docs/measurements/fa-hd96-phi3-2026-11-22)
	if os.Getenv("GOINFER_BENCH_NO_FUSE") != "" { // attribution: the per-row path with its fused kernels off
		opts.Knobs = &decoder.Knobs{"GOINFER_CUDA_NO_FUSE": "1"}
	}
	m, err := decoder.Load(path, opts)
	if err != nil {
		b.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	if rf == nil {
		b.Fatalf("resident did not engage: %s", m.ResidentDecline())
	}
	const id = 1000 // a fixed ordinary token; greedy decode feeds back its own argmax below
	pos := 0
	for ; pos < depth; pos++ {
		if _, err := rf.Forward(m.EmbedResidentForTest(id), pos); err != nil {
			b.Fatalf("prefill pos %d: %v", pos, err)
		}
	}
	r, ok := rf.(*cudaResident)
	if !ok {
		b.Fatalf("resident runner is %T, not *cudaResident", rf)
	}
	if b.N > 1024 {
		b.Fatalf("b.N=%d exceeds the resident context headroom (1024)", b.N)
	}
	next := id
	b.ResetTimer()
	t0 := time.Now()
	for i := 0; i < b.N; i++ {
		tok, err := r.ForwardArgmax(m.EmbedResidentForTest(next), pos)
		if err != nil {
			b.Fatalf("decode pos %d: %v", pos, err)
		}
		next, pos = tok, pos+1
	}
	b.ReportMetric(float64(b.N)/time.Since(t0).Seconds(), "tok/s")
}
