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
// after prefilling GOINFER_BENCH_DEPTH positions: one op is one decoded token. It is the Phase 1b
// speed-gate instrument of docs/tasks/task-int4-weight-quality-2026-09.md (goinfer against goinfer,
// separate processes per quant). Env: GOINFER_BENCH_MODEL (a .gguf), GOINFER_BENCH_QUANT,
// GOINFER_BENCH_ACT_GROUP (0 or 32), GOINFER_BENCH_DEPTH (default 128). Prefill is outside the timer.
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
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: os.Getenv("GOINFER_BENCH_QUANT"),
		ActQuantGroup: group, ResidentContext: depth + 1024})
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
