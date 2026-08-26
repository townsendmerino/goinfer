package decoder

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"
	"testing"
	"time"
)

// G15 instrument — profile ONE batched CPU prefill at a chosen prompt length and
// quantization, so the int4 cliff can be diffed against int8int8 at the same M.
//
// Why this exists rather than a one-off script: the cliff is a step between ~1.5k
// and ~3k tokens (int4: 1520 tok 99.9s -> 3020 tok 1587.1s, n^4.03; int8int8 over
// the same step 93.2s -> 334.9s, n^1.86, no cliff). The discriminating question is
// "what does int4 do at M=3020 that int8int8 does not", which needs the two arms
// measured identically — same code, same K, same box, one variable.
//
// The profile covers the PREFILL ONLY. Model load is excluded deliberately: it is
// seconds of unrelated I/O and quantization that would otherwise dominate a short
// profile and differ between the arms by construction.
//
// Run (one quant per process — loadBenchModel is a sync.Once):
//
//	GOINFER_PREQUANT_GGUF=<model.gguf> GOINFER_BENCH_QUANT=int4 \
//	GOINFER_G15_K=3020 GOINFER_G15_PROF=/tmp/int4-3020.prof \
//	go test ./decoder/ -run TestG15PrefillProfile -timeout 3600s -v
//
// Then: go tool pprof -top -nodecount=30 <prof>
func TestG15PrefillProfile(t *testing.T) {
	out := os.Getenv("GOINFER_G15_PROF")
	if out == "" {
		t.Skip("set GOINFER_G15_PROF=<path> (and GOINFER_G15_K, GOINFER_BENCH_QUANT) to profile a prefill")
	}
	K := 3020
	if v := os.Getenv("GOINFER_G15_K"); v != "" {
		if _, err := fmt.Sscanf(v, "%d", &K); err != nil {
			t.Fatalf("GOINFER_G15_K=%q: %v", v, err)
		}
	}
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	if !m.canBatchN(K) {
		t.Skipf("model has no batched prefill at K=%d", K)
	}

	ids := make([]int, K)
	for i := range ids {
		ids[i] = 785 // any valid id; content is irrelevant to prefill cost
	}
	cache := m.NewCache(K + 8)

	f, err := os.Create(out)
	if err != nil {
		t.Fatalf("create profile: %v", err)
	}
	defer f.Close()

	// Report the allocation delta alongside the profile: the GC hypothesis for the
	// cliff predicts a large one, and it costs nothing to answer here rather than
	// in a second run.
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)

	if err := pprof.StartCPUProfile(f); err != nil {
		t.Fatalf("start profile: %v", err)
	}
	start := time.Now()
	_, err = m.forwardLayersN(context.Background(), ids, cache)
	elapsed := time.Since(start)
	pprof.StopCPUProfile()
	if err != nil {
		t.Fatalf("prefill K=%d: %v", K, err)
	}
	runtime.ReadMemStats(&after)

	quant := os.Getenv("GOINFER_BENCH_QUANT")
	if quant == "" {
		quant = "int8int8 (bench default)"
	}
	fmt.Fprintf(os.Stderr,
		"G15 prefill: quant=%s K=%d elapsed=%.1fs (%.1f tok/s) alloc=%.1f GB in %d GCs -> %s\n",
		quant, K, elapsed.Seconds(), float64(K)/elapsed.Seconds(),
		float64(after.TotalAlloc-before.TotalAlloc)/(1<<30), after.NumGC-before.NumGC, out)
}
