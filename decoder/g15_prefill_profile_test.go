package decoder

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"runtime/pprof"
	"strings"
	"testing"
	"time"
)

// Prefill profiler — time and profile ONE batched CPU prefill at a chosen prompt
// length and quantization.
//
// Built for queue G15 (a suspected int4-specific prefill cliff at ~3k tokens).
// **G15 was WITHDRAWN: the cliff was a measurement artifact** — one 3020-token
// int4 timing of 1587.1 s that three later measurements put at ~350 s, int4 and
// int8int8 scaling alike at ~n^1.85. This instrument is what caught it, by
// disagreeing with the number, so it is kept: the disagreement was the finding.
//
// Two things it does that the original measurement did not, both of which are now
// required by `docs/benchmarks.md`'s methodology list:
//
//  1. It records MACHINE STATE beside the number (load average before and after).
//     A timing with no recorded state cannot be argued with later, which is what
//     made the artifact expensive rather than merely wrong.
//  2. It REFUSES to run on a busy box unless explicitly overridden, because the
//     artifact's leading suspect is an abandoned prefill still burning a core.
//
// The profile covers the PREFILL ONLY. Model load is excluded deliberately: it is
// seconds of unrelated I/O and quantization that would otherwise dominate a short
// profile and differ between arms by construction.
//
// Run (one quant per process — loadBenchModel is a sync.Once):
//
//	GOINFER_PREQUANT_GGUF=<model.gguf> GOINFER_BENCH_QUANT=int4 \
//	GOINFER_G15_K=3020 GOINFER_G15_PROF=/tmp/int4-3020.prof \
//	go test ./decoder/ -run TestG15PrefillProfile -timeout 3600s -v
//
// Then: go tool pprof -top -nodecount=30 <prof>
//
// Set GOINFER_G15_ALLOW_BUSY=1 to run anyway on a loaded box — and then say so
// beside any number it produces.

// loadAvg returns the 1/5/15-minute load averages as printed by uptime, or "" if
// they cannot be read. Best-effort by design: an unreadable load average must
// degrade the record, never fail the measurement.
func loadAvg() string {
	out, err := exec.Command("uptime").Output()
	if err != nil {
		return ""
	}
	line := string(out)
	i := strings.LastIndex(line, ":")
	if i < 0 {
		return strings.TrimSpace(line)
	}
	return strings.TrimSpace(line[i+1:])
}

// firstLoad parses the 1-minute load average out of loadAvg()'s text.
func firstLoad(s string) (float64, bool) {
	f := strings.FieldsFunc(s, func(r rune) bool { return r == ' ' || r == ',' })
	if len(f) == 0 {
		return 0, false
	}
	var v float64
	if _, err := fmt.Sscanf(f[0], "%f", &v); err != nil {
		return 0, false
	}
	return v, true
}

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
	// Pre-flight: a busy box is how the G15 artifact happened. Refuse rather than
	// silently produce a number nobody can later argue with.
	loadBefore := loadAvg()
	if v, ok := firstLoad(loadBefore); ok && v > 1.5 && os.Getenv("GOINFER_G15_ALLOW_BUSY") == "" {
		t.Skipf("box is not idle (load %.2f, want <=1.5): a prefill timing taken under load is not a "+
			"measurement. Wait for it to settle, or set GOINFER_G15_ALLOW_BUSY=1 and record the load "+
			"beside the number.", v)
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
		"prefill: quant=%s K=%d elapsed=%.1fs (%.1f tok/s) alloc=%.1f GB in %d GCs -> %s\n"+
			"  machine: load before [%s] after [%s] cpus=%d\n",
		quant, K, elapsed.Seconds(), float64(K)/elapsed.Seconds(),
		float64(after.TotalAlloc-before.TotalAlloc)/(1<<30), after.NumGC-before.NumGC, out,
		loadBefore, loadAvg(), runtime.NumCPU())
}
