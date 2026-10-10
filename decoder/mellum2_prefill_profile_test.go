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

// Mellum2 long-prefill profiler: the ATTENTION-vs-MoE-FFN split of a batched MoE prefill at a chosen prompt
// length. A dense profile does not answer it: two levers attack different terms (f32 attention, the O(K²)
// attention term, which `--cpu-fast-attention` REFUSES on MoE; and expert-major batching, the O(K) per-row
// MoE FFN) and their shares move in opposite directions as K grows, so the question has a crossover. Mellum2
// is also 24/28 SLIDING-window layers at window 1024, so a dense model's attention share overstates its at
// every K past the window.
//
// It cannot go through loadBenchModel (that resolves GOINFER_PREQUANT_GGUF, a file asset; this checkpoint
// is a safetensors DIRECTORY), so it loads the directory the way mellum2_parity_test.go does.
//
// Run (one quant per process):
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_MELLUM_K=2048 \
//	GOINFER_MELLUM_PROF=/tmp/mellum-2048.prof \
//	go test ./decoder/ -run TestMellum2PrefillProfile -timeout 3600s -v
//
// Then: go tool pprof -top -nodecount=30 <prof>
//
// SWAP IS RECORDED, NOT ASSUMED AWAY: a run that pages produces a plausible, wrong profile. `vm.swapusage` is
// read before and after and printed with the result; a non-zero delta invalidates the run. It is deliberately
// NOT an RSS check, because darwin's UBC reclaims under pressure and RSS reports what survived, not what was
// asked for.
func TestMellum2PrefillProfile(t *testing.T) {
	out := os.Getenv("GOINFER_MELLUM_PROF")
	if out == "" {
		t.Skip("set GOINFER_MELLUM_PROF=<path> (and GOINFER_MELLUM_K, GOINFER_BENCH_QUANT) to profile a Mellum2 prefill")
	}
	requireHeavyModel(t)
	K := 2048
	if v := os.Getenv("GOINFER_MELLUM_K"); v != "" {
		if _, err := fmt.Sscanf(v, "%d", &K); err != nil {
			t.Fatalf("GOINFER_MELLUM_K=%q: %v", v, err)
		}
	}
	loadBefore := loadAvg()
	if other, found := competingRun(); found && os.Getenv("GOINFER_MELLUM_ALLOW_BUSY") == "" {
		t.Skipf("another goinfer process is running (%s): a prefill timing taken beside our own "+
			"competing work is how the withdrawn G15 cliff happened. Wait for it, or set "+
			"GOINFER_MELLUM_ALLOW_BUSY=1 and say so beside the number.", other)
	}

	// Default to the FULL checkpoint; GOINFER_MELLUM_CKPT points at a layer slice where the full one will not
	// fit. A 4-layer slice is representative for THIS model only: layer_types interleave on a period of 4
	// (s,s,s,f) and every layer is `sparse`, so layers [0,4) reproduce the 25% full_attention share and the MoE
	// geometry. A slice does NOT preserve the embedding's share of the total, and the LM head is excluded by
	// construction (forwardLayersN stops at the layer stack).
	path := assetPath(t, "GOINFER_MELLUM_CKPT")
	quant := os.Getenv("GOINFER_BENCH_QUANT")
	if quant == "" {
		quant = "int8int8"
	}

	// Load is minutes of I/O; say so as it happens (os.Stderr, since t.Logf is buffered until the function
	// returns) so a stalled load is distinguishable from a slow one.
	fmt.Fprintf(os.Stderr, "mellum2-profile: loading %s quant=%s (bf16 -> quantized)\n", path, quant)
	swapBefore := swapUsed()
	loadStart := time.Now()
	m, err := Load(path, Options{Quant: quant})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	fmt.Fprintf(os.Stderr, "mellum2-profile: loaded in %.1fs\n", time.Since(loadStart).Seconds())
	if m.w.arch.Name != "mellum" {
		t.Fatalf("arch = %q, want mellum", m.w.arch.Name)
	}
	// The point is the BATCHED path: a silent fall-through to the sequential one would profile a different
	// program and still print a number.
	if !m.canBatchN(K) {
		t.Fatalf("canBatchN(%d) = false: this profile is only meaningful on the batched path", K)
	}

	// TOKEN CONTENT IS NOT NEUTRAL HERE: on a MoE it decides ROUTING (near-identical rows select near-identical
	// experts, so the top-8 expert weights stay cache-resident and the FFN measures a best case), so the dense
	// harnesses' constant-id prompt is a trap. The two id patterns are the experiment's two ARMS:
	//
	// 	varied  — deterministic spread over the vocab; real routing diversity.
	// 	uniform — one repeated id; degenerate routing. The CONTROL, and the CEILING for expert-major
	// 	          batching (a chunk whose rows all select the same experts is what that lever manufactures).
	//
	// The arms are matched on everything else (same K, same shapes, same attention work), so the paired
	// difference isolates the routing term.
	pattern := os.Getenv("GOINFER_MELLUM_IDS")
	if pattern == "" {
		pattern = "varied"
	}
	vocab := m.w.arch.VocabSize
	ids := make([]int, K)
	switch pattern {
	case "varied":
		for i := range ids {
			ids[i] = (i*131 + 7) % vocab
		}
	case "uniform":
		for i := range ids {
			ids[i] = 785
		}
	default:
		t.Fatalf("GOINFER_MELLUM_IDS=%q: want varied or uniform", pattern)
	}
	cache := m.NewCache(K + 8)

	f, err := os.Create(out)
	if err != nil {
		t.Fatalf("create profile: %v", err)
	}
	defer f.Close()

	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)

	fmt.Fprintf(os.Stderr, "mellum2-profile: prefill K=%d ids=%s starting at %s\n", K, pattern, time.Now().Format("15:04:05"))
	if err := pprof.StartCPUProfile(f); err != nil {
		t.Fatalf("start profile: %v", err)
	}
	start := time.Now()
	_, err = m.forwardLayersN(context.Background(), ids, cache, cpuFastAttention())
	elapsed := time.Since(start)
	pprof.StopCPUProfile()
	if err != nil {
		t.Fatalf("prefill K=%d: %v", K, err)
	}
	runtime.ReadMemStats(&after)
	swapAfter := swapUsed()

	fmt.Fprintf(os.Stderr,
		"mellum2-prefill: quant=%s ids=%s K=%d elapsed=%.1fs (%.2f tok/s) alloc=%.1f GB in %d GCs -> %s\n"+
			"  machine: load before [%s] after [%s] cpus=%d swap used %s -> %s\n",
		quant, pattern, K, elapsed.Seconds(), float64(K)/elapsed.Seconds(),
		float64(after.TotalAlloc-before.TotalAlloc)/(1<<30), after.NumGC-before.NumGC, out,
		loadBefore, loadAvg(), runtime.NumCPU(), swapBefore, swapAfter)
}

// swapUsed returns the darwin `vm.swapusage` "used" field, or "" if it cannot be
// read. Best-effort by design: an unreadable swap figure must degrade the record,
// never fail the measurement — the same contract loadAvg keeps.
func swapUsed() string {
	out, err := exec.Command("sysctl", "-n", "vm.swapusage").Output()
	if err != nil {
		return ""
	}
	// fields read "total = 0.00M  used = 0.00M  free = 0.00M (encrypted)"
	fields := strings.Fields(string(out))
	for i, f := range fields {
		if f == "used" && i+2 < len(fields) {
			return fields[i+2]
		}
	}
	return strings.TrimSpace(string(out))
}
