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

// Mellum2 long-prefill profiler — the ATTENTION-vs-MoE-FFN split of a batched
// MoE prefill, at a chosen prompt length.
//
// WHY THIS EXISTS, and why the dense profile does not answer it. The K=8192
// profile quoted in queue-performance.md (`MatmulAVAcc64` 51.1%, `MatmulQKAcc64`
// 18.7%, ~70% attention) is a DENSE 1.5B. Two levers are live for long-context
// prefill and they attack different terms:
//
//	f32 attention  (G24, shipped as --cpu-fast-attention)  -> the O(K²) attention term
//	expert-major batching (task-moe-streaming.md Lever 4)  -> the O(K) per-row MoE FFN
//
// Their shares move in OPPOSITE directions as K grows, so "which is binding" has
// no single answer — it has a crossover, and the crossover is what this measures.
// Two structural facts make the dense number a bad proxy for the MoE case:
//
//  1. `--cpu-fast-attention` REFUSES MoE (G24), so on this model the attention
//     lever is not merely unmeasured, it is unavailable as shipped.
//  2. Mellum2 is 24/28 SLIDING-window layers at window 1024 (config.json
//     `layer_types`). Only the 4 full_attention layers grow as O(K²); the other 24
//     cap their nKeys at 1024. A dense model's attention share therefore
//     OVERSTATES this one's at every K past the window.
//
// Mellum2 is the only locally-available MoE that takes the batched path at all —
// canBatchN excludes deepseek_v2 (MLA), qwen3_5_moe and gemma4, which is every
// other MoE checkpoint in ~/models. It cannot go through loadBenchModel: that
// resolves GOINFER_PREQUANT_GGUF, whose asset kind is "file", and this checkpoint
// is a safetensors DIRECTORY. So it loads the directory directly, the way
// mellum2_parity_test.go does.
//
// Run (one quant per process):
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_MELLUM_K=2048 \
//	GOINFER_MELLUM_PROF=/tmp/mellum-2048.prof \
//	go test ./decoder/ -run TestMellum2PrefillProfile -timeout 3600s -v
//
// Then: go tool pprof -top -nodecount=30 <prof>
//
// SWAP IS RECORDED, NOT ASSUMED AWAY. This is a ~12 GB int8int8 model on a 16 GB
// box, and a run that pages produces a plausible, wrong profile — the exact
// failure class that made the withdrawn G15 cliff expensive. `vm.swapusage` is
// read before and after and printed with the result; a non-zero delta invalidates
// the run. It is deliberately NOT an RSS check: darwin's UBC reclaims under
// pressure, so RSS reports what survived rather than what was asked for (it read
// LOWER at the failure point than at the baseline in the Metal slot sweep).
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

	// Default to the FULL checkpoint; GOINFER_MELLUM_CKPT points at a layer slice
	// where the full one will not fit. A 4-layer slice is representative here and
	// that is a property of THIS model, not a general licence: Mellum2 interleaves
	// layer_types on a period of 4 (s,s,s,f), so 21 sliding / 7 full — exactly 25%
	// full_attention — is reproduced exactly by layers [0,4), and every layer is
	// `sparse`, so the MoE geometry is unchanged. What a slice does NOT preserve is
	// the embedding's share of the total (4 layers amortize it over 7x less work);
	// the LM head is excluded by construction, since forwardLayersN stops at the
	// layer stack.
	path := os.Getenv("GOINFER_MELLUM_CKPT")
	if path == "" {
		path = os.Getenv("HOME") + "/models/mellum2-unq"
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no Mellum2 checkpoint (%v)", err)
	}
	quant := os.Getenv("GOINFER_BENCH_QUANT")
	if quant == "" {
		quant = "int8int8"
	}

	// Load is minutes of I/O on a 23 GB bf16 checkpoint; say so as it happens
	// rather than after, so a stalled load is distinguishable from a slow one.
	// os.Stderr, not t.Logf: t.Logf is buffered until the function returns.
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
	// The whole point is the BATCHED path. A silent fall-through to the
	// sequential one would profile a different program and still print a number.
	if !m.canBatchN(K) {
		t.Fatalf("canBatchN(%d) = false: this profile is only meaningful on the batched path", K)
	}

	// TOKEN CONTENT IS NOT NEUTRAL HERE, and the dense harnesses' constant-id
	// prompt is a trap on a MoE. G15's profiler fills the prompt with one repeated
	// id because on a dense model content genuinely cannot change prefill cost.
	// On a MoE it decides ROUTING: near-identical rows select near-identical
	// experts, so the top-8 expert weights stay in cache and the FFN measures a
	// best case the real workload never sees.
	//
	// So the two id patterns are the experiment's two ARMS, not a detail:
	//
	//	varied  — deterministic spread over the vocab; real routing diversity.
	//	uniform — one repeated id; degenerate routing. This is the CONTROL, and
	//	          it is also the CEILING for expert-major batching: a chunk whose
	//	          rows all select the same experts is exactly what that lever is
	//	          trying to manufacture. It cannot beat actually having it.
	//
	// The arms are matched on everything else — same K, same shapes, same
	// attention work (attention cost is content-independent) — so the paired
	// difference isolates the routing term instead of pooling it with the rest.
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
