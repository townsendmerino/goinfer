//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestGptOssResidentParityReal20B runs a REAL gpt-oss-20b forward on the Metal resident path.
//
// WHY IT EXISTS. Metal declares FeatAttnSink/FeatOutBias/FeatRopeMscale, but the evidence under
// that declaration is TestGptOssResidentParity on a hand-built 2-layer tiny fixture — sub-T3 by
// this repo's own tiering. docs/queue-correctness.md G7 records that "nothing has run a whole
// gpt-oss forward on the resident path" on EITHER backend, and 2224441 is the precedent for why
// that matters: a declaration made on kernel-level parity was correctly reverted. This is that
// missing run, on the backend where it is reachable.
//
// WHY IT IS NOT residentParity(). That helper loads the checkpoint TWICE — once Metal, once CPU —
// which is free at tiny scale and impossible here: 12.1 GB twice on a 16 GB machine. This runs
// the CPU arm FIRST, keeps only its logits and its token sequence, releases the model, and THEN
// loads Metal. Same comparison, half the peak footprint. The token sequence is CPU-determined and
// replayed, so both arms score the same positions.
//
// A DECLINE IS A RESULT, NOT AN ERROR. Metal wires the mmap pages a command buffer touches, so
// whole-model residency at 12.1 GB on 16 GB RAM is the known-hard case (the 26B experiment died
// exactly here). If the resident path declines, this test says so with the reason and fails —
// because admission currently claims gpt_oss IS admitted on Metal, so a decline means the
// declaration and the runtime disagree, which is the thing worth knowing either way.
func TestGptOssResidentParityReal20B(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 (loads a 12 GB model from ~/models)")
	}
	path := os.Getenv("GOINFER_GPTOSS_GGUF")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "gpt-oss-20b-MXFP4.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no gpt-oss checkpoint at %s: %v", path, err)
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	// FITS-IN-RAM GUARD — this test WILL hang a machine without it, and nothing in the engine
	// stops it. Measured 2026-08-31 on a 16 GB MacBook: loading this 11.28 GB checkpoint on the
	// Metal resident path drove swap to 35.98 GB of 36 GB (885 MB free), left the process in
	// uninterruptible I/O wait at 29% CPU with RSS creeping 1.8 -> 2.0 GB over 12 minutes, and
	// never completed or declined. The resident path has a KV CONTEXT cap (metal/backend.go:98)
	// but NO weight-size feasibility check, so it accepts a model larger than RAM and thrashes.
	// Keyed on bytes computed here, not on the OS's account of what is free: Darwin's UBC reclaims
	// under pressure, so "available" reports what survived rather than what can be asked for.
	if st, err := os.Stat(path); err == nil {
		var ram uint64
		if out, e := exec.Command("sysctl", "-n", "hw.memsize").Output(); e == nil {
			ram, _ = strconv.ParseUint(strings.TrimSpace(string(out)), 10, 64)
		}
		// Both arms hold weights at once only briefly, but Metal WIRES the mmap pages a command
		// buffer touches, so the resident arm alone needs the full model plus KV and scratch.
		if need := uint64(st.Size()) * 3 / 2; ram > 0 && need > ram {
			t.Skipf("checkpoint %.2f GB needs ~%.1f GB with Metal wiring against %.1f GB RAM — "+
				"this thrashes swap to exhaustion rather than declining (see docs/queue-correctness.md G7)",
				float64(st.Size())/(1<<30), float64(need)/(1<<30), float64(ram)/(1<<30))
		}
	}
	if !decoder.ResidentBackendFeatures("metal")[decoder.FeatAttnSink] {
		t.Skip("metal does not declare FeatAttnSink")
	}

	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	seed, err := tk.Encode("The capital of France is", true)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	const steps = 12

	// ---- Phase 1: CPU arm. Record logits AND the greedy token sequence, then release. ----
	cpuLogits := make([][]float32, 0, steps)
	toks := make([]int, 0, steps)
	t0 := time.Now()
	func() {
		mcpu, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
		if err != nil {
			t.Fatalf("load (cpu): %v", err)
		}
		defer mcpu.Close()
		_, nL, _, nKV, hd, _, _ := mcpu.Dims()
		cache := decoder.NewKVCache(nL, nKV, hd, 0, 1024)
		tok := seed[0]
		for i := range steps {
			l, err := mcpu.ForwardForTest(tok, cache)
			if err != nil {
				t.Fatalf("cpu forward %d: %v", i, err)
			}
			cpuLogits = append(cpuLogits, append([]float32(nil), l...))
			toks = append(toks, tok)
			if i+1 < len(seed) {
				tok = seed[i+1]
			} else {
				tok = argmaxF(l)
			}
			fmt.Fprintf(os.Stderr, "[gptoss-20b] cpu %d/%d elapsed=%s\n", i+1, steps,
				time.Since(t0).Round(time.Second))
		}
	}()

	// ---- Phase 2: Metal arm, replaying the SAME tokens. ----
	mg, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load (metal): %v", err)
	}
	defer mg.Close()
	rf := mg.ResidentForwardForTest()
	if rf == nil { // a silent CPU fallback would pass every assertion below trivially
		t.Fatalf("metal resident DECLINED at 12 GB (%s) — admission says gpt_oss IS admitted on "+
			"metal, so the declaration and the runtime disagree", mg.ResidentDecline())
	}
	st := parityStats{steps: steps, minCos: 1}
	for i, tok := range toks {
		gpuL, err := rf.Forward(mg.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("gpu forward %d: %v", i, err)
		}
		st.observeCos(cosF(cpuLogits[i], gpuL))
		if argmaxF(cpuLogits[i]) == argmaxF(gpuL) {
			st.exact++
		}
		fmt.Fprintf(os.Stderr, "[gptoss-20b] metal %d/%d cos=%.6f elapsed=%s\n", i+1, steps,
			cosF(cpuLogits[i], gpuL), time.Since(t0).Round(time.Second))
	}
	t.Logf("gpt-oss-20b REAL resident parity: %d/%d argmax-exact, min cosine %.6f (%d NaN)",
		st.exact, st.steps, st.minCos, st.nan)
	// Same int4-noise bar the other resident gates use.
	assertParity(t, "gpt-oss-20b-real", st, 0.95)
}

// TestGptOssResidentMemGuardDeclines is the END-TO-END half of the fits-in-memory guard: the
// unit test above pins the arithmetic, this pins that the arithmetic is actually REACHED and
// that the outcome is a clean decline rather than the swap-exhaustion hang it replaced.
//
// It is safe to run on the machine that hung: the guard fires before Metal allocates anything,
// and the CPU-side weight load it does reach is the same one that completed fine (12/12 steps)
// during the measurement. If this test ever hangs, the guard has regressed — which is precisely
// what it is here to catch.
func TestGptOssResidentMemGuardDeclines(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	path := os.Getenv("GOINFER_GPTOSS_GGUF")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "gpt-oss-20b-MXFP4.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no gpt-oss checkpoint at %s", path)
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	var ram uint64
	if out, e := exec.Command("sysctl", "-n", "hw.memsize").Output(); e == nil {
		ram, _ = strconv.ParseUint(strings.TrimSpace(string(out)), 10, 64)
	}
	if st, err := os.Stat(path); err != nil || ram == 0 || fitsResidentBudget(st.Size(), ram) {
		t.Skip("this machine fits the checkpoint — the decline path is not the case under test here")
	}

	t0 := time.Now()
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	t.Logf("weights %.2f GB against %.1f GB RAM; load returned in %s",
		float64(m.ResidentWeightBytes())/(1<<30), float64(ram)/(1<<30),
		time.Since(t0).Round(time.Second))
	if m.ResidentActive() {
		t.Fatal("resident is ACTIVE for a model larger than the memory budget — the guard did not fire")
	}
	if m.ResidentDecline() == "" {
		t.Error("declined without recording a reason — the decline must be attributable")
	}
	t.Logf("declined cleanly: %s", m.ResidentDecline())
}
