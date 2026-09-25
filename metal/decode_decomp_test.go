//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMetalDecodeDecomp is S0 of the Metal decode-at-depth scoping (2026-09-25): where does one decode token's GPU
// time go, by kernel category, at KV depths 128 / 2048 / 3900, on a real checkpoint — and how much of it is the
// per-token dispatch floor rather than work?
//
// METHOD — no replica. It times the PRODUCTION Forward (its command buffer's GPU timestamps, recorded by
// forwardLogits) while one category's pipelines are swapped for a no-op kernel. Every dispatch is still issued with
// the same grid, so the launch/dispatch cost stays and only that category's work disappears:
//
//	in-sequence work cost of a category = full − full-with-that-category-no-op'd
//	dispatch floor                     = the token with EVERY pipeline no-op'd
//
// Kernel timing does not depend on the values read (no data-dependent branches), so the stale buffers a no-op
// leaves behind change no other kernel's work. The KV cache is filled to each depth by PrefillLast, and decode then
// repeats at that one position (each call rewrites the same slot), so the depth is constant across reps. Arms are
// interleaved rep by rep, back to back, so every arm sees the same sustained GPU state.
//
// Plain dense families on the W4A8 decode path only (the one qwen2.5 runs). Env: GOINFER_METAL_DDECOMP=1 (required),
// GOINFER_METAL_DDECOMP_MODEL (default the 1.5B q4_k_m), GOINFER_METAL_DDECOMP_DEPTHS (default 128,2048,3900),
// GOINFER_METAL_DDECOMP_REPS (default 5), GOINFER_METAL_DDECOMP_TOKENS (decode steps per arm per rep, default 20).
func TestMetalDecodeDecomp(t *testing.T) {
	if os.Getenv("GOINFER_METAL_DDECOMP") != "1" {
		t.Skip("set GOINFER_METAL_DDECOMP=1 (loads a real checkpoint; minutes of GPU time)")
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GOINFER_METAL_DDECOMP_MODEL")
	if path == "" {
		path = filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md): a timing from it measures the disk", path)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}
	intList := func(env string, def []int) []int {
		v := os.Getenv(env)
		if v == "" {
			return def
		}
		var out []int
		for _, f := range strings.Split(v, ",") {
			n, err := strconv.Atoi(strings.TrimSpace(f))
			if err != nil {
				t.Fatalf("%s: %v", env, err)
			}
			out = append(out, n)
		}
		return out
	}
	depths := intList("GOINFER_METAL_DDECOMP_DEPTHS", []int{128, 2048, 3900})
	reps := intList("GOINFER_METAL_DDECOMP_REPS", []int{5})[0]
	steps := intList("GOINFER_METAL_DDECOMP_TOKENS", []int{20})[0]
	t0 := time.Now()
	hb := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[ddecomp %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}

	maxD := 0
	for _, d := range depths {
		maxD = max(maxD, d)
	}
	// Pin the resident context to 4096 (metalCtxCapDefault): enough for every depth here (3900+1), and what the fit
	// guard prices the KV at. NOT metalCtxCapMax — that is 32768 since 26f64807, and pricing a 32k KV (~1.8 GB for
	// the 1.5B) gets an otherwise-fitting load refused under ordinary memory pressure.
	m, err := decoder.Load(path, decoder.Options{Quant: "int4", ResidentContext: metalCtxCapDefault})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("build resident: %v", err)
	}
	defer r.Close()
	if r.moe != nil || r.g4moe != nil || r.sandwich || r.postOnly || r.parallelBlock || r.kvI8 || r.layerNorm || r.decodeLaneW4F16 {
		t.Skip("decode decomposition covers the plain dense W4A8 decode path only")
	}
	if maxD+1 > r.ctxCap {
		t.Fatalf("depth %d does not fit the resident context %d", maxD, r.ctxCap)
	}
	hb("loaded %s: H=%d I=%d nL=%d nH=%d V=%d, ctx %d, attention_fa on: %v (floor %d), aliased: %v",
		filepath.Base(path), r.H, r.I, r.nL, r.nH, r.V, r.ctxCap, r.decodeAttnFA, attnFADepthFloor, r.alias != nil)

	// The no-op kernel. Buffers bound at indices it does not declare, and any threadgroup-memory length set by
	// DispatchTG, are ignored — it launches the same grid and returns.
	var noop Pipeline
	func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		pool := NewARPool()
		defer pool.Drain()
		lib, err := r.d.CompileLibrary("#include <metal_stdlib>\nusing namespace metal;\nkernel void s0_noop() {}\n", MSL3_1)
		if err != nil {
			t.Fatalf("compile no-op: %v", err)
		}
		if noop, err = r.d.NewComputePipeline(lib, "s0_noop"); err != nil {
			t.Fatalf("no-op pipeline: %v", err)
		}
	}()

	type cat struct {
		name  string
		pipes []*Pipeline
	}
	cats := []cat{
		{"attention", []*Pipeline{&r.pAttn, &r.pAttnFA, &r.pAttnFACombine}},
		{"GEMV qkv (+bias)", []*Pipeline{&r.pSABias}},
		{"GEMV o (+residual)", []*Pipeline{&r.pSAResid}},
		{"GEMV gate/up", []*Pipeline{&r.pSA}},
		{"GEMV down (+residual)", []*Pipeline{&r.pGemvResid}},
		{"LM head (int8)", []*Pipeline{&r.pGemvW8}},
		{"norm+quant (rms)", []*Pipeline{&r.pRms}},
		{"rope, kv store, act-quant, ctx-quant", []*Pipeline{&r.pRope2, &r.pKv, &r.pSw, &r.pQv}},
	}
	var all []*Pipeline
	for _, c := range cats {
		all = append(all, c.pipes...)
	}
	swap := func(ps []*Pipeline) (restore func()) {
		saved := make([]Pipeline, len(ps))
		for i, p := range ps {
			saved[i] = *p
			*p = noop
		}
		return func() {
			for i, p := range ps {
				*p = saved[i]
			}
		}
	}

	vocab := r.V
	for _, D := range depths {
		embs := make([][]float32, D)
		for i := range embs {
			embs[i] = m.EmbedResidentForTest((i*131 + 7) % vocab)
		}
		r.PrefillLast(embs, 0)
		if err := r.takeExecErr(); err != nil {
			t.Fatalf("prefill to %d: %v", D, err)
		}
		tok := (D*131 + 7) % vocab
		// decode at position D, attending over D+1 keys, repeatedly (the same slot is rewritten)
		step := func() float64 {
			r.Forward(tok, D)
			if err := r.takeExecErr(); err != nil {
				t.Fatalf("decode at %d: %v", D, err)
			}
			return (r.gpuEnd - r.gpuStart) * 1e3
		}
		for i := 0; i < 5; i++ { // warm
			step()
		}
		ref := append([]float32(nil), r.logitsHost...)
		arm := func(ps []*Pipeline) float64 { // median GPU ms per token over `steps` decode steps
			var restore func()
			if ps != nil {
				restore = swap(ps)
			}
			xs := make([]float64, steps)
			for i := range xs {
				xs[i] = step()
			}
			if restore != nil {
				restore()
			}
			return median(xs)
		}
		full := []float64{}
		floor := []float64{}
		per := make([][]float64, len(cats))
		for rep := 0; rep < reps; rep++ {
			full = append(full, arm(nil))
			for ci, c := range cats {
				per[ci] = append(per[ci], arm(c.pipes))
			}
			floor = append(floor, arm(all))
			hb("depth %d rep %d/%d: full %.3f ms/token, dispatch floor %.3f", D, rep+1, reps, full[rep], floor[rep])
		}
		// Sanity: after restoring every pipeline, a decode step reproduces the reference logits exactly.
		step()
		for i := range ref {
			if math.Float32bits(ref[i]) != math.Float32bits(r.logitsHost[i]) {
				t.Fatalf("depth %d: logits after the no-op arms differ from before at %d — a pipeline was not restored", D, i)
			}
		}

		fm := median(full)
		fmt.Fprintf(os.Stderr, "\n=== Metal decode decomposition, depth %d (%d keys), medians of %d reps × %d tokens ===\n", D, D+1, reps, steps)
		var sum float64
		for ci, c := range cats {
			var d []float64
			for i := range full {
				d = append(d, full[i]-per[ci][i])
			}
			w := median(d)
			sum += w
			fmt.Fprintf(os.Stderr, "  %-38s work %7.3f ms  %5.1f%% of token   paired %s\n", c.name, w, 100*w/fm, fmtMs(d))
		}
		fl := median(floor)
		fmt.Fprintf(os.Stderr, "  %-38s      %7.3f ms  %5.1f%% of token   spread %.1f%%\n", "dispatch floor (every pipeline no-op)", fl, 100*fl/fm, 100*spreadOf(floor))
		fmt.Fprintf(os.Stderr, "  full token %.3f ms (%.1f tok/s), spread %.1f%%; categories' work + floor = %.3f ms (%.1f%% of full)\n\n",
			fm, 1000/fm, 100*spreadOf(full), sum+fl, 100*(sum+fl)/fm)
	}
}
