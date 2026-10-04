//go:build darwin && goinfer_testhooks

package metal

import (
	"errors"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// Gates F2 and F3 of docs/tasks/task-metal-int8-2026-10.md, pre-registered 2026-10-01: does a dense int8 model on
// Metal's native int8 path (r.w8) agree with the CPU at the same quant, and is it as close to f32 as the CPU's int8?
// They run on the 0.5B coder by day; GOINFER_W8_GATE_MODEL names another checkpoint (the 1.5B runs with gate S).

func w8GateModel(t *testing.T) string {
	t.Helper()
	requireHeavyModel(t)
	useNativeInt8(t, true)
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if p := os.Getenv("GOINFER_W8_GATE_MODEL"); p != "" {
		path = p
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	return path
}

// TestW8Native_F2_matchesCPUAtSameQuant is gate F2: in residentParity's harness (24 greedy steps in lockstep, logit
// cosine and argmax per step), Metal int8int8 against CPU int8int8 must agree at least as well as Metal int4 against
// CPU int4 in the same run: minimum cosine no lower than the int4 pair's and at least 0.99, and argmax agreement at no
// fewer steps.
func TestW8Native_F2_matchesCPUAtSameQuant(t *testing.T) {
	path := w8GateModel(t)
	seed := seedPrompt(t, path, probeText)
	assertNativeInt8(t, path, "int8int8", true)
	assertNativeInt8(t, path, "int4", false)
	p8 := residentParityAt(t, path, "int8int8", seed, 24)
	p4 := residentParityAt(t, path, "int4", seed, 24)
	if p8.nan > 0 || p4.nan > 0 {
		t.Fatalf("non-finite cosines: int8 pair %d, int4 pair %d", p8.nan, p4.nan)
	}
	t.Logf("F2: int8 pair min cosine %.6f, argmax %d/%d; int4 pair min cosine %.6f, argmax %d/%d",
		p8.minCos, p8.exact, p8.steps, p4.minCos, p4.exact, p4.steps)
	if p8.minCos < p4.minCos || p8.minCos < 0.99 {
		t.Errorf("F2 fails on cosine: int8 pair %.6f against the int4 pair's %.6f and the 0.99 floor", p8.minCos, p4.minCos)
	}
	if p8.exact < p4.exact {
		t.Errorf("F2 fails on argmax: the int8 pair agrees at %d steps, the int4 pair at %d", p8.exact, p4.exact)
	}
}

// TestW8Native_F3_closerToF32 is gate F3: on one token sequence (the prompt, then the CPU f32 model's greedy
// continuation, 24 positions), the mean per-position KL(f32 ‖ Metal int8int8) is at most 1.10 × KL(f32 ‖ CPU
// int8int8). KL(f32 ‖ Metal int4) is reported beside it. Positions 0 and 1 are left out, as residentParity's cosine
// leaves them out. The models load one at a time at a 1024-position context, so a 16 GB Mac holds the f32 reference
// only while it runs; by day the fit guard still refused it there (needs ~2.8 GB against a 2.2 GB budget), so F3
// runs on the night queue.
func TestW8Native_F3_closerToF32(t *testing.T) {
	path := w8GateModel(t)
	seed := seedPrompt(t, path, probeText)
	const steps = 24
	ref, toks := cpuLogitsAt(t, path, "", seed, steps, nil)
	cpu8, _ := cpuLogitsAt(t, path, "int8int8", seed, steps, toks)
	cpu8h, _ := cpuLogitsAtKV(t, path, "int8int8", seed, steps, toks, true)
	met8 := metalLogitsAt(t, path, "int8int8", toks, true)
	met4 := metalLogitsAt(t, path, "int4", toks, false)
	meanKL := func(arm [][]float32) float64 {
		var sum float64
		for i := 2; i < steps; i++ {
			sum += klLogits(ref[i], arm[i])
		}
		return sum / float64(steps-2)
	}
	kCPU8, kMet8, kMet4 := meanKL(cpu8), meanKL(met8), meanKL(met4)
	// Reported, not gated (the bar is the owner's to change): the CPU at int8int8 with Metal's f16 KV precision, the
	// like-for-like reference TestW8Native_perLayerBisect points to.
	kCPU8h := meanKL(cpu8h)
	worse := 0
	for i := 2; i < steps; i++ {
		if klLogits(ref[i], met8[i]) > klLogits(ref[i], cpu8h[i]) {
			worse++
		}
	}
	t.Logf("F3 reported: KL(f32 ‖ CPU int8int8, f16 KV) %.6f; Metal int8int8 is %.3f× it, and further from f32 at %d of %d positions",
		kCPU8h, kMet8/kCPU8h, worse, steps-2)
	if os.Getenv("GOINFER_W8_F3_FAST") == "1" { // the same Metal int8int8 arm with fast math kept (w8FastMath)
		prev := w8FastMath
		w8FastMath = true
		met8f := metalLogitsAt(t, path, "int8int8", toks, true)
		w8FastMath = prev
		t.Logf("F3 reported: Metal int8int8, fast math: KL(f32 ‖ ·) %.6f, %.3f× the f16-KV CPU's", meanKL(met8f), meanKL(met8f)/kCPU8h)
	}
	t.Logf("F3: mean KL(f32 ‖ ·) over %d positions: CPU int8int8 %.6f, Metal int8int8 %.6f (%.3f× the CPU's), Metal int4 %.6f",
		steps-2, kCPU8, kMet8, kMet8/kCPU8, kMet4)
	for _, k := range []float64{kCPU8, kMet8, kMet4} {
		if math.IsNaN(k) || math.IsInf(k, 0) {
			t.Fatalf("non-finite KL: CPU int8 %g, Metal int8 %g, Metal int4 %g", kCPU8, kMet8, kMet4)
		}
	}
	if kMet8 > 1.10*kCPU8 {
		t.Errorf("F3 fails: KL(f32 ‖ Metal int8int8) %.6f is above 1.10 × the CPU int8int8's %.6f", kMet8, kCPU8)
	}
}

// useNativeInt8 sets nativeInt8 for the rest of the test and restores it after.
func useNativeInt8(t *testing.T, on bool) {
	prev := nativeInt8
	nativeInt8 = on
	t.Cleanup(func() { nativeInt8 = prev })
}

// assertNativeInt8 loads path on Metal at quant and checks the resident took the native int8 path exactly when want.
func assertNativeInt8(t *testing.T, path, quant string, want bool) {
	t.Helper()
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: quant})
	if err != nil {
		t.Fatalf("load (metal, %s): %v", quant, err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	if rf == nil {
		skipIfMemoryDeclined(t, m)
		t.Fatalf("metal resident declined at %s: %s", quant, m.ResidentDecline())
	}
	a, ok := rf.(*metalResident)
	if !ok || a.r.w8 != want {
		t.Fatalf("%s: native int8 path = %v, want %v (%s)", quant, ok && a.r.w8, want, m.DecodePath())
	}
}

// cpuLogitsAt runs path on the CPU at quant over steps positions and returns each position's logits and the tokens
// read. With forced nil it reads seed and then its own greedy continuation; otherwise it reads forced.
func cpuLogitsAt(t *testing.T, path, quant string, seed []int, steps int, forced []int) ([][]float32, []int) {
	t.Helper()
	return cpuLogitsAtKV(t, path, quant, seed, steps, forced, false)
}

// cpuLogitsAtKV is cpuLogitsAt; with f16KV every K and V the CPU stores is rounded to f16 (Metal's KV precision) right
// after the position that wrote it.
func cpuLogitsAtKV(t *testing.T, path, quant string, seed []int, steps int, forced []int, f16KV bool) ([][]float32, []int) {
	t.Helper()
	m, err := decoder.Load(path, decoder.Options{Quant: quant, ResidentContext: 1024})
	if errors.Is(err, decoder.ErrWontFitResident) { // the machine's memory right now, not the gate's answer
		t.Skipf("the fit guard refused the CPU %q load on this machine now: %v", quant, err)
	}
	if err != nil {
		t.Fatalf("load (cpu, %q): %v", quant, err)
	}
	defer m.Close()
	_, nL, _, nKV, hd, _, _ := m.Dims()
	cache := decoder.NewKVCache(nL, nKV, hd, 0, 1024, nil)
	out, toks := make([][]float32, steps), make([]int, steps)
	tok := seed[0]
	for i := range steps {
		if forced != nil {
			tok = forced[i]
		}
		toks[i] = tok
		l, err := m.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward (%q) at %d: %v", quant, i, err)
		}
		out[i] = append([]float32(nil), l...)
		if f16KV {
			for layer := range nL {
				k, v, _ := cache.LayerKVForTest(layer)
				for _, sl := range [][]float32{k[i*nKV*hd : (i+1)*nKV*hd], v[i*nKV*hd : (i+1)*nKV*hd]} {
					for j, x := range sl {
						sl[j] = f16ToF32(f32ToF16(x))
					}
				}
			}
		}
		if forced == nil {
			if i+1 < len(seed) {
				tok = seed[i+1]
			} else {
				tok = argmaxF(l)
			}
		}
	}
	return out, toks
}

// metalLogitsAt runs path on the Metal resident at quant over toks and returns each position's logits, after checking
// the resident took the native int8 path exactly when wantNative.
func metalLogitsAt(t *testing.T, path, quant string, toks []int, wantNative bool) [][]float32 {
	t.Helper()
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: quant, ResidentContext: 1024})
	if err != nil {
		t.Fatalf("load (metal, %s): %v", quant, err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	if rf == nil {
		skipIfMemoryDeclined(t, m)
		t.Fatalf("metal resident declined at %s: %s", quant, m.ResidentDecline())
	}
	if a, ok := rf.(*metalResident); !ok || a.r.w8 != wantNative {
		t.Fatalf("%s: native int8 path = %v, want %v (%s)", quant, ok && a.r.w8, wantNative, m.DecodePath())
	}
	out := make([][]float32, len(toks))
	for i, tok := range toks {
		l, err := rf.Forward(m.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("metal forward (%s) at %d: %v", quant, i, err)
		}
		out[i] = append([]float32(nil), l...)
	}
	return out
}

// klLogits is KL(softmax(p) ‖ softmax(q)) in float64.
func klLogits(p, q []float32) float64 {
	lse := func(v []float32) float64 {
		mx := math.Inf(-1)
		for _, x := range v {
			mx = math.Max(mx, float64(x))
		}
		var s float64
		for _, x := range v {
			s += math.Exp(float64(x) - mx)
		}
		return mx + math.Log(s)
	}
	lp, lq := lse(p), lse(q)
	var kl float64
	for i := range p {
		a := float64(p[i]) - lp
		kl += math.Exp(a) * (a - (float64(q[i]) - lq))
	}
	return kl
}
