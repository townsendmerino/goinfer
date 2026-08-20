//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestDeltaNetKernels_cpuParity is the Metal twin of cuda.TestDeltaNetKernels_cpuParity — same
// geometry, same steps, same per-stage cosine/rel thresholds, same "chained, not isolated" shape
// (each kernel consumes the PREVIOUS kernel's real device output, gating the composition the
// resident runner will execute, not five isolated primitives — this repo's own A' zero-copy
// post-mortem: "isolation proves the primitive, never the composition").
//
// WHY COMPARE TO THE CPU AND NOT TO HF. decoder.gatedDeltaNetStep is already gated against
// transformers' torch_recurrent_gated_delta_rule, so this makes the chain kernel == CPU == HF. A
// reference written in this package would be a second unvalidated implementation of the thing
// under test.
//
// WHY MANY STEPS. The conv ring and the matrix state both compound. A recurrence can agree at
// step 1 and be visibly wrong at step 50; a single-step check catches neither a ring-slide bug
// nor a state-decay bug.
func TestDeltaNetKernels_cpuParity(t *testing.T) {
	worst, skip := deltaNetChainDrift(t, allKernels)
	if skip {
		return
	}
	for _, w := range worst {
		t.Logf("  %-11s worst cosine=%.9f worst maxAbs/rms=%.3g", w.name, w.cos, w.rel)
		if w.cos < 0.999999 || w.rel > 1e-3 {
			t.Errorf("%s drifts from the CPU reference: worst cosine=%.9f worst maxAbs/rms=%.3g",
				w.name, w.cos, w.rel)
		}
	}
}

// deltaNetChainDrift runs the chained 64-step DeltaNet mixer gate (kernels compiled from src, the
// real geometry, real CPU-reference-driven inputs) and returns each stage's worst cosine/rel
// WITHOUT asserting — the shared core both TestDeltaNetKernels_cpuParity (asserts PASS on the real
// kernels) and TestDeltaNetKernels_mutations (asserts FAIL on a deliberately broken variant) drive.
// skip reports "no metal device", which the caller should treat as t.Skip already having fired.
func deltaNetChainDrift(t *testing.T, src string) (worst []stage, skip bool) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
		return nil, true
	}
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe := func(name string) Pipeline {
		p, e := d.NewComputePipeline(lib, name)
		if e != nil {
			t.Fatalf("pipeline %s: %v", name, e)
		}
		return p
	}
	pConv, pGates := pipe("delta_conv"), pipe("delta_gates")
	pNorm, pRule, pGNorm := pipe("delta_norm"), pipe("delta_rule"), pipe("delta_gnorm")

	// Qwen3.8-27B's real DeltaNet geometry — same constants as the CUDA gate. hidden is small on
	// purpose: it sizes only the projections, which these kernels do not perform (ordinary
	// GEMVs), while the head dims size the recurrence, which they do.
	const (
		hk, hv   = 128, 128
		nk, nv   = 16, 48
		rep      = nv / nk
		convK    = 4
		hidden   = 64
		steps    = 64
		keyDim   = nk * hk
		valueDim = nv * hv
		convDim  = 2*keyDim + valueDim
		eps      = 1e-5
	)
	qScale := float32(1 / math.Sqrt(float64(hk)))

	rng := rand.New(rand.NewSource(11))
	rnd := func(n int, scale float64) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64() * scale)
		}
		return s
	}
	convW, dtBias := rnd(convDim*convK, 0.5), rnd(nv, 0.1)
	negExpA, normW := negExpRef(rnd(nv, 0.5)), rnd(hv, 0.1)
	cw, cst := decoder.NewDeltaNetForTest(convK, hk, hv, nk, nv, hidden,
		rnd(convDim*hidden, 0.05), rnd(valueDim*hidden, 0.05), rnd(nv*hidden, 0.05), rnd(nv*hidden, 0.05),
		convW, dtBias, negExpA, normW, rnd(hidden*valueDim, 0.05))

	var capMixed, capConv, capGateIn, capBG, capPre, capGated, capZ []float32
	decoder.SetDeltaCapHook(func(mixed, conv, gateIn, bg, pre, gated, z []float32) {
		capMixed, capConv, capGateIn, capBG, capPre, capGated, capZ = mixed, conv, gateIn, bg, pre, gated, z
	})
	defer decoder.SetDeltaCapHook(nil)

	// Device buffers persist across steps exactly as the resident runner would hold them — what
	// makes this a drift test rather than 64 independent single-step checks. dWin/dState are the
	// two that MUST start zeroed (matching newDeltaState); everything else is written before read.
	dMixed, dConv := d.NewBufferLen(convDim), d.NewBufferLen(convDim)
	dConvW := d.NewBufferFloats(convW)
	dWin := d.NewBufferFloats(make([]float32, (convK-1)*convDim))
	dBt, dAt := d.NewBufferLen(nv), d.NewBufferLen(nv)
	dDt, dNegA, dNormW := d.NewBufferFloats(dtBias), d.NewBufferFloats(negExpA), d.NewBufferFloats(normW)
	dHeadP := d.NewBufferLen(nv * 2)
	dQn, dKn := d.NewBufferLen(keyDim), d.NewBufferLen(keyDim)
	dState := d.NewBufferFloats(make([]float32, nv*hv*hk))
	dCore, dZ, dGated := d.NewBufferLen(valueDim), d.NewBufferLen(valueDim), d.NewBufferLen(valueDim)

	uConvDim, uK := d.NewBufferU32(uint32(convDim)), d.NewBufferU32(uint32(convK))
	uNv, uNk, uHk, uHv := d.NewBufferU32(uint32(nv)), d.NewBufferU32(uint32(nk)), d.NewBufferU32(uint32(hk)), d.NewBufferU32(uint32(hv))
	uKeyDim, uQScale := d.NewBufferU32(uint32(keyDim)), d.NewBufferFloats([]float32{qScale})
	uRep, uVBase := d.NewBufferU32(uint32(rep)), d.NewBufferU32(uint32(2*keyDim))
	uEps := d.NewBufferFloats([]float32{eps})

	q_ := d.NewCommandQueue()

	worst = []stage{{name: "delta_conv", cos: 1}, {name: "delta_gates", cos: 1},
		{name: "delta_rule", cos: 1}, {name: "delta_gnorm", cos: 1}}

	for range steps {
		decoder.DeltaNetStepForTest(nil, rnd(hidden, 1.0), cw, hidden, eps, cst)
		if capPre == nil {
			t.Fatal("capture hook never fired — the gate would be vacuous")
		}
		uploadF32(dMixed, capMixed)
		uploadF32(dBt, capGateIn[:nv])
		uploadF32(dAt, capGateIn[nv:])
		uploadF32(dZ, capZ)

		e := q_.Begin()
		e.Dispatch(pConv, convDim, 256, dMixed, dConvW, dWin, dConv, uConvDim, uK)
		e.Dispatch(pGates, nv, 64, dBt, dAt, dDt, dNegA, dHeadP, uNv)
		e.Dispatch(pNorm, nk*128, 128, dConv, dQn, dKn, uNk, uHk, uKeyDim, uQScale)
		e.Dispatch(pRule, nv*hv, 128, dQn, dKn, dConv, dHeadP, dState, dCore, uNv, uHk, uHv, uRep, uVBase)
		e.Dispatch(pGNorm, nv*128, 128, dCore, dZ, dNormW, dGated, uNv, uHv, uEps)
		e.End()

		for i, c := range []struct {
			want []float32
			got  []float32
		}{
			{capConv, dConv.Floats()},
			{capBG, dHeadP.Floats()},
			{capPre, dCore.Floats()},
			{capGated, dGated.Floats()},
		} {
			cos, maxAbs := cosF32(c.want, c.got)
			var ss float64
			for _, v := range c.want {
				ss += float64(v) * float64(v)
			}
			rel := maxAbs / math.Sqrt(ss/float64(len(c.want)))
			if cos < worst[i].cos {
				worst[i].cos = cos
			}
			if rel > worst[i].rel {
				worst[i].rel = rel
			}
		}
	}

	t.Logf("DeltaNet mixer chain vs CPU on Metal, %d steps at REAL geometry (nk=%d nv=%d hk=%d hv=%d):",
		steps, nk, nv, hk, hv)
	return worst, false
}

// stage is one chain-gate checkpoint's worst-case drift over a run.
type stage struct {
	name string
	cos  float64
	rel  float64
}

// uploadF32 overwrites dst's contents from src via a fresh source buffer + blit — the test-loop
// equivalent of CUDA's per-step CopyHtoD onto an already-allocated device buffer. Metal buffers
// here are UMA (shared), so a direct byte copy into the destination's own backing store is
// simplest and avoids re-allocating a buffer identity each step (which would break the
// dWin/dState aliasing the dispatch calls above already captured by value).
func uploadF32(dst Buffer, src []float32) {
	copy(dst.Floats(), src)
}

// negExpRef mirrors negExpAFromLog: the CPU weights store -exp(A_log), precomputed at load.
func negExpRef(aLog []float32) []float32 {
	out := make([]float32, len(aLog))
	for i, v := range aLog {
		out[i] = float32(-math.Exp(float64(v)))
	}
	return out
}

func cosF32(a, b []float32) (cos, maxAbs float64) {
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
		if d := math.Abs(x - y); d > maxAbs {
			maxAbs = d
		}
	}
	if na == 0 || nb == 0 {
		return 1, maxAbs
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb)), maxAbs
}

// TestDeltaNorm_zeroHead is the Metal twin of cuda.TestDeltaNorm_zeroHead: delta_norm on its own,
// including the degenerate head the chain test above cannot reach — a head whose conv output is
// exactly zero. silu(0) is 0, so an all-zero head is reachable, and there the two plausible
// spellings of the norm diverge completely: sqrt(1/(0+1e-6)) is a finite 1000 applied to a zero
// vector (the reference), while rsqrt(0) is +inf and poisons every downstream state entry with
// NaN (the mutation this test exists to catch — see delta_norm's comment in deltanet_kernels.go).
func TestDeltaNorm_zeroHead(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pNorm, err := d.NewComputePipeline(lib, "delta_norm")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const nk, hk = 16, 128
	const keyDim = nk * hk
	qScale := float32(1 / math.Sqrt(float64(hk)))
	rng := rand.New(rand.NewSource(7))
	conv := make([]float32, 2*keyDim)
	for i := range conv {
		conv[i] = float32(rng.NormFloat64())
	}
	for i := range hk { // head 0's q all zero (the +inf case); head 1's k denormal-small
		conv[i] = 0
		conv[keyDim+hk+i] = 1e-7
	}

	dConv := d.NewBufferFloats(conv)
	dQn, dKn := d.NewBufferLen(keyDim), d.NewBufferLen(keyDim)
	uNk, uHk, uKeyDim := d.NewBufferU32(uint32(nk)), d.NewBufferU32(uint32(hk)), d.NewBufferU32(uint32(keyDim))
	uQScale := d.NewBufferFloats([]float32{qScale})

	q_ := d.NewCommandQueue()
	e := q_.Begin()
	e.Dispatch(pNorm, nk*128, 128, dConv, dQn, dKn, uNk, uHk, uKeyDim, uQScale)
	e.End()

	gotQ, gotK := dQn.Floats(), dKn.Floats()
	for _, c := range []struct {
		name  string
		got   []float32
		off   int
		scale float32
	}{{"q", gotQ, 0, qScale}, {"k", gotK, keyDim, 1}} {
		for h := range nk {
			seg := c.got[h*hk : (h+1)*hk]
			for i, v := range seg {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatalf("%s head %d element %d is %v — the epsilon is not doing its job", c.name, h, i, v)
				}
			}
			want := decoder.L2NormScaledForTest(conv[c.off+h*hk:c.off+(h+1)*hk], c.scale)
			cos, maxAbs := cosF32(want, seg)
			if h == 0 && c.name == "q" { // the zero head: cosine is undefined, absolute is what matters
				if maxAbs > 1e-6 {
					t.Errorf("zero head: want all-zero output, maxAbs=%.3g", maxAbs)
				}
				continue
			}
			if cos < 0.999999 || maxAbs > 1e-5 {
				t.Errorf("%s head %d: cosine=%.9f maxAbs=%.3g", c.name, h, cos, maxAbs)
			}
		}
	}
	t.Logf("delta_norm vs l2normScaled over %d heads incl. a zero head and a 1e-7 head: clean", nk)
}

// TestDeltaNetKernels_mutations proves the chain gate above is actually strict, not merely
// passing — the discipline the task brief demands verbatim from the CUDA port: "Run the same
// mutations; any that pass means your gate is weaker than it looks." Each case applies ONE
// targeted, textual break to the real deltaNetKernels source (a single unique substring replaced
// exactly once — Fatal if the anchor text has drifted, so a kernel rewrite can't silently make a
// mutation a no-op) and re-runs deltaNetChainDrift, asserting the SAME threshold now fails.
//
// Four of CUDA's six documented mutations are reproducible here: this file gates only the
// five-kernel recurrence chain, not yet the softmax-layer output gate (delta_qsplit/
// delta_attn_gate) or the resident runner's Reset() — neither has Go-side wiring yet, so
// "not applying the output gate" and "removing the state reset" have no wiring to mutate against
// until that lands. Re-open this list once they do.
func TestDeltaNetKernels_mutations(t *testing.T) {
	cases := []struct {
		name string
		from string
		to   string
	}{
		{
			// headV % rep (rep=3, headV<48) stays in [0,3) — wrong (should span [0,nk)=16) but
			// in-bounds, unlike headV*rep which reads kn/qn up to ~141*hk past their nk*hk
			// allocation and faults the process outright rather than producing wrong-but-readable
			// values. A mutation that crashes instead of drifting is not a useful test case.
			name: "invert GVA head map (headV/rep -> headV%rep)",
			from: "uint headK = headV / rep; // GVA: rep value heads share one key head",
			to:   "uint headK = headV % rep; // MUTATED: GVA head map inverted",
		},
		{
			name: "drop the v-slice offset (vBase dropped from the v read)",
			from: "float delta = (v[vBase+headV*hv+vd] - kvdot) * beta;",
			to:   "float delta = (v[headV*hv+vd] - kvdot) * beta; // MUTATED: vBase dropped",
		},
		{
			// A first attempt swapped hv/hk in the offset formula, but at this test's REAL geometry
			// hk==hv==128 makes that a no-op (numerically identical). A second attempt permuted
			// (headV,vd)->row bijectively — also invisible: a thread's row is never read by any
			// OTHER thread, so consistently relocating one thread's own private storage changes
			// nothing about what it computes, only where. The actual bug class "un-transposing"
			// guards against is state ADDRESSES COLLIDING across threads, which a bijection can't
			// produce by construction. This drops the vd term instead, so every value head sharing
			// one key head... no, every vd for a fixed headV now aliases ONE row: max offset
			// (nv-1)*hk = 47*128 = 6016, safely in-bounds, and the per-thread exclusivity the file
			// header calls out ("no cross-thread sharing") is exactly what breaks.
			name: "collapse the state row (vd dropped from the address, threads alias)",
			from: "device float* S = state + (uint)(headV*hv+vd)*hk;",
			to:   "device float* S = state + (uint)(headV)*hk; // MUTATED: vd dropped, threads alias",
		},
		{
			name: "drop the conv ring slide (win never advances)",
			from: "    for (uint j=0; j+1<K-1; j++) win[j*convDim+c] = win[(j+1)*convDim+c];\n    win[(K-2)*convDim+c] = xc;",
			to:   "    // MUTATED: ring slide dropped, win never advances",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			n := strings.Count(allKernels, c.from)
			if n != 1 {
				t.Fatalf("mutation anchor matched %d times (want exactly 1) — the kernel source has "+
					"drifted from this test; update the anchor text: %q", n, c.from)
			}
			mutated := strings.Replace(allKernels, c.from, c.to, 1)
			if mutated == allKernels {
				t.Fatal("mutation produced no change — anchor did not match allKernels")
			}
			worst, skip := deltaNetChainDrift(t, mutated)
			if skip {
				return
			}
			failed := false
			for _, w := range worst {
				t.Logf("  %-11s worst cosine=%.9f worst maxAbs/rms=%.3g", w.name, w.cos, w.rel)
				if w.cos < 0.999999 || w.rel > 1e-3 {
					failed = true
				}
			}
			if !failed {
				t.Errorf("mutation %q passed the chain gate — the gate is weaker than it looks", c.name)
			}
		})
	}
}
