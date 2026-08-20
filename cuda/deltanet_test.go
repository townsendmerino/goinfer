//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"math"
	"math/rand"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// The whole Gated-DeltaNet mixer chain on CUDA vs the CPU reference, at REAL head geometry,
// driven for enough tokens that a drifting state shows.
//
// Five kernels run CHAINED — each consumes the previous one's real device output, not the CPU's —
// so this gates the composition the resident runner will execute rather than five isolated
// primitives. That is the A′ zero-copy post-mortem's lesson recorded verbatim in this package:
// "isolation proves the primitive, never the composition." Each stage is scored separately so a
// failure names the culprit instead of the chain.
//
// WHY THIS STARTS AT delta_conv AND THE WEBGPU TEST DID NOT. WebGPU's causal conv IS the Mamba-2
// conv, already gated there. CUDA has no SSM engine — no conv-ring, no persistent state, nothing
// recurrent in any of its 24 kernels — so delta_conv is new code and has to be gated from its own
// input. That is why the capture hook grew a `mixed` slot.
//
// WHY COMPARE TO THE CPU AND NOT TO HF. The CPU recurrence is already gated against transformers'
// torch_recurrent_gated_delta_rule, so this makes the chain kernel ≡ CPU ≡ HF. A reference written
// here would be a second unvalidated implementation of the thing under test.
//
// WHY MANY STEPS. The conv ring and the matrix state both COMPOUND. A recurrence can agree at step
// 1 and be visibly wrong at step 50; a single-step check catches neither a ring-slide bug nor a
// state-decay bug.
func TestDeltaNetKernels_cpuParity(t *testing.T) {
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit failed (no driver): %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	cctx, err := dev.Primary()
	if err != nil {
		t.Skipf("no primary context: %v", err)
	}
	defer cctx.Close()
	bg := context.Background()
	mod, err := cctx.LoadModule(deltaNetPTX)
	if err != nil {
		t.Fatalf("JIT deltanet.ptx: %v", err)
	}
	fn := func(name string) *gc.Function {
		f, e := mod.Function(name)
		if e != nil {
			t.Fatalf("Function(%s): %v", name, e)
		}
		return f
	}
	kConv, kGates := fn("delta_conv"), fn("delta_gates")
	kNorm, kRule, kGNorm := fn("delta_norm"), fn("delta_rule"), fn("delta_gnorm")

	// Qwen3.8-27B's real DeltaNet geometry. `hidden` is small on purpose: it sizes only the
	// projections, which these kernels do not perform (they are ordinary GEMVs), while the head
	// dims size the recurrence, which they do.
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
	// CPU side: a whole DeltaNet layer, so the values reaching the kernels are the reference's own.
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

	// Device buffers persist across steps exactly as the resident runner would hold them — which
	// is what makes this a drift test rather than 64 independent single-step checks.
	alloc := func(n int) *gc.Buffer[float32] {
		b, e := gc.Alloc[float32](cctx, n)
		if e != nil {
			t.Fatalf("Alloc(%d): %v", n, e)
		}
		return b
	}
	up := func(v []float32) *gc.Buffer[float32] {
		b := alloc(len(v))
		if e := gc.CopyHtoD(bg, b, v); e != nil {
			t.Fatalf("H2D: %v", e)
		}
		return b
	}
	dMixed, dConv := alloc(convDim), alloc(convDim)
	dConvW, dWin := up(convW), up(make([]float32, (convK-1)*convDim)) // zeroed ring, like newDeltaState
	dBt, dAt := alloc(nv), alloc(nv)
	dDt, dNegA, dNormW := up(dtBias), up(negExpA), up(normW)
	dHeadP := alloc(nv * 2)
	dQn, dKn := alloc(keyDim), alloc(keyDim)
	dState := up(make([]float32, nv*hv*hk)) // zeroed, like newDeltaState
	dCore, dZ, dGated := alloc(valueDim), alloc(valueDim), alloc(valueDim)
	for _, b := range []*gc.Buffer[float32]{dMixed, dConv, dConvW, dWin, dBt, dAt, dDt, dNegA,
		dNormW, dHeadP, dQn, dKn, dState, dCore, dZ, dGated} {
		defer b.Close()
	}
	stream, err := cctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	down := func(b *gc.Buffer[float32], n int) []float32 {
		out := make([]float32, n)
		if e := gc.CopyDtoH(bg, out, b); e != nil {
			t.Fatalf("D2H: %v", e)
		}
		return out
	}

	type stage struct {
		name string
		cos  float64
		rel  float64
	}
	worst := []stage{{name: "delta_conv", cos: 1}, {name: "delta_gates", cos: 1},
		{name: "delta_rule", cos: 1}, {name: "delta_gnorm", cos: 1}}

	for step := range steps {
		decoder.DeltaNetStepForTest(nil, rnd(hidden, 1.0), cw, hidden, eps, cst)
		if capPre == nil {
			t.Fatal("capture hook never fired — the gate would be vacuous")
		}
		for _, c := range []struct {
			dst *gc.Buffer[float32]
			src []float32
		}{{dMixed, capMixed}, {dBt, capGateIn[:nv]}, {dAt, capGateIn[nv:]}, {dZ, capZ}} {
			if e := gc.CopyHtoD(bg, c.dst, c.src); e != nil {
				t.Fatalf("H2D step %d: %v", step, e)
			}
		}
		launch := func(f *gc.Function, cfg gc.LaunchConfig, args ...gc.KernelArg) {
			if e := f.LaunchOn(bg, stream, cfg, args...); e != nil {
				t.Fatalf("launch step %d: %v", step, e)
			}
		}
		launch(kConv, gc.LaunchConfig1D(convDim, 256),
			gc.Arg(dMixed), gc.Arg(dConvW), gc.Arg(dWin), gc.Arg(dConv),
			gc.ArgValue(int32(convDim)), gc.ArgValue(int32(convK)))
		launch(kGates, gc.LaunchConfig1D(nv, 64),
			gc.Arg(dBt), gc.Arg(dAt), gc.Arg(dDt), gc.Arg(dNegA), gc.Arg(dHeadP), gc.ArgValue(int32(nv)))
		launch(kNorm, gc.LaunchConfig{GridX: nk, GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 2 * 128 * 4},
			gc.Arg(dConv), gc.Arg(dQn), gc.Arg(dKn),
			gc.ArgValue(int32(nk)), gc.ArgValue(int32(hk)), gc.ArgValue(int32(keyDim)), gc.ArgValue(qScale))
		launch(kRule, gc.LaunchConfig1D(nv*hv, 128),
			gc.Arg(dQn), gc.Arg(dKn), gc.Arg(dConv), gc.Arg(dHeadP), gc.Arg(dState), gc.Arg(dCore),
			gc.ArgValue(int32(nv)), gc.ArgValue(int32(hk)), gc.ArgValue(int32(hv)),
			gc.ArgValue(int32(rep)), gc.ArgValue(int32(2*keyDim)))
		launch(kGNorm, gc.LaunchConfig{GridX: nv, GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 4},
			gc.Arg(dCore), gc.Arg(dZ), gc.Arg(dNormW), gc.Arg(dGated),
			gc.ArgValue(int32(nv)), gc.ArgValue(int32(hv)), gc.ArgValue(float32(eps)))
		if e := stream.Synchronize(bg); e != nil {
			t.Fatalf("sync step %d: %v", step, e)
		}

		for i, c := range []struct {
			want []float32
			got  []float32
		}{
			{capConv, down(dConv, convDim)},
			{capBG, down(dHeadP, nv*2)},
			{capPre, down(dCore, valueDim)},
			{capGated, down(dGated, valueDim)},
		} {
			cos, maxAbs := cosF32(c.want, c.got)
			var ss float64
			for _, v := range c.want {
				ss += float64(v) * float64(v)
			}
			// Relative to RMS, not absolute: |core| grows with the state, so a fixed absolute
			// bound would tighten silently over the run.
			rel := maxAbs / math.Sqrt(ss/float64(len(c.want)))
			if cos < worst[i].cos {
				worst[i].cos = cos
			}
			if rel > worst[i].rel {
				worst[i].rel = rel
			}
		}
	}

	t.Logf("DeltaNet mixer chain vs CPU on CUDA, %d steps at REAL geometry (nk=%d nv=%d hk=%d hv=%d):",
		steps, nk, nv, hk, hv)
	for _, w := range worst {
		t.Logf("  %-11s worst cosine=%.9f worst maxAbs/rms=%.3g", w.name, w.cos, w.rel)
		// f32 vs f32 on the same summation order: the only legitimate differences are the CPU's
		// float64 norm accumulators and the fast-math intrinsics (__expf/__logf) the kernels use
		// where the CPU calls libm. The plan's kill criterion says a recurrence that drifts is
		// worse than no kernel — do NOT loosen these without finding why.
		if w.cos < 0.999999 || w.rel > 1e-3 {
			t.Errorf("%s drifts from the CPU reference: worst cosine=%.9f worst maxAbs/rms=%.3g",
				w.name, w.cos, w.rel)
		}
	}
}

// negExpRef mirrors negExpAFromLog: the CPU weights store −exp(A_log), precomputed at load.
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

// delta_norm on its own, including the degenerate head the chain test cannot reach: a head whose
// conv output is exactly zero. silu(0) is 0, so an all-zero head is reachable, and there the two
// plausible spellings of the norm diverge completely rather than subtly — rsqrtf(0) is +inf and
// poisons every downstream state entry with NaN, while the reference's sqrt(1/(0+1e-6)) is a
// finite 1000 applied to a zero vector.
//
// This exists because the same mutation was tried on the WebGPU side and the drift test did NOT
// catch it: at ordinary magnitudes 1e-6 is far below f32 resolution. Carrying the epsilon in a
// kernel comment without a test that fails when it is removed is a claim, not a guarantee.
func TestDeltaNorm_zeroHead(t *testing.T) {
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit failed: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	cctx, err := dev.Primary()
	if err != nil {
		t.Skipf("no primary context: %v", err)
	}
	defer cctx.Close()
	bg := context.Background()
	mod, err := cctx.LoadModule(deltaNetPTX)
	if err != nil {
		t.Fatalf("JIT: %v", err)
	}
	kNorm, err := mod.Function("delta_norm")
	if err != nil {
		t.Fatalf("Function: %v", err)
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

	dConv, e1 := gc.Alloc[float32](cctx, len(conv))
	dQn, e2 := gc.Alloc[float32](cctx, keyDim)
	dKn, e3 := gc.Alloc[float32](cctx, keyDim)
	if e1 != nil || e2 != nil || e3 != nil {
		t.Fatalf("alloc: %v %v %v", e1, e2, e3)
	}
	defer dConv.Close()
	defer dQn.Close()
	defer dKn.Close()
	if err := gc.CopyHtoD(bg, dConv, conv); err != nil {
		t.Fatalf("H2D: %v", err)
	}
	stream, err := cctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	cfg := gc.LaunchConfig{GridX: nk, GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 2 * 128 * 4}
	if err := kNorm.LaunchOn(bg, stream, cfg, gc.Arg(dConv), gc.Arg(dQn), gc.Arg(dKn),
		gc.ArgValue(int32(nk)), gc.ArgValue(int32(hk)), gc.ArgValue(int32(keyDim)), gc.ArgValue(qScale)); err != nil {
		t.Fatalf("launch: %v", err)
	}
	if err := stream.Synchronize(bg); err != nil {
		t.Fatalf("sync: %v", err)
	}
	gotQ, gotK := make([]float32, keyDim), make([]float32, keyDim)
	if e := gc.CopyDtoH(bg, gotQ, dQn); e != nil {
		t.Fatalf("D2H: %v", e)
	}
	if e := gc.CopyDtoH(bg, gotK, dKn); e != nil {
		t.Fatalf("D2H: %v", e)
	}

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
			// Against the CPU normalizer itself, not a reimplementation of it.
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
