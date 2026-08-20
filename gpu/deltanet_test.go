//go:build gpu && goinfer_testhooks

package gpu

import (
	"math"
	"math/rand"
	"testing"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestDeltaRule_cpuParity — the whole Gated-DeltaNet mixer chain on the GPU vs the CPU reference,
// at REAL head geometry, driven for enough tokens that a drifting state shows.
//
// Four kernels, run CHAINED (each consumes the previous one's real GPU output, not the CPU's), so
// this gates the composition the resident runner will execute rather than four isolated
// primitives — the A′ zero-copy post-mortem's lesson, recorded as "isolation proves the primitive,
// never the composition". Each stage is scored separately so a failure names the culprit:
//
//	deltaGates → (beta, decay)   deltaNorm → (q̂, k̂)   deltaRule → core   deltaGNorm → gated
//
// The two GEMVs and the causal conv are deliberately outside: they are ordinary matmuls and
// mambaConv, both already gated.
//
// WHY REAL GEOMETRY AND NOT THE TINY GOLDEN. testdata/qwen35_deltanet_golden.json pins a real HF
// layer, but at hk=hv=8, nv=4: 32 threads, and a state row shorter than a cache line. Qwen3.8 runs
// hk=hv=128, nk=16, nv=48 — 6144 threads each owning a 128-float row. Those are the numbers the
// kernel has to be right at, and the scaled-fixture discipline exists here
// (TestGemma4MoEScaled_residentParity) precisely because toy widths hide this class of bug.
//
// WHY COMPARE TO THE CPU AND NOT TO HF. The CPU recurrence is already gated against transformers'
// torch_recurrent_gated_delta_rule (decoder's DeltaNet golden), so this makes the chain
// kernel ≡ CPU ≡ HF. A reference written inside this package would be a second unvalidated
// implementation of the thing under test.
//
// WHY MANY STEPS. A recurrence can agree at step 1 and be visibly wrong at step 50 — the state is
// the carrier. Same reasoning as TestMambaSSM_driftParity; the running worst is logged so a
// failure says whether the error is constant (a formula bug) or growing (a state bug).
func TestDeltaRule_cpuParity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()
	for _, e := range []func() error{ctx.ensureDeltaRule, ctx.ensureDeltaNorm, ctx.ensureDeltaGates, ctx.ensureDeltaGNorm} {
		if err := e(); err != nil {
			t.Fatal(err)
		}
	}

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

	// CPU side: a whole DeltaNet layer, so the values reaching the kernels are the reference's own
	// — no hand-built intermediate for this test to get wrong. dtBias/negExpA/normW are held
	// because deltaGates and deltaGNorm read them as weights.
	dtBias, negExpA, normW := rnd(nv, 0.1), negExpRef(rnd(nv, 0.5)), rnd(hv, 0.1)
	cw, cst := decoder.NewDeltaNetForTest(convK, hk, hv, nk, nv, hidden,
		rnd(convDim*hidden, 0.05), rnd(valueDim*hidden, 0.05), rnd(nv*hidden, 0.05), rnd(nv*hidden, 0.05),
		rnd(convDim*convK, 0.5), dtBias, negExpA, normW, rnd(hidden*valueDim, 0.05))

	var capConv, capGateIn, capBG, capPre, capGated, capZ []float32
	decoder.SetDeltaCapHook(func(conv, gateIn, bg, pre, gated, z []float32) {
		capConv, capGateIn, capBG, capPre, capGated, capZ = conv, gateIn, bg, pre, gated, z
	})
	defer decoder.SetDeltaCapHook(nil)

	// GPU side: the buffers persist across steps exactly as the resident runner would hold them,
	// which is what makes this a drift test rather than 64 independent single-step checks.
	mk := func(label string, n int, usage wgpu.BufferUsage) *wgpu.Buffer {
		b, e := ctx.device.CreateBuffer(&wgpu.BufferDescriptor{Label: label, Size: uint64(n * 4), Usage: usage})
		if e != nil {
			t.Fatal(e)
		}
		return b
	}
	init := func(label string, c []float32) *wgpu.Buffer {
		b, e := ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: label, Contents: wgpu.ToBytes(c), Usage: wgpu.BufferUsageStorage})
		if e != nil {
			t.Fatal(e)
		}
		return b
	}
	stor := wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst
	out := wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc
	convBuf := mk("conv", convDim, stor)
	vBuf := mk("v", valueDim, stor)
	zBuf := mk("z", valueDim, stor)
	btBuf, atBuf := mk("bt", nv, stor), mk("at", nv, stor)
	dtBuf, naBuf, nwBuf := init("dtBias", dtBias), init("negExpA", negExpA), init("normW", normW)
	qnBuf, knBuf := mk("qn", keyDim, wgpu.BufferUsageStorage), mk("kn", keyDim, wgpu.BufferUsageStorage)
	headPBuf := mk("headP", nv*2, out)
	stateBuf := mk("state", nv*hv*hk, stor)
	coreBuf := mk("core", valueDim, out)
	gatedBuf := mk("gated", valueDim, out)
	stag := mk("stag", valueDim, wgpu.BufferUsageMapRead|wgpu.BufferUsageCopyDst)
	ctx.queue.WriteBuffer(stateBuf, 0, wgpu.ToBytes(make([]float32, nv*hv*hk))) // zeroed, like newDeltaState

	uni := func(label string, w []uint32) *wgpu.Buffer {
		b, e := ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label: label, Contents: wgpu.ToBytes(w), Usage: wgpu.BufferUsageUniform})
		if e != nil {
			t.Fatal(e)
		}
		return b
	}
	normDims := uni("ndims", []uint32{nk, hk, keyDim, 0, math.Float32bits(qScale), 0, 0, 0})
	ruleDims := uni("rdims", []uint32{nv, nk, hk, hv, rep, 0, 0, 0})
	gateDims := uni("gdims", []uint32{nv, 0, 0, 0})
	gnDims := uni("gndims", []uint32{nv, hv, 0, 0, math.Float32bits(eps), 0, 0, 0})

	bgOf := func(layout *wgpu.BindGroupLayout, bufs ...*wgpu.Buffer) *wgpu.BindGroup {
		ents := make([]wgpu.BindGroupEntry, len(bufs))
		for i, b := range bufs {
			ents[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: b, Size: b.GetSize()}
		}
		g, e := ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: ents})
		if e != nil {
			t.Fatal(e)
		}
		return g
	}
	normBG := bgOf(ctx.deltaNormLayout, convBuf, qnBuf, knBuf, normDims)
	gateBG := bgOf(ctx.deltaGatesLayout, btBuf, atBuf, dtBuf, naBuf, headPBuf, gateDims)
	ruleBG := bgOf(ctx.deltaRuleLayout, qnBuf, knBuf, vBuf, headPBuf, stateBuf, coreBuf, ruleDims)
	gnBG := bgOf(ctx.deltaGNormLayout, coreBuf, zBuf, nwBuf, gatedBuf, gnDims)
	defer func() {
		for _, g := range []*wgpu.BindGroup{normBG, gateBG, ruleBG, gnBG} {
			g.Release()
		}
		for _, b := range []*wgpu.Buffer{convBuf, vBuf, zBuf, btBuf, atBuf, dtBuf, naBuf, nwBuf,
			qnBuf, knBuf, headPBuf, stateBuf, coreBuf, gatedBuf, stag, normDims, ruleDims, gateDims, gnDims} {
			b.Release()
		}
	}()

	read := func(src *wgpu.Buffer, n int) []float32 {
		e2, _ := ctx.device.CreateCommandEncoder(nil)
		e2.CopyBufferToBuffer(src, 0, stag, 0, uint64(n*4))
		c2, _ := e2.Finish(nil)
		ctx.queue.Submit(c2)
		c2.Release()
		e2.Release()
		st := wgpu.BufferMapAsyncStatusUnknown
		stag.MapAsync(wgpu.MapModeRead, 0, uint64(n*4), func(s wgpu.BufferMapAsyncStatus) { st = s })
		ctx.device.Poll(true, nil)
		if st != wgpu.BufferMapAsyncStatusSuccess {
			t.Fatalf("map: %v", st)
		}
		o := make([]float32, n)
		copy(o, wgpu.FromBytes[float32](stag.GetMappedRange(0, uint(n*4))))
		stag.Unmap()
		return o
	}

	// Worst-over-steps per stage, so a failure names WHICH kernel drifted rather than only that
	// the chain did. The stages run chained (deltaRule consumes deltaGates' and deltaNorm's real
	// GPU output, not the CPU's), so an error in an early kernel reaches the late ones — which is
	// the composition the resident runner will actually execute.
	type stage struct {
		name string
		cos  float64
		rel  float64
	}
	worst := []stage{{name: "deltaGates", cos: 1}, {name: "deltaRule", cos: 1}, {name: "deltaGNorm", cos: 1}}
	for step := range steps {
		decoder.DeltaNetStepForTest(nil, rnd(hidden, 1.0), cw, hidden, eps, cst)
		if capPre == nil {
			t.Fatal("capture hook never fired — the gate would be vacuous")
		}

		ctx.queue.WriteBuffer(convBuf, 0, wgpu.ToBytes(capConv))
		ctx.queue.WriteBuffer(vBuf, 0, wgpu.ToBytes(capConv[2*keyDim:]))
		ctx.queue.WriteBuffer(zBuf, 0, wgpu.ToBytes(capZ))
		ctx.queue.WriteBuffer(btBuf, 0, wgpu.ToBytes(capGateIn[:nv]))
		ctx.queue.WriteBuffer(atBuf, 0, wgpu.ToBytes(capGateIn[nv:]))

		enc, _ := ctx.device.CreateCommandEncoder(nil)
		pass := enc.BeginComputePass(nil)
		for _, d := range []struct {
			pl *wgpu.ComputePipeline
			bg *wgpu.BindGroup
			n  int
			wg int
		}{
			{ctx.deltaGatesPipeline, gateBG, nv, 64},
			{ctx.deltaNormPipeline, normBG, nk, 32},
			{ctx.deltaRulePipeline, ruleBG, nv * hv, 64},
			{ctx.deltaGNormPipeline, gnBG, nv, 64},
		} {
			pass.SetPipeline(d.pl)
			pass.SetBindGroup(0, d.bg, nil)
			pass.DispatchWorkgroups(uint32((d.n+d.wg-1)/d.wg), 1, 1)
		}
		pass.End()
		pass.Release()
		cmd, _ := enc.Finish(nil)
		ctx.queue.Submit(cmd)
		cmd.Release()
		enc.Release()

		for i, c := range []struct {
			want []float32
			got  []float32
		}{
			{capBG, read(headPBuf, nv*2)},
			{capPre, read(coreBuf, valueDim)},
			{capGated, read(gatedBuf, valueDim)},
		} {
			cos, maxAbs := cosSim(c.want, c.got)
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
		if step < 2 || (step+1)%32 == 0 {
			t.Logf("  step %2d: gates cos=%.9f  rule cos=%.9f rel=%.2g  gnorm cos=%.9f",
				step+1, worst[0].cos, worst[1].cos, worst[1].rel, worst[2].cos)
		}
	}

	t.Logf("DeltaNet mixer chain vs CPU, %d steps at REAL geometry (nk=%d nv=%d hk=%d hv=%d):",
		steps, nk, nv, hk, hv)
	for _, w := range worst {
		t.Logf("  %-11s worst cosine=%.9f worst maxAbs/rms=%.3g", w.name, w.cos, w.rel)
		// f32-vs-f32 on identical summation order: the only legitimate sources of difference are
		// the CPU's float64 norm accumulators and FMA contraction. The plan's kill criterion says
		// a recurrence that drifts is worse than no kernel — do NOT loosen these without finding why.
		if w.cos < 0.999999 || w.rel > 1e-3 {
			t.Errorf("%s drifts from the CPU reference: worst cosine=%.9f worst maxAbs/rms=%.3g",
				w.name, w.cos, w.rel)
		}
	}
}

// TestDeltaNorm_cpuParity gates the q/k l2-normalizer on its own, including the degenerate head
// the recurrence gate cannot reach: a head whose conv output is exactly zero. silu(0) is 0, so an
// all-zero head is reachable in principle, and there the two plausible spellings of the norm
// diverge completely rather than subtly — inverseSqrt(0) is +inf and poisons the state with NaN,
// while the reference's sqrt(1/(0+1e-6)) is a finite 1000 applied to a zero vector. At ordinary
// magnitudes 1e-6 is far below f32 resolution and the recurrence test cannot tell the two apart
// (verified by mutation), so this case is why the epsilon is in the shader.
func TestDeltaNorm_cpuParity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()
	if err := ctx.ensureDeltaNorm(); err != nil {
		t.Fatal(err)
	}

	const (
		nk, hk = 16, 128
		keyDim = nk * hk
	)
	qScale := float32(1 / math.Sqrt(float64(hk)))
	rng := rand.New(rand.NewSource(7))
	conv := make([]float32, 2*keyDim)
	for i := range conv {
		conv[i] = float32(rng.NormFloat64())
	}
	// head 0: q all zero (the +inf case). head 1: k denormal-small, so the epsilon dominates the
	// sum rather than merely nudging it. The rest stay ordinary.
	for i := range hk {
		conv[i] = 0
		conv[keyDim+hk+i] = 1e-7
	}

	mkS := func(c []float32) *wgpu.Buffer {
		b, e := ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(c), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
		if e != nil {
			t.Fatal(e)
		}
		return b
	}
	convB := mkS(conv)
	qnB, knB := mkS(make([]float32, keyDim)), mkS(make([]float32, keyDim))
	stag, _ := ctx.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(keyDim * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	dims, _ := ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Contents: wgpu.ToBytes([]uint32{nk, hk, keyDim, 0, math.Float32bits(qScale), 0, 0, 0}), Usage: wgpu.BufferUsageUniform})
	bgrp, _ := ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: ctx.deltaNormLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: convB, Size: convB.GetSize()}, {Binding: 1, Buffer: qnB, Size: qnB.GetSize()},
		{Binding: 2, Buffer: knB, Size: knB.GetSize()}, {Binding: 3, Buffer: dims, Size: dims.GetSize()},
	}})
	defer func() {
		bgrp.Release()
		for _, b := range []*wgpu.Buffer{convB, qnB, knB, stag, dims} {
			b.Release()
		}
	}()

	enc, _ := ctx.device.CreateCommandEncoder(nil)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(ctx.deltaNormPipeline)
	pass.SetBindGroup(0, bgrp, nil)
	pass.DispatchWorkgroups(uint32((nk+31)/32), 1, 1)
	pass.End()
	pass.Release()
	cmd, _ := enc.Finish(nil)
	ctx.queue.Submit(cmd)
	cmd.Release()
	enc.Release()

	read := func(src *wgpu.Buffer) []float32 {
		e2, _ := ctx.device.CreateCommandEncoder(nil)
		e2.CopyBufferToBuffer(src, 0, stag, 0, uint64(keyDim*4))
		c2, _ := e2.Finish(nil)
		ctx.queue.Submit(c2)
		c2.Release()
		e2.Release()
		st := wgpu.BufferMapAsyncStatusUnknown
		stag.MapAsync(wgpu.MapModeRead, 0, uint64(keyDim*4), func(s wgpu.BufferMapAsyncStatus) { st = s })
		ctx.device.Poll(true, nil)
		if st != wgpu.BufferMapAsyncStatusSuccess {
			t.Fatalf("map: %v", st)
		}
		out := make([]float32, keyDim)
		copy(out, wgpu.FromBytes[float32](stag.GetMappedRange(0, uint(keyDim*4))))
		stag.Unmap()
		return out
	}
	gotQ, gotK := read(qnB), read(knB)

	for _, c := range []struct {
		name  string
		got   []float32
		off   int
		scale float32
	}{{"q", gotQ, 0, qScale}, {"k", gotK, keyDim, 1}} {
		for h := range nk {
			want := decoder.L2NormScaledForTest(conv[c.off+h*hk:c.off+(h+1)*hk], c.scale)
			cos, maxAbs := cosSim(want, c.got[h*hk:(h+1)*hk])
			for i, v := range c.got[h*hk : (h+1)*hk] {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatalf("%s head %d element %d is %v — the epsilon is not doing its job", c.name, h, i, v)
				}
			}
			if h == 0 && c.name == "q" { // the all-zero head: cosine is undefined, absolute is what matters
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
	t.Logf("deltaNorm vs l2normScaled over %d heads incl. a zero head and a 1e-7 head: clean", nk)
}

// negExpRef mirrors negExpAFromLog: the CPU weights store −exp(A_log), precomputed at load.
func negExpRef(aLog []float32) []float32 {
	out := make([]float32, len(aLog))
	for i, v := range aLog {
		out[i] = float32(-math.Exp(float64(v)))
	}
	return out
}
