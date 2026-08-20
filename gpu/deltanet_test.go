//go:build gpu && goinfer_testhooks

package gpu

import (
	"math"
	"math/rand"
	"testing"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestDeltaRule_cpuParity — the Gated-DeltaNet recurrence on the GPU vs the CPU reference, at
// REAL head geometry, driven for enough tokens that a drifting state shows.
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
	if err := ctx.ensureDeltaRule(); err != nil {
		t.Fatal(err)
	}
	if err := ctx.ensureDeltaNorm(); err != nil {
		t.Fatal(err)
	}

	// Qwen3.8-27B's real DeltaNet geometry. `hidden` is small on purpose: it sizes only the
	// projections, which this kernel does not perform (they are ordinary GEMVs), while the head
	// dims size the recurrence, which it does.
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

	// CPU side: a whole DeltaNet layer, so the values reaching the recurrence are the reference's
	// own — no hand-built intermediate for this test to get wrong.
	cw, cst := decoder.NewDeltaNetForTest(convK, hk, hv, nk, nv, hidden,
		rnd(convDim*hidden, 0.05), rnd(valueDim*hidden, 0.05), rnd(nv*hidden, 0.05), rnd(nv*hidden, 0.05),
		rnd(convDim*convK, 0.5), rnd(nv, 0.1), negExpRef(rnd(nv, 0.5)), rnd(hv, 0.1), rnd(hidden*valueDim, 0.05))

	var capConv, capBG, capCore []float32
	decoder.SetDeltaCapHook(func(conv, bg, core []float32) {
		capConv, capBG, capCore = conv, bg, append(capCore[:0], core...) // core is overwritten by step 4
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
	stor := wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst
	convBuf := mk("conv", convDim, stor)
	qnBuf := mk("qn", keyDim, wgpu.BufferUsageStorage)
	knBuf := mk("kn", keyDim, wgpu.BufferUsageStorage)
	vBuf := mk("v", valueDim, stor)
	headPBuf := mk("headP", nv*2, stor)
	stateBuf := mk("state", nv*hv*hk, stor)
	yBuf := mk("y", valueDim, wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
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

	bg := func(layout *wgpu.BindGroupLayout, bufs ...*wgpu.Buffer) *wgpu.BindGroup {
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
	normBG := bg(ctx.deltaNormLayout, convBuf, qnBuf, knBuf, normDims)
	ruleBG := bg(ctx.deltaRuleLayout, qnBuf, knBuf, vBuf, headPBuf, stateBuf, yBuf, ruleDims)
	defer func() {
		normBG.Release()
		ruleBG.Release()
		for _, b := range []*wgpu.Buffer{convBuf, qnBuf, knBuf, vBuf, headPBuf, stateBuf, yBuf, stag, normDims, ruleDims} {
			b.Release()
		}
	}()

	worstCos, worstRel := 1.0, 0.0
	for step := range steps {
		decoder.DeltaNetStepForTest(nil, rnd(hidden, 1.0), cw, hidden, 1e-5, cst)
		if capCore == nil {
			t.Fatal("capture hook never fired — the gate would be vacuous")
		}

		ctx.queue.WriteBuffer(convBuf, 0, wgpu.ToBytes(capConv))
		ctx.queue.WriteBuffer(vBuf, 0, wgpu.ToBytes(capConv[2*keyDim:]))
		ctx.queue.WriteBuffer(headPBuf, 0, wgpu.ToBytes(capBG))

		enc, _ := ctx.device.CreateCommandEncoder(nil)
		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(ctx.deltaNormPipeline)
		pass.SetBindGroup(0, normBG, nil)
		pass.DispatchWorkgroups(uint32((nk+31)/32), 1, 1)
		pass.SetPipeline(ctx.deltaRulePipeline)
		pass.SetBindGroup(0, ruleBG, nil)
		pass.DispatchWorkgroups(uint32((nv*hv+63)/64), 1, 1)
		pass.End()
		pass.Release()
		enc.CopyBufferToBuffer(yBuf, 0, stag, 0, uint64(valueDim*4))
		cmd, _ := enc.Finish(nil)
		ctx.queue.Submit(cmd)
		cmd.Release()
		enc.Release()

		st := wgpu.BufferMapAsyncStatusUnknown
		stag.MapAsync(wgpu.MapModeRead, 0, uint64(valueDim*4), func(s wgpu.BufferMapAsyncStatus) { st = s })
		ctx.device.Poll(true, nil)
		if st != wgpu.BufferMapAsyncStatusSuccess {
			t.Fatalf("map: %v", st)
		}
		got := make([]float32, valueDim)
		copy(got, wgpu.FromBytes[float32](stag.GetMappedRange(0, uint(valueDim*4))))
		stag.Unmap()

		cos, maxAbs := cosSim(capCore, got)
		var ss float64
		for _, v := range capCore {
			ss += float64(v) * float64(v)
		}
		rms := math.Sqrt(ss / valueDim)
		rel := maxAbs / rms // absolute error alone says nothing: |core| grows with the state
		if cos < worstCos {
			worstCos = cos
		}
		if rel > worstRel {
			worstRel = rel
		}
		if step < 3 || (step+1)%16 == 0 {
			t.Logf("  step %2d: cosine=%.9f maxAbs/rms=%.3g (rms=%.4g)", step+1, cos, rel, rms)
		}
	}
	t.Logf("deltaRule vs CPU, %d steps at REAL geometry (nk=%d nv=%d hk=%d hv=%d): worst cosine=%.9f worst maxAbs/rms=%.3g",
		steps, nk, nv, hk, hv, worstCos, worstRel)

	// f32-vs-f32 on identical summation order: the only legitimate sources of difference are the
	// CPU's float64 norm accumulator and FMA contraction. The plan's kill criterion says a
	// recurrence that drifts is worse than no kernel — do NOT loosen these without finding why.
	if worstCos < 0.999999 || worstRel > 1e-3 {
		t.Errorf("deltaRule drifts from the CPU recurrence: worst cosine=%.9f worst maxAbs/rms=%.3g",
			worstCos, worstRel)
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
