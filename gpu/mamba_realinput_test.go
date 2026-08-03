//go:build gpu

package gpu

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// Phase-A reconciliation (docs/ssm-int8-quality.md): the GPU conv/ssm/gatedNorm kernels pass
// isolation parity on RANDOM inputs over 300 tokens (TestMambaLayer_compose) yet the resident
// granite drops to 66%. This test feeds the kernels REAL granite layer-0 inputs — captured from
// an actual model run — over a multi-token sequence and diffs the GPU mixer output against
// mamba2Step's. Diverge → kernel-math bug in the real-input regime; match → the bug is wiring
// (the resident hands the kernels different inputs/state than mamba2Step gets).
func TestMambaRealInputParity(t *testing.T) {
	requireHeavyModel(t)
	if os.Getenv("GOINFER_SSM_QUALITY") == "" {
		t.Skip("real-input mamba parity — needs granite model; set GOINFER_SSM_QUALITY=1")
	}
	ctx := newOrSkipHW(t)
	defer ctx.Close()
	for _, e := range []func() error{ctx.ensureMambaConv, ctx.ensureMambaSSM, ctx.ensureMambaGNorm} {
		if err := e(); err != nil {
			t.Fatal(err)
		}
	}
	path := os.ExpandEnv("$HOME/models/granite/granite-4.0-h-tiny-Q8_0.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no granite model: %v", err)
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()

	hidden, nLayers, _, _, _, _, _ := m.Dims()
	_ = hidden
	nHeads, headDim, dState, nGroups, dConv, _, _, _, _, ok := m.GraniteResidentParams()
	if !ok {
		t.Fatal("not granite")
	}
	dInner := nHeads * headDim
	gSize := nGroups * dState
	convDim := dInner + 2*gSize
	K := dConv
	projDim := 2*dInner + 2*gSize + nHeads
	eps := float32(m.NormEps())
	nMamba := 0
	for i := 0; i < nLayers; i++ {
		if m.GraniteMambaLayer(i) {
			nMamba++
		}
	}
	t.Logf("dims: nHeads=%d headDim=%d dState=%d nGroups=%d K=%d dInner=%d convDim=%d projDim=%d nMamba=%d/%d",
		nHeads, headDim, dState, nGroups, K, dInner, convDim, projDim, nMamba, nLayers)

	// Capture layer-0 (first mamba layer) in_proj output + mixer output across a real run.
	var capProj, capGated [][]float32
	cnt := 0
	decoder.SetMambaCapHook(func(proj, gated []float32) {
		if cnt%nMamba == 0 { // first mamba layer of each token == actual layer 0
			capProj = append(capProj, append([]float32(nil), proj...))
			capGated = append(capGated, append([]float32(nil), gated...))
		}
		cnt++
	})
	pid, _ := tk.Encode("The Eiffel Tower is a wrought-iron lattice tower in Paris, France, built in 1889.", true)
	ch, gen := m.Generate(context.Background(), pid, 40, decoder.SamplingParams{Temperature: 0})
	for range ch {
	}
	if gen.Err() != nil {
		t.Fatal(gen.Err())
	}
	decoder.SetMambaCapHook(nil)
	t.Logf("captured %d tokens of real layer-0 mamba I/O", len(capProj))
	if len(capProj) < 10 {
		t.Fatalf("too few captures: %d", len(capProj))
	}

	// Real layer-0 weights → GPU compose (conv → ssm → gatedNorm), fresh persistent state.
	_, convW, convB, aLog, dW, dtBias, normW, _ := m.GraniteMambaWeights(0)
	headP := make([]float32, nHeads*3)
	for h := 0; h < nHeads; h++ {
		headP[h*3] = float32(-math.Exp(float64(aLog[h])))
		headP[h*3+1] = dtBias[h]
		headP[h*3+2] = dW[h]
	}
	repeat := nHeads / nGroups
	dev := ctx.device
	stor := wgpu.BufferUsageStorage
	mkW := func(c []float32) *wgpu.Buffer {
		b, _ := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(c), Usage: stor})
		return b
	}
	mkRW := func(n int, extra wgpu.BufferUsage) *wgpu.Buffer {
		b, _ := dev.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: stor | extra})
		return b
	}
	uni := func(v []uint32) *wgpu.Buffer {
		b, _ := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(v), Usage: wgpu.BufferUsageUniform})
		return b
	}
	xBuf := mkRW(convDim, wgpu.BufferUsageCopyDst)
	zBuf := mkRW(dInner, wgpu.BufferUsageCopyDst)
	dtBuf := mkRW(nHeads, wgpu.BufferUsageCopyDst)
	cwBuf, cbBuf, hpBuf, nwBuf := mkW(convW), mkW(convB), mkW(headP), mkW(normW)
	winBuf := mkRW((K-1)*convDim, wgpu.BufferUsageCopyDst)
	convBuf := mkRW(convDim, wgpu.BufferUsageCopySrc)
	yBuf := mkRW(dInner, wgpu.BufferUsageCopySrc)
	gBuf := mkRW(dInner, wgpu.BufferUsageCopySrc)
	ssmBuf := mkRW(nHeads*headDim*dState, wgpu.BufferUsageCopyDst)
	stag, _ := dev.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(convDim * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	ctx.queue.WriteBuffer(winBuf, 0, wgpu.ToBytes(make([]float32, (K-1)*convDim)))
	ctx.queue.WriteBuffer(ssmBuf, 0, wgpu.ToBytes(make([]float32, nHeads*headDim*dState)))
	dConvU := uni([]uint32{uint32(convDim), uint32(K), 0, 0})
	dSSMU := uni([]uint32{uint32(nHeads), uint32(headDim), uint32(dState), uint32(nGroups), uint32(repeat), uint32(gSize), uint32(dInner), 0})
	dGNU := uni([]uint32{uint32(dInner), 1, uint32(dInner), math.Float32bits(eps), 0, 0, 0, 0})
	ent := func(bufs ...*wgpu.Buffer) []wgpu.BindGroupEntry {
		e := make([]wgpu.BindGroupEntry, len(bufs))
		for i, b := range bufs {
			e[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: b, Size: b.GetSize()}
		}
		return e
	}
	bgC, _ := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: ctx.mambaConvLayout, Entries: ent(xBuf, cwBuf, cbBuf, winBuf, convBuf, dConvU)})
	bgS, _ := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: ctx.mambaSSMLayout, Entries: ent(convBuf, dtBuf, hpBuf, ssmBuf, yBuf, dSSMU)})
	bgG, _ := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: ctx.mambaGNormLayout, Entries: ent(yBuf, zBuf, nwBuf, gBuf, dGNU)})

	readBuf := func(b *wgpu.Buffer, n int) []float32 {
		enc, _ := dev.CreateCommandEncoder(nil)
		enc.CopyBufferToBuffer(b, 0, stag, 0, uint64(n*4))
		cmd, _ := enc.Finish(nil)
		ctx.queue.Submit(cmd)
		cmd.Release()
		enc.Release()
		st := wgpu.BufferMapAsyncStatusUnknown
		stag.MapAsync(wgpu.MapModeRead, 0, uint64(n*4), func(s wgpu.BufferMapAsyncStatus) { st = s })
		ctx.device.Poll(true, nil)
		if st != wgpu.BufferMapAsyncStatusSuccess {
			t.Fatalf("map: %v", st)
		}
		out := make([]float32, n)
		copy(out, wgpu.FromBytes[float32](stag.GetMappedRange(0, uint(n*4))))
		stag.Unmap()
		return out
	}
	gpuMixer := func(proj []float32) []float32 {
		z := proj[0:dInner]
		xBC := proj[dInner : dInner+convDim]
		dt := proj[projDim-nHeads:]
		ctx.queue.WriteBuffer(xBuf, 0, wgpu.ToBytes(xBC))
		ctx.queue.WriteBuffer(zBuf, 0, wgpu.ToBytes(z))
		ctx.queue.WriteBuffer(dtBuf, 0, wgpu.ToBytes(dt))
		enc, _ := dev.CreateCommandEncoder(nil)
		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(ctx.mambaConvPipeline)
		pass.SetBindGroup(0, bgC, nil)
		pass.DispatchWorkgroups(uint32(convDim+63)/64, 1, 1)
		pass.SetPipeline(ctx.mambaSSMPipeline)
		pass.SetBindGroup(0, bgS, nil)
		pass.DispatchWorkgroups(uint32(nHeads*headDim+63)/64, 1, 1)
		pass.SetPipeline(ctx.mambaGNormPipeline)
		pass.SetBindGroup(0, bgG, nil)
		pass.DispatchWorkgroups(1, 1, 1)
		pass.End()
		pass.Release()
		cmd, _ := enc.Finish(nil)
		ctx.queue.Submit(cmd)
		cmd.Release()
		enc.Release()
		return readBuf(gBuf, dInner)
	}

	worst := 1.0
	firstBad := -1
	for tok := 0; tok < len(capProj); tok++ {
		g := gpuMixer(capProj[tok])
		cos, mx := cosSim(capGated[tok], g)
		if cos < worst {
			worst = cos
		}
		if cos < 0.999 && firstBad < 0 {
			firstBad = tok
		}
		if tok < 5 || cos < 0.999 {
			t.Logf("  tok %2d: mixer cosine=%.6f maxAbs=%.4g", tok, cos, mx)
		}
	}
	t.Logf("REAL-INPUT mixer parity: worst cosine over %d tokens = %.6f; first divergence tok=%d", len(capProj), worst, firstBad)
	if worst < 0.999 {
		t.Logf("=> GPU kernels DIVERGE from mamba2Step on REAL inputs => kernel-math bug (fork A)")
	} else {
		t.Logf("=> GPU kernels MATCH mamba2Step on real inputs => the in-model bug is WIRING (fork B)")
	}
}
