//go:build gpu

package gpu

import (
	"math"
	"math/rand"
	"testing"

	"github.com/cogentcore/webgpu/wgpu"
)

// softplusRef / cosSim mirror decoder.softplusf and the parity metric so the golden
// matches the f32 mamba2Step recurrence exactly (f64 A, f32 dA, f32 state).
func softplusRef(x float32) float32 {
	if x > 20 {
		return x
	}
	return float32(math.Log1p(math.Exp(float64(x))))
}

func cosSim(a, b []float32) (cos, maxAbs float64) {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
		if d := math.Abs(float64(a[i] - b[i])); d > maxAbs {
			maxAbs = d
		}
	}
	if na == 0 || nb == 0 {
		return 1, maxAbs
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb)), maxAbs
}

func siluRef(x float32) float32 { x64 := float64(x); return float32(x64 / (1 + math.Exp(-x64))) }

// TestMambaGatedNorm_parity gates gated = y·silu(z) + grouped RMSNorm for both nGroups=1
// (Granite) and nGroups>1 (Nemotron-H) vs mamba2.go step 5 / gatedRMSNormGrouped.
func TestMambaGatedNorm_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()
	if err := ctx.ensureMambaGNorm(); err != nil {
		t.Fatal(err)
	}
	const dInner = 2048
	eps := float32(1e-5)
	for _, nGroups := range []int{1, 8} {
		groupSize := dInner / nGroups
		rng := rand.New(rand.NewSource(int64(nGroups) + 11))
		y := make([]float32, dInner)
		z := make([]float32, dInner)
		w := make([]float32, dInner)
		for i := 0; i < dInner; i++ {
			y[i] = float32(rng.NormFloat64())
			z[i] = float32(rng.NormFloat64())
			w[i] = float32(rng.NormFloat64()*0.2 + 1)
		}
		mk := func(c []float32, u wgpu.BufferUsage) *wgpu.Buffer {
			b, _ := ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(c), Usage: u})
			return b
		}
		yB, zB, wB := mk(y, wgpu.BufferUsageStorage), mk(z, wgpu.BufferUsageStorage), mk(w, wgpu.BufferUsageStorage)
		gB, _ := ctx.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(dInner * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
		stag, _ := ctx.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(dInner * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
		dims := mk([]float32{}, wgpu.BufferUsageUniform)
		dims.Release()
		dims, _ = ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes([]uint32{dInner, uint32(nGroups), uint32(groupSize), math.Float32bits(eps), 0, 0, 0, 0}), Usage: wgpu.BufferUsageUniform})
		bg, _ := ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: ctx.mambaGNormLayout, Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: yB, Size: yB.GetSize()}, {Binding: 1, Buffer: zB, Size: zB.GetSize()},
			{Binding: 2, Buffer: wB, Size: wB.GetSize()}, {Binding: 3, Buffer: gB, Size: gB.GetSize()}, {Binding: 4, Buffer: dims, Size: dims.GetSize()},
		}})
		enc, _ := ctx.device.CreateCommandEncoder(nil)
		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(ctx.mambaGNormPipeline)
		pass.SetBindGroup(0, bg, nil)
		pass.DispatchWorkgroups(uint32(nGroups), 1, 1)
		pass.End()
		pass.Release()
		enc.CopyBufferToBuffer(gB, 0, stag, 0, uint64(dInner*4))
		cmd, _ := enc.Finish(nil)
		ctx.queue.Submit(cmd)
		cmd.Release()
		enc.Release()
		st := wgpu.BufferMapAsyncStatusUnknown
		stag.MapAsync(wgpu.MapModeRead, 0, uint64(dInner*4), func(s wgpu.BufferMapAsyncStatus) { st = s })
		ctx.device.Poll(true, nil)
		if st != wgpu.BufferMapAsyncStatusSuccess {
			t.Fatalf("map: %v", st)
		}
		got := make([]float32, dInner)
		copy(got, wgpu.FromBytes[float32](stag.GetMappedRange(0, uint(dInner*4))))
		stag.Unmap()
		// golden
		gold := make([]float32, dInner)
		for i := 0; i < dInner; i++ {
			gold[i] = y[i] * siluRef(z[i])
		}
		for g := 0; g < nGroups; g++ {
			var ss float64
			for j := 0; j < groupSize; j++ {
				v := gold[g*groupSize+j]
				ss += float64(v) * float64(v)
			}
			inv := float32(1 / math.Sqrt(ss/float64(groupSize)+float64(eps)))
			for j := 0; j < groupSize; j++ {
				gold[g*groupSize+j] = w[g*groupSize+j] * gold[g*groupSize+j] * inv
			}
		}
		cos, maxAbs := cosSim(gold, got)
		t.Logf("  nGroups=%d: cosine=%.8f maxAbs=%.3g", nGroups, cos, maxAbs)
		if cos < 0.99999 || maxAbs > 1e-3 {
			t.Errorf("gatedNorm nGroups=%d drift: cosine=%.8f maxAbs=%.3g", nGroups, cos, maxAbs)
		}
		for _, b := range []*wgpu.Buffer{yB, zB, wB, gB, stag, dims} {
			b.Release()
		}
		bg.Release()
	}
}

// TestMambaConv_parity gates the depthwise causal conv + ring window over 64 tokens vs the
// mamba2.go step-2 reference (incl. the early-token zero-pad), checked every token.
func TestMambaConv_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()
	if err := ctx.ensureMambaConv(); err != nil {
		t.Fatal(err)
	}
	const convDim, K = 1024, 4
	rng := rand.New(rand.NewSource(3))
	convW := make([]float32, convDim*K)
	convB := make([]float32, convDim)
	for i := range convW {
		convW[i] = float32(rng.NormFloat64() * 0.4)
	}
	for i := range convB {
		convB[i] = float32(rng.NormFloat64() * 0.2)
	}

	mk := func(label string, n int, u wgpu.BufferUsage) *wgpu.Buffer {
		b, _ := ctx.device.CreateBuffer(&wgpu.BufferDescriptor{Label: label, Size: uint64(n * 4), Usage: u})
		return b
	}
	xBuf := mk("xBC", convDim, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	cwBuf, _ := ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "convW", Contents: wgpu.ToBytes(convW), Usage: wgpu.BufferUsageStorage})
	cbBuf, _ := ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "convB", Contents: wgpu.ToBytes(convB), Usage: wgpu.BufferUsageStorage})
	winBuf := mk("win", (K-1)*convDim, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	convBuf := mk("conv", convDim, wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	stag := mk("stag", convDim, wgpu.BufferUsageMapRead|wgpu.BufferUsageCopyDst)
	ctx.queue.WriteBuffer(winBuf, 0, wgpu.ToBytes(make([]float32, (K-1)*convDim))) // zero ring
	dims, _ := ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "dims", Contents: wgpu.ToBytes([]uint32{convDim, K, 0, 0}), Usage: wgpu.BufferUsageUniform})
	bg, _ := ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: ctx.mambaConvLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: xBuf, Size: xBuf.GetSize()}, {Binding: 1, Buffer: cwBuf, Size: cwBuf.GetSize()},
		{Binding: 2, Buffer: cbBuf, Size: cbBuf.GetSize()}, {Binding: 3, Buffer: winBuf, Size: winBuf.GetSize()},
		{Binding: 4, Buffer: convBuf, Size: convBuf.GetSize()}, {Binding: 5, Buffer: dims, Size: dims.GetSize()},
	}})
	defer func() {
		for _, b := range []*wgpu.Buffer{xBuf, cwBuf, cbBuf, winBuf, convBuf, stag, dims} {
			b.Release()
		}
		bg.Release()
	}()
	gpuConv := func(xBC []float32) []float32 {
		ctx.queue.WriteBuffer(xBuf, 0, wgpu.ToBytes(xBC))
		enc, _ := ctx.device.CreateCommandEncoder(nil)
		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(ctx.mambaConvPipeline)
		pass.SetBindGroup(0, bg, nil)
		pass.DispatchWorkgroups((convDim+63)/64, 1, 1)
		pass.End()
		pass.Release()
		enc.CopyBufferToBuffer(convBuf, 0, stag, 0, uint64(convDim*4))
		cmd, _ := enc.Finish(nil)
		ctx.queue.Submit(cmd)
		cmd.Release()
		enc.Release()
		st := wgpu.BufferMapAsyncStatusUnknown
		stag.MapAsync(wgpu.MapModeRead, 0, uint64(convDim*4), func(s wgpu.BufferMapAsyncStatus) { st = s })
		ctx.device.Poll(true, nil)
		if st != wgpu.BufferMapAsyncStatusSuccess {
			t.Fatalf("map: %v", st)
		}
		out := make([]float32, convDim)
		copy(out, wgpu.FromBytes[float32](stag.GetMappedRange(0, uint(convDim*4))))
		stag.Unmap()
		return out
	}
	var win [][]float32
	golden := func(xBC []float32) []float32 {
		conv := make([]float32, convDim)
		for c := 0; c < convDim; c++ {
			s := convB[c] + convW[c*K+(K-1)]*xBC[c]
			for j := 0; j < K-1; j++ {
				if idx := len(win) - (K - 1) + j; idx >= 0 {
					s += convW[c*K+j] * win[idx][c]
				}
			}
			conv[c] = siluRef(s)
		}
		win = append(win, append([]float32(nil), xBC...))
		if len(win) > K-1 {
			win = win[len(win)-(K-1):]
		}
		return conv
	}
	worst := 1.0
	for tok := 1; tok <= 64; tok++ {
		xBC := make([]float32, convDim)
		for i := range xBC {
			xBC[i] = float32(rng.NormFloat64())
		}
		cos, maxAbs := cosSim(golden(xBC), gpuConv(xBC))
		if cos < worst {
			worst = cos
		}
		if cos < 0.99999 || maxAbs > 1e-3 {
			t.Errorf("conv tok %d: cosine=%.8f maxAbs=%.3g", tok, cos, maxAbs)
		}
	}
	t.Logf("mambaConv worst cosine over 64 tokens: %.8f", worst)
}

// TestMambaLayer_compose chains conv → ssm → gatedNorm in ONE compute pass with persistent
// {win, ssm} state (the resident single-pass model), and parity-gates the composed mixer
// output vs the CPU reference (mamba2.go steps 2-5) over 300 tokens — the "one SSM layer
// correct" milestone before wiring the full interleaved model. (in/out_proj are linear GEMVs,
// validated separately; here random xBC/z/dt feed the new chain directly.)
func TestMambaLayer_compose(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()
	for _, e := range []func() error{ctx.ensureMambaConv, ctx.ensureMambaSSM, ctx.ensureMambaGNorm} {
		if err := e(); err != nil {
			t.Fatal(err)
		}
	}
	const nHeads, P, N, nGroups, K = 8, 64, 128, 2, 4
	const repeat = nHeads / nGroups
	const dInner = nHeads * P
	const gSize = nGroups * N
	const convDim = dInner + 2*gSize
	const normGroups = 1
	eps := float32(1e-5)
	rng := rand.New(rand.NewSource(5))
	convW := make([]float32, convDim*K)
	convB := make([]float32, convDim)
	headP := make([]float32, nHeads*3)
	aLog := make([]float32, nHeads)
	dtBias := make([]float32, nHeads)
	dW := make([]float32, nHeads)
	normW := make([]float32, dInner)
	for i := range convW {
		convW[i] = float32(rng.NormFloat64() * 0.4)
	}
	for i := range convB {
		convB[i] = float32(rng.NormFloat64() * 0.2)
	}
	for h := 0; h < nHeads; h++ {
		aLog[h], dtBias[h], dW[h] = float32(rng.NormFloat64()*0.5-1), float32(rng.NormFloat64()*0.3), float32(rng.NormFloat64()*0.5)
		headP[h*3], headP[h*3+1], headP[h*3+2] = float32(-math.Exp(float64(aLog[h]))), dtBias[h], dW[h]
	}
	for i := range normW {
		normW[i] = float32(rng.NormFloat64()*0.2 + 1)
	}
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
	xBuf, zBuf, dtBuf := mkRW(convDim, wgpu.BufferUsageCopyDst), mkRW(dInner, wgpu.BufferUsageCopyDst), mkRW(nHeads, wgpu.BufferUsageCopyDst)
	cwBuf, cbBuf, hpBuf, nwBuf := mkW(convW), mkW(convB), mkW(headP), mkW(normW)
	winBuf, convBuf, yBuf, gBuf := mkRW((K-1)*convDim, wgpu.BufferUsageCopyDst), mkRW(convDim, 0), mkRW(dInner, 0), mkRW(dInner, wgpu.BufferUsageCopySrc)
	ssmBuf := mkRW(nHeads*P*N, wgpu.BufferUsageCopyDst)
	stag, _ := dev.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(dInner * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	ctx.queue.WriteBuffer(winBuf, 0, wgpu.ToBytes(make([]float32, (K-1)*convDim)))
	ctx.queue.WriteBuffer(ssmBuf, 0, wgpu.ToBytes(make([]float32, nHeads*P*N)))
	dConv := uni([]uint32{convDim, K, 0, 0})
	dSSM := uni([]uint32{nHeads, P, N, nGroups, repeat, gSize, dInner, 0})
	dGN := uni([]uint32{dInner, normGroups, dInner / normGroups, math.Float32bits(eps), 0, 0, 0, 0})
	ent := func(bufs ...*wgpu.Buffer) []wgpu.BindGroupEntry {
		e := make([]wgpu.BindGroupEntry, len(bufs))
		for i, b := range bufs {
			e[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: b, Size: b.GetSize()}
		}
		return e
	}
	bgC, _ := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: ctx.mambaConvLayout, Entries: ent(xBuf, cwBuf, cbBuf, winBuf, convBuf, dConv)})
	bgS, _ := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: ctx.mambaSSMLayout, Entries: ent(convBuf, dtBuf, hpBuf, ssmBuf, yBuf, dSSM)})
	bgG, _ := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: ctx.mambaGNormLayout, Entries: ent(yBuf, zBuf, nwBuf, gBuf, dGN)})

	gpuMixer := func(xBC, z, dt []float32) []float32 {
		ctx.queue.WriteBuffer(xBuf, 0, wgpu.ToBytes(xBC))
		ctx.queue.WriteBuffer(zBuf, 0, wgpu.ToBytes(z))
		ctx.queue.WriteBuffer(dtBuf, 0, wgpu.ToBytes(dt))
		enc, _ := dev.CreateCommandEncoder(nil)
		pass := enc.BeginComputePass(nil) // ONE pass: conv → ssm → gatedNorm (data-dependent barriers)
		pass.SetPipeline(ctx.mambaConvPipeline)
		pass.SetBindGroup(0, bgC, nil)
		pass.DispatchWorkgroups((convDim+63)/64, 1, 1)
		pass.SetPipeline(ctx.mambaSSMPipeline)
		pass.SetBindGroup(0, bgS, nil)
		pass.DispatchWorkgroups((nHeads*P+63)/64, 1, 1)
		pass.SetPipeline(ctx.mambaGNormPipeline)
		pass.SetBindGroup(0, bgG, nil)
		pass.DispatchWorkgroups(normGroups, 1, 1)
		pass.End()
		pass.Release()
		enc.CopyBufferToBuffer(gBuf, 0, stag, 0, uint64(dInner*4))
		cmd, _ := enc.Finish(nil)
		ctx.queue.Submit(cmd)
		cmd.Release()
		enc.Release()
		stt := wgpu.BufferMapAsyncStatusUnknown
		stag.MapAsync(wgpu.MapModeRead, 0, uint64(dInner*4), func(s wgpu.BufferMapAsyncStatus) { stt = s })
		ctx.device.Poll(true, nil)
		if stt != wgpu.BufferMapAsyncStatusSuccess {
			t.Fatalf("map: %v", stt)
		}
		out := make([]float32, dInner)
		copy(out, wgpu.FromBytes[float32](stag.GetMappedRange(0, uint(dInner*4))))
		stag.Unmap()
		return out
	}
	// CPU golden: mamba2.go steps 2-5 with its own persistent state.
	var win [][]float32
	ssm := make([]float32, nHeads*P*N)
	cpuMixer := func(xBC, z, dt []float32) []float32 {
		conv := make([]float32, convDim)
		for c := 0; c < convDim; c++ {
			s := convB[c] + convW[c*K+(K-1)]*xBC[c]
			for j := 0; j < K-1; j++ {
				if idx := len(win) - (K - 1) + j; idx >= 0 {
					s += convW[c*K+j] * win[idx][c]
				}
			}
			conv[c] = siluRef(s)
		}
		win = append(win, append([]float32(nil), xBC...))
		if len(win) > K-1 {
			win = win[len(win)-(K-1):]
		}
		x, B, C := conv[:dInner], conv[dInner:dInner+gSize], conv[dInner+gSize:]
		y := make([]float32, dInner)
		for head := 0; head < nHeads; head++ {
			g := head / repeat
			A := -math.Exp(float64(aLog[head]))
			dth := softplusRef(dt[head] + dtBias[head])
			dA := float32(math.Exp(float64(dth) * A))
			for pi := 0; pi < P; pi++ {
				sBase := (head*P + pi) * N
				dx := dth * x[head*P+pi]
				var acc float32
				for n := 0; n < N; n++ {
					ssm[sBase+n] = ssm[sBase+n]*dA + dx*B[g*N+n]
					acc += ssm[sBase+n] * C[g*N+n]
				}
				y[head*P+pi] = acc + dW[head]*x[head*P+pi]
			}
		}
		gated := make([]float32, dInner)
		for i := range gated {
			gated[i] = y[i] * siluRef(z[i])
		}
		var ss float64
		for _, v := range gated {
			ss += float64(v) * float64(v)
		}
		inv := float32(1 / math.Sqrt(ss/float64(dInner)+float64(eps)))
		for i := range gated {
			gated[i] = normW[i] * gated[i] * inv
		}
		return gated
	}
	worst := 1.0
	for tok := 1; tok <= 300; tok++ {
		xBC := make([]float32, convDim)
		z := make([]float32, dInner)
		dt := make([]float32, nHeads)
		for i := range xBC {
			xBC[i] = float32(rng.NormFloat64() * 0.7)
		}
		for i := range z {
			z[i] = float32(rng.NormFloat64())
		}
		for i := range dt {
			dt[i] = float32(rng.NormFloat64() * 0.4)
		}
		cos, maxAbs := cosSim(cpuMixer(xBC, z, dt), gpuMixer(xBC, z, dt))
		if cos < worst {
			worst = cos
		}
		if tok == 1 || tok == 300 {
			t.Logf("  compose tok %d: cosine=%.8f maxAbs=%.3g", tok, cos, maxAbs)
		}
		if cos < 0.9999 || maxAbs > 1e-2 {
			t.Errorf("compose tok %d drift: cosine=%.8f maxAbs=%.3g", tok, cos, maxAbs)
		}
	}
	t.Logf("  one-pass conv→ssm→gatedNorm worst cosine over 300 tokens: %.8f", worst)
}

// TestMambaSSM_driftParity is the SPINE gate: the resident selective-SSM state update vs
// the f32 mamba2Step recurrence, run token-by-token with COMPOUNDING state, checked at
// 1/16/256/1k/2k tokens. A pass requires parity that does not DRIFT long (the stable
// dA∈(0,1) recurrence should keep f32 error bounded). Per-token inputs vary so the state
// genuinely evolves.
func TestMambaSSM_driftParity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()
	if err := ctx.ensureMambaSSM(); err != nil {
		t.Fatal(err)
	}
	const nHeads, P, N, nGroups = 8, 64, 128, 2
	const repeat = nHeads / nGroups
	const dInner = nHeads * P
	const gSize = nGroups * N
	const convDim = dInner + 2*gSize // [x|B|C]

	rng := rand.New(rand.NewSource(7))
	// static per-head weights: aLog, dtBias, D. headP for the kernel = {Aexp,-, D}.
	aLog := make([]float32, nHeads)
	dtBias := make([]float32, nHeads)
	dW := make([]float32, nHeads)
	headP := make([]float32, nHeads*3)
	for h := 0; h < nHeads; h++ {
		aLog[h] = float32(rng.NormFloat64()*0.5 - 1.0)
		dtBias[h] = float32(rng.NormFloat64() * 0.3)
		dW[h] = float32(rng.NormFloat64() * 0.5)
		headP[h*3+0] = float32(-math.Exp(float64(aLog[h]))) // Aexp (f64 then f32)
		headP[h*3+1] = dtBias[h]
		headP[h*3+2] = dW[h]
	}

	// GPU buffers (built once; ssm persists; conv/dtRaw/y rewritten per token).
	mkBuf := func(label string, n int, usage wgpu.BufferUsage) *wgpu.Buffer {
		b, e := ctx.device.CreateBuffer(&wgpu.BufferDescriptor{Label: label, Size: uint64(n * 4), Usage: usage})
		if e != nil {
			t.Fatal(e)
		}
		return b
	}
	convBuf := mkBuf("conv", convDim, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	dtBuf := mkBuf("dt", nHeads, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	headPBuf, _ := ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "headP", Contents: wgpu.ToBytes(headP), Usage: wgpu.BufferUsageStorage})
	ssmBuf := mkBuf("ssm", nHeads*P*N, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	yBuf := mkBuf("y", dInner, wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	stag := mkBuf("stag", dInner, wgpu.BufferUsageMapRead|wgpu.BufferUsageCopyDst)
	ctx.queue.WriteBuffer(ssmBuf, 0, wgpu.ToBytes(make([]float32, nHeads*P*N))) // zero state
	dims, _ := ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "dims",
		Contents: wgpu.ToBytes([]uint32{nHeads, P, N, nGroups, repeat, gSize, dInner, 0}), Usage: wgpu.BufferUsageUniform})
	bg, _ := ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: ctx.mambaSSMLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: convBuf, Size: convBuf.GetSize()}, {Binding: 1, Buffer: dtBuf, Size: dtBuf.GetSize()},
		{Binding: 2, Buffer: headPBuf, Size: headPBuf.GetSize()}, {Binding: 3, Buffer: ssmBuf, Size: ssmBuf.GetSize()},
		{Binding: 4, Buffer: yBuf, Size: yBuf.GetSize()}, {Binding: 5, Buffer: dims, Size: dims.GetSize()},
	}})
	defer func() {
		for _, b := range []*wgpu.Buffer{convBuf, dtBuf, headPBuf, ssmBuf, yBuf, stag, dims} {
			b.Release()
		}
		bg.Release()
	}()

	gpuStep := func(conv, dt []float32) []float32 {
		ctx.queue.WriteBuffer(convBuf, 0, wgpu.ToBytes(conv))
		ctx.queue.WriteBuffer(dtBuf, 0, wgpu.ToBytes(dt))
		enc, _ := ctx.device.CreateCommandEncoder(nil)
		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(ctx.mambaSSMPipeline)
		pass.SetBindGroup(0, bg, nil)
		pass.DispatchWorkgroups((nHeads*P+63)/64, 1, 1)
		pass.End()
		pass.Release()
		enc.CopyBufferToBuffer(yBuf, 0, stag, 0, uint64(dInner*4))
		cmd, _ := enc.Finish(nil)
		ctx.queue.Submit(cmd)
		cmd.Release()
		enc.Release()
		st := wgpu.BufferMapAsyncStatusUnknown
		stag.MapAsync(wgpu.MapModeRead, 0, uint64(dInner*4), func(s wgpu.BufferMapAsyncStatus) { st = s })
		ctx.device.Poll(true, nil)
		if st != wgpu.BufferMapAsyncStatusSuccess {
			t.Fatalf("map: %v", st)
		}
		out := make([]float32, dInner)
		copy(out, wgpu.FromBytes[float32](stag.GetMappedRange(0, uint(dInner*4))))
		stag.Unmap()
		return out
	}

	// golden: the f32 mamba2Step step-4 recurrence (f64 A, f32 dA, f32 state).
	ssm := make([]float32, nHeads*P*N)
	golden := func(conv, dt []float32) []float32 {
		x := conv[:dInner]
		B := conv[dInner : dInner+gSize]
		C := conv[dInner+gSize:]
		y := make([]float32, dInner)
		for head := 0; head < nHeads; head++ {
			g := head / repeat
			A := -math.Exp(float64(aLog[head]))
			dth := softplusRef(dt[head] + dtBias[head])
			dA := float32(math.Exp(float64(dth) * A))
			for pi := 0; pi < P; pi++ {
				sBase := (head*P + pi) * N
				dx := dth * x[head*P+pi]
				var acc float32
				for n := 0; n < N; n++ {
					ssm[sBase+n] = ssm[sBase+n]*dA + dx*B[g*N+n]
					acc += ssm[sBase+n] * C[g*N+n]
				}
				y[head*P+pi] = acc + dW[head]*x[head*P+pi]
			}
		}
		return y
	}

	checks := map[int]bool{1: true, 16: true, 256: true, 1024: true, 2048: true}
	worst := 1.0
	for tok := 1; tok <= 2048; tok++ {
		conv := make([]float32, convDim)
		for i := range conv {
			conv[i] = float32(rng.NormFloat64() * 0.7)
		}
		dt := make([]float32, nHeads)
		for i := range dt {
			dt[i] = float32(rng.NormFloat64() * 0.4)
		}
		yG := golden(conv, dt)
		yK := gpuStep(conv, dt)
		if checks[tok] {
			cos, maxAbs := cosSim(yG, yK)
			t.Logf("  tok %5d: cosine=%.8f maxAbs=%.3g", tok, cos, maxAbs)
			if cos < worst {
				worst = cos
			}
			if cos < 0.9999 || maxAbs > 1e-2 {
				t.Errorf("tok %d DRIFT: cosine=%.8f maxAbs=%.3g", tok, cos, maxAbs)
			}
		}
	}
	t.Logf("  worst cosine over 2k tokens: %.8f", worst)
}
