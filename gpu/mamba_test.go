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
