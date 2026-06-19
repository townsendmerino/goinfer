//go:build gpu

package gpu

import (
	"math"
	"math/rand"
	"testing"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/townsendmerino/aikit/linalg"
)

// P2 parity: the resident non-gated squared-ReLU FFN — up(W8A8) → relu²→int8 (relu2Quant) →
// down(W8A8) — vs the CPU reference down(relu²(up(x))). int8 weights, real-ish f32 activations
// over many tokens (state-free FFN, so per-token), like the mamba-kernel isolation parities.
func TestNemotronRelu2FFN_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()
	for _, e := range []func() error{ctx.ensureGEMV, ctx.ensureQuantize, ctx.ensureRelu2} {
		if err := e(); err != nil {
			t.Fatal(err)
		}
	}
	const hidden, inter = 512, 1408 // non-mult-of-16 inter exercises the pad path (1408=88*16 ok; use 1400)
	const interOdd = 1400
	rng := rand.New(rand.NewSource(11))
	up := make([]float32, interOdd*hidden) // up: [inter, hidden]
	down := make([]float32, hidden*interOdd)
	for i := range up {
		up[i] = float32(rng.NormFloat64() * 0.08)
	}
	for i := range down {
		down[i] = float32(rng.NormFloat64() * 0.08)
	}
	upQ, upS := linalg.QuantizeRowsInt8(up, interOdd, hidden)
	downQ, downS := linalg.QuantizeRowsInt8(down, hidden, interOdd)
	upRM, err := ctx.UploadW8A8(upQ, upS, interOdd, hidden)
	if err != nil {
		t.Fatal(err)
	}
	downRM, err := ctx.UploadW8A8(downQ, downS, hidden, interOdd)
	if err != nil {
		t.Fatal(err)
	}

	// GPU FFN: quantize x → up GEMV → relu2Quant → down GEMV. Built per call (isolation test).
	relu2f := func(x float32) float32 {
		if x < 0 {
			return 0
		}
		return x * x
	}
	dev := ctx.device
	stor := wgpu.BufferUsageStorage
	gpuFFN := func(x []float32) []float32 {
		// quantize x[hidden]→int8
		xq, xs := linalg.QuantizeRowsInt8(x, 1, hidden)
		xqb, _ := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(packInt8(xq, 1, hidden)), Usage: stor})
		xsb, _ := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(xs), Usage: stor})
		upOut, _ := dev.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(interOdd * 4), Usage: stor | wgpu.BufferUsageCopySrc})
		upDims, _ := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes([]uint32{1, uint32(padK(hidden)), uint32(interOdd), 0}), Usage: wgpu.BufferUsageUniform})
		upBG, _ := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: upRM.gLayout(ctx), Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: xqb, Size: xqb.GetSize()}, {Binding: 1, Buffer: upRM.wbuf(), Size: upRM.wbuf().GetSize()},
			{Binding: 2, Buffer: xsb, Size: xsb.GetSize()}, {Binding: 3, Buffer: upRM.sbuf(), Size: upRM.sbuf().GetSize()},
			{Binding: 4, Buffer: upOut, Size: upOut.GetSize()}, {Binding: 5, Buffer: upDims, Size: upDims.GetSize()}}})
		// relu2Quant: upOut[inter] → int8 [padK(inter)/4] + scale
		kp := padK(interOdd)
		rq, _ := dev.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(kp / 4 * 4), Usage: stor | wgpu.BufferUsageCopySrc})
		rs, _ := dev.CreateBuffer(&wgpu.BufferDescriptor{Size: 4, Usage: stor | wgpu.BufferUsageCopySrc})
		rDims, _ := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes([]uint32{uint32(interOdd), uint32(kp), 0, 0}), Usage: wgpu.BufferUsageUniform})
		rBG := ctx.relu2QuantBind(upOut, rq, rs, rDims)
		// down GEMV: rq[inter int8] → hidden
		dOut, _ := dev.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(hidden * 4), Usage: stor | wgpu.BufferUsageCopySrc})
		dDims, _ := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes([]uint32{1, uint32(padK(interOdd)), uint32(hidden), 0}), Usage: wgpu.BufferUsageUniform})
		dBG, _ := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: downRM.gLayout(ctx), Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: rq, Size: rq.GetSize()}, {Binding: 1, Buffer: downRM.wbuf(), Size: downRM.wbuf().GetSize()},
			{Binding: 2, Buffer: rs, Size: rs.GetSize()}, {Binding: 3, Buffer: downRM.sbuf(), Size: downRM.sbuf().GetSize()},
			{Binding: 4, Buffer: dOut, Size: dOut.GetSize()}, {Binding: 5, Buffer: dDims, Size: dDims.GetSize()}}})
		stag, _ := dev.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(hidden * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
		enc, _ := dev.CreateCommandEncoder(nil)
		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(upRM.gPipe(ctx))
		pass.SetBindGroup(0, upBG, nil)
		gx, gy := gemvGrid(interOdd)
		pass.DispatchWorkgroups(gx, gy, 1)
		pass.SetPipeline(ctx.relu2Pipeline)
		pass.SetBindGroup(0, rBG, nil)
		pass.DispatchWorkgroups(1, 1, 1)
		pass.SetPipeline(downRM.gPipe(ctx))
		pass.SetBindGroup(0, dBG, nil)
		gx2, gy2 := gemvGrid(hidden)
		pass.DispatchWorkgroups(gx2, gy2, 1)
		pass.End()
		pass.Release()
		enc.CopyBufferToBuffer(dOut, 0, stag, 0, uint64(hidden*4))
		cmd, _ := enc.Finish(nil)
		ctx.queue.Submit(cmd)
		cmd.Release()
		enc.Release()
		st := wgpu.BufferMapAsyncStatusUnknown
		stag.MapAsync(wgpu.MapModeRead, 0, uint64(hidden*4), func(s wgpu.BufferMapAsyncStatus) { st = s })
		ctx.device.Poll(true, nil)
		if st != wgpu.BufferMapAsyncStatusSuccess {
			t.Fatalf("map: %v", st)
		}
		out := make([]float32, hidden)
		copy(out, wgpu.FromBytes[float32](stag.GetMappedRange(0, uint(hidden*4))))
		stag.Unmap()
		for _, b := range []*wgpu.Buffer{xqb, xsb, upOut, upDims, rq, rs, rDims, dOut, dDims, stag} {
			b.Release()
		}
		return out
	}
	// CPU reference: down(relu²(up(x))), with the SAME int8 weights (round-tripped) so the only
	// difference is the resident int8 ACTIVATION quant — the parity we want (not weight quant).
	cpuFFN := func(x []float32) []float32 {
		up2 := make([]float32, interOdd)
		for r := 0; r < interOdd; r++ {
			var s float64
			for c := 0; c < hidden; c++ {
				s += float64(float32(upQ[r*hidden+c])*upS[r]) * float64(x[c])
			}
			up2[r] = relu2f(float32(s))
		}
		out := make([]float32, hidden)
		for r := 0; r < hidden; r++ {
			var s float64
			for c := 0; c < interOdd; c++ {
				s += float64(float32(downQ[r*interOdd+c])*downS[r]) * float64(up2[c])
			}
			out[r] = float32(s)
		}
		return out
	}
	worst := 1.0
	for tok := 0; tok < 32; tok++ {
		x := make([]float32, hidden)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		cos, mx := cosSim(cpuFFN(x), gpuFFN(x))
		if cos < worst {
			worst = cos
		}
		if cos < 0.998 {
			t.Errorf("relu² FFN tok %d: cosine=%.6f maxAbs=%.4g", tok, cos, mx)
		}
	}
	t.Logf("Nemotron relu² FFN (W8A8 up→relu2Quant→W8A8 down) vs CPU: worst cosine over 32 tokens=%.6f", worst)
	_ = math.Sqrt
}
