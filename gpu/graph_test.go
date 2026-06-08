//go:build gpu

package gpu

import (
	"testing"
	"time"

	"github.com/cogentcore/webgpu/wgpu"
)

// TestGraph_oneFencePerToken is the §0.5 probe: does recording a whole token's
// dispatches into ONE command buffer with ONE fence beat the current staged path
// (113 submit+poll syncs/token for the dense 1.5B)? It records the real per-token
// dispatch sequence — 28 layers × [rmsnorm×2, quantize×3, 7 gemv, swiglu] + final
// rmsnorm/quantize/lm_head = 367 dispatches at real shapes (H=1536, FFN=8960,
// vocab=151936) against resident weights — then times two replays of the IDENTICAL
// work: (A) one submit + one Poll; (B) flushed at 113 points (one Poll each). Data
// is not a correct forward pass (no attention compute); this measures the fence
// cost structure only. Logs; run -v.
func TestGraph_oneFencePerToken(t *testing.T) {
	if testing.Short() {
		t.Skip("graph probe")
	}
	ctx := newOrSkipHW(t)
	defer ctx.Close()
	if err := ctx.ensureGEMV(); err != nil {
		t.Fatal(err)
	}
	if err := ctx.ensureQuantize(); err != nil {
		t.Fatal(err)
	}
	if err := ctx.ensureLayer(); err != nil {
		t.Fatal(err)
	}

	const (
		H     = 1536
		KV    = 256
		FFN   = 8960
		VOCAB = 151936
		L     = 28
	)
	// resident weights (random; reused across layers — dispatch cost is identical)
	mkW := func(N, K int, seed uint64) *ResidentW8A8 {
		bq, bs := quantW(N, K, seed)
		rm, err := ctx.UploadW8A8(bq, bs, N, K)
		if err != nil {
			t.Fatal(err)
		}
		return rm
	}
	wQ := mkW(H, H, 1)
	wKV := mkW(KV, H, 2)
	wO := mkW(H, H, 3)
	wGate := mkW(FFN, H, 4)
	wDown := mkW(H, FFN, 5)
	wLM := mkW(VOCAB, H, 6)
	defer func() {
		for _, w := range []*ResidentW8A8{wQ, wKV, wO, wGate, wDown, wLM} {
			w.Release()
		}
	}()

	// scratch buffers
	stor := func(n int) *wgpu.Buffer {
		b, err := ctx.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	packBuf := func(words int) *wgpu.Buffer {
		b, err := ctx.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(words * 4), Usage: wgpu.BufferUsageStorage})
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	uni := func(v []uint32) *wgpu.Buffer {
		b, err := ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(v), Usage: wgpu.BufferUsageUniform})
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	xH, xnH := stor(H), stor(H)
	aH, asH := packBuf(H/4), stor(1)
	gI, uI, mI := stor(FFN), stor(FFN), stor(FFN)
	aI, asI := packBuf(padK(FFN)/4), stor(1)
	dSmall, dKV, dLM := stor(H), stor(KV), stor(VOCAB)
	stagLM, _ := ctx.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(VOCAB * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	defer stagLM.Release()

	// bindgroup builders
	gemvBG := func(aBuf, asBuf *wgpu.Buffer, rm *ResidentW8A8, dst *wgpu.Buffer) *wgpu.BindGroup {
		dims := uni([]uint32{1, uint32(rm.kp), uint32(rm.rows), 0})
		bg, err := ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: ctx.gemvLayout, Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: aBuf, Size: aBuf.GetSize()}, {Binding: 1, Buffer: rm.bq, Size: rm.bq.GetSize()},
			{Binding: 2, Buffer: asBuf, Size: asBuf.GetSize()}, {Binding: 3, Buffer: rm.bScales, Size: rm.bScales.GetSize()},
			{Binding: 4, Buffer: dst, Size: dst.GetSize()}, {Binding: 5, Buffer: dims, Size: dims.GetSize()},
		}})
		if err != nil {
			t.Fatal(err)
		}
		return bg
	}
	rmsW := stor(H)
	rmsBG := func() *wgpu.BindGroup {
		p := uni([]uint32{H, 0x3727c5ac, 0, 0}) // eps≈1e-5 bits
		bg, err := ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: ctx.rmsnormLayout, Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: xH, Size: xH.GetSize()}, {Binding: 1, Buffer: rmsW, Size: rmsW.GetSize()},
			{Binding: 2, Buffer: xnH, Size: xnH.GetSize()}, {Binding: 3, Buffer: p, Size: p.GetSize()},
		}})
		if err != nil {
			t.Fatal(err)
		}
		return bg
	}()
	quantBG := func(src, qb, sc *wgpu.Buffer, K int) *wgpu.BindGroup {
		p := uni([]uint32{1, uint32(K), uint32(padK(K)), 0})
		bg, err := ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: ctx.quantizeLayout, Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: src, Size: src.GetSize()}, {Binding: 1, Buffer: qb, Size: qb.GetSize()},
			{Binding: 2, Buffer: sc, Size: sc.GetSize()}, {Binding: 3, Buffer: p, Size: p.GetSize()},
		}})
		if err != nil {
			t.Fatal(err)
		}
		return bg
	}
	qH := quantBG(xnH, aH, asH, H)
	qI := quantBG(mI, aI, asI, FFN)
	swBG := func() *wgpu.BindGroup {
		p := uni([]uint32{FFN, 0, 0, 0})
		bg, err := ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: ctx.swigluLayout, Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: gI, Size: gI.GetSize()}, {Binding: 1, Buffer: uI, Size: uI.GetSize()},
			{Binding: 2, Buffer: mI, Size: mI.GetSize()}, {Binding: 3, Buffer: p, Size: p.GetSize()},
		}})
		if err != nil {
			t.Fatal(err)
		}
		return bg
	}()

	// op = one recorded dispatch
	type op struct {
		pl     *wgpu.ComputePipeline
		bg     *wgpu.BindGroup
		gx, gy uint32
	}
	gemvOp := func(bg *wgpu.BindGroup, n int) op { gx, gy := gemvGrid(n); return op{ctx.gemvPipeline, bg, gx, gy} }
	unit := func(pl *wgpu.ComputePipeline, bg *wgpu.BindGroup) op { return op{pl, bg, 1, 1} }
	swOp := op{ctx.swigluPipeline, swBG, (FFN + 63) / 64, 1}
	var ops []op
	addLayer := func() {
		ops = append(ops,
			unit(ctx.rmsnormPipeline, rmsBG),
			unit(ctx.quantizePipeline, qH),
			gemvOp(gemvBG(aH, asH, wQ, dSmall), H),
			gemvOp(gemvBG(aH, asH, wKV, dKV), KV),
			gemvOp(gemvBG(aH, asH, wKV, dKV), KV),
			gemvOp(gemvBG(aH, asH, wO, dSmall), H),
			unit(ctx.rmsnormPipeline, rmsBG),
			unit(ctx.quantizePipeline, qH),
			gemvOp(gemvBG(aH, asH, wGate, gI), FFN),
			gemvOp(gemvBG(aH, asH, wGate, uI), FFN),
			swOp,
			unit(ctx.quantizePipeline, qI),
			gemvOp(gemvBG(aI, asI, wDown, dSmall), H),
		)
	}
	for i := 0; i < L; i++ {
		addLayer()
	}
	ops = append(ops,
		unit(ctx.rmsnormPipeline, rmsBG),
		unit(ctx.quantizePipeline, qH),
		gemvOp(gemvBG(aH, asH, wLM, dLM), VOCAB),
	)
	t.Logf("recorded %d dispatches/token (L=%d)", len(ops), L)

	readLogits := func() {
		stagLM.MapAsync(wgpu.MapModeRead, 0, uint64(VOCAB*4), func(s wgpu.BufferMapAsyncStatus) { _ = s })
		ctx.device.Poll(true, nil)
		stagLM.Unmap()
	}

	// (A) one fence: all dispatches, one submit, one Poll.
	oneFence := func() {
		enc, _ := ctx.device.CreateCommandEncoder(nil)
		pass := enc.BeginComputePass(nil)
		for _, o := range ops {
			pass.SetPipeline(o.pl)
			pass.SetBindGroup(0, o.bg, nil)
			pass.DispatchWorkgroups(o.gx, o.gy, 1)
		}
		pass.End()
		enc.CopyBufferToBuffer(dLM, 0, stagLM, 0, uint64(VOCAB*4))
		cmd, ferr := enc.Finish(nil)
		if ferr != nil {
			t.Fatal(ferr)
		}
		ctx.queue.Submit(cmd)
		readLogits()
		cmd.Release()
		enc.Release()
	}

	// (B) staged: flush (submit + Poll) at 113 points — the real sync count.
	const fences = 113
	grp := (len(ops) + fences - 1) / fences
	staged := func() {
		for i := 0; i < len(ops); i += grp {
			end := i + grp
			if end > len(ops) {
				end = len(ops)
			}
			enc, _ := ctx.device.CreateCommandEncoder(nil)
			pass := enc.BeginComputePass(nil)
			for _, o := range ops[i:end] {
				pass.SetPipeline(o.pl)
				pass.SetBindGroup(0, o.bg, nil)
				pass.DispatchWorkgroups(o.gx, o.gy, 1)
			}
			pass.End()
			cmd, _ := enc.Finish(nil)
			ctx.queue.Submit(cmd)
			ctx.device.Poll(true, nil) // the per-group fence
			cmd.Release()
			enc.Release()
		}
	}

	const iters = 30
	oneFence()
	staged() // warm
	t0 := time.Now()
	for i := 0; i < iters; i++ {
		oneFence()
	}
	one := time.Since(t0) / iters
	t1 := time.Now()
	for i := 0; i < iters; i++ {
		staged()
	}
	stg := time.Since(t1) / iters

	t.Logf("per-token (%d dispatches)  |  ONE fence %.3f ms (%.1f tok/s)  |  STAGED %d fences %.3f ms (%.1f tok/s)",
		len(ops), ms(one), 1000/ms(one), (len(ops)+grp-1)/grp, ms(stg), 1000/ms(stg))
	t.Logf("  → one-fence vs staged %.2f×", float64(stg)/float64(one))
}

// quantW makes a random int8 weight [N,K] + per-row scales (no f32 alloc of the
// whole matrix at once for the big vocab case).
func quantW(N, K int, seed uint64) ([]int8, []float32) {
	q := make([]int8, N*K)
	s := make([]float32, N)
	st := seed
	for i := range q {
		st = st*6364136223846793005 + 1442695040888963407
		q[i] = int8(int(st>>56) - 128)
	}
	for i := range s {
		s[i] = 0.01
	}
	return q, s
}
