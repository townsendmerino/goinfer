//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// DecodeTokenFusedBatched is the Stage-B (docs/spec/07) batched verify forward: it
// runs M token rows (a speculative block at consecutive positions) through the dense
// W8A8 layers in ONE command buffer, with the weight-heavy PROJECTIONS done as M=K
// tiled GEMMs — each weight streamed once across all M rows — and the cheap per-row
// ops (rmsnorm, RoPE, attention, SwiGLU, residual, KV-store) looped over the rows.
// This is the projection-side weight-stream collapse that the per-row resident
// verify (runBatch: M separate M=1 GEMVs) cannot do.
//
// It is bit-equivalent to M sequential DecodeTokenFused calls at positions[0..M-1]
// (same int8 inputs, same int32 accumulation; the tiled GEMM equals the GEMV): all
// rows' K/V are written to the shared cache before any row's attention reads it, so
// row i attends to rows 0..i-1 exactly as sequential decode would. Gated by
// TestDecodeTokenFusedBatched_parity. Dense W8A8 only (no MoE/MLA/SSM/bias/QK-norm)
// — the first arch of the Stage-B rollout.
func (c *Context) DecodeTokenFusedBatched(xs [][]float32, m ModelW, hidden, nH, nKV, hd, inter int, positions []int, start int, eps, scale float32, addOne bool) ([][]float32, error) {
	M := len(xs)
	if M == 0 {
		return nil, nil
	}
	if len(positions) != M {
		return nil, fmt.Errorf("gpu: DecodeTokenFusedBatched %d rows but %d positions", M, len(positions))
	}
	for _, f := range []func() error{c.ensureGEMV, c.ensureQuantize, c.ensureLayer, c.ensureAttn, c.ensureTiled} {
		if err := f(); err != nil {
			return nil, err
		}
	}
	kvDim := nKV * hd

	var keep []func()
	defer func() {
		for _, f := range keep {
			f()
		}
	}()
	keepBuf := func(b *wgpu.Buffer) { keep = append(keep, b.Release) }
	keepBG := func(b *wgpu.BindGroup) { keep = append(keep, b.Release) }

	enc, err := c.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	defer enc.Release()

	storF := func(n int) *wgpu.Buffer {
		b, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
		keepBuf(b)
		return b
	}
	uni := func(v []uint32) *wgpu.Buffer {
		b, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(v), Usage: wgpu.BufferUsageUniform})
		keepBuf(b)
		return b
	}
	bind := func(layout *wgpu.BindGroupLayout, bufs ...*wgpu.Buffer) *wgpu.BindGroup {
		es := make([]wgpu.BindGroupEntry, len(bufs))
		for i, b := range bufs {
			es[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: b, Size: b.GetSize()}
		}
		bg, e := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: es})
		if e != nil {
			panic(e)
		}
		keepBG(bg)
		return bg
	}
	disp := func(pl *wgpu.ComputePipeline, bg *wgpu.BindGroup, gx, gy uint32) {
		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(pl)
		pass.SetBindGroup(0, bg, nil)
		pass.DispatchWorkgroups(gx, gy, 1)
		pass.End()
		pass.Release()
	}

	// --- per-row ops (one row's [hidden]/[dim] buffer) ---
	rms := func(in, w *wgpu.Buffer) *wgpu.Buffer {
		out := storF(hidden)
		p := uni([]uint32{uint32(hidden), f32bits(eps), boolU32(addOne), 0})
		disp(c.rmsnormPipeline, bind(c.rmsnormLayout, in, w, out, p), 64, 1)
		return out
	}
	quant1 := func(in *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer) {
		kp := padK(K)
		q := storF(kp / 4)
		s := storF(1)
		p := uni([]uint32{1, uint32(K), uint32(kp), 0})
		disp(c.quantizePipeline, bind(c.quantizeLayout, in, q, s, p), 1, 1)
		return q, s
	}
	rope := func(vec, invFreq *wgpu.Buffer, heads, pos int) {
		half := hd / 2
		p := uni([]uint32{uint32(heads), uint32(hd), uint32(half), uint32(pos), f32bits(1), 0, 0, 0})
		disp(c.ropePipeline, bind(c.ropeLayout, vec, invFreq, p), uint32(heads*half+63)/64, 1)
	}
	residual := func(x, y *wgpu.Buffer) {
		p := uni([]uint32{uint32(hidden), 0, 0, 0})
		disp(c.residualPipeline, bind(c.residualLayout, x, y, p), uint32(hidden+63)/64, 1)
	}

	// tiledProj batches a projection over all M rows: gather the M quantized rows
	// into one packed buffer, run ONE tiled GEMM (weights streamed once), scatter the
	// M output rows back to per-row buffers. This is the Stage-B weight-stream win.
	tiledProj := func(xnRows []*wgpu.Buffer, rm *ResidentW8A8) []*wgpu.Buffer {
		K, N := rm.cols, rm.rows
		kw := rm.kp / 4
		aqC := storF(M * kw)
		asC := storF(M)
		for r, xn := range xnRows {
			q, s := quant1(xn, K)
			enc.CopyBufferToBuffer(q, 0, aqC, uint64(r*kw*4), uint64(kw*4))
			enc.CopyBufferToBuffer(s, 0, asC, uint64(r*4), 4)
		}
		dstC := storF(M * N)
		p := uni([]uint32{uint32(M), uint32(rm.kp), uint32(N), 0})
		disp(c.tiledPipeline, bind(c.tiledLayout, aqC, rm.bq, asC, rm.bScales, dstC, p), (uint32(N)+15)/16, (uint32(M)+15)/16)
		outs := make([]*wgpu.Buffer, M)
		for r := range outs {
			o := storF(N)
			enc.CopyBufferToBuffer(dstC, uint64(r*N*4), o, 0, uint64(N*4))
			outs[r] = o
		}
		return outs
	}

	// Upload the M input rows.
	xd := make([]*wgpu.Buffer, M)
	for r := range xs {
		b, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(xs[r]), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
		if err != nil {
			return nil, err
		}
		keepBuf(b)
		xd[r] = b
	}

	for i := range m.Layers {
		lw := &m.Layers[i]
		// attention: batched q/k/v projection, per-row RoPE + KV-store, per-row attn,
		// batched o projection.
		xn := make([]*wgpu.Buffer, M)
		for r := range xn {
			xn[r] = rms(xd[r], lw.Attn.Norm.buf)
		}
		q := tiledProj(xn, lw.Attn.QProj)
		k := tiledProj(xn, lw.Attn.KProj)
		v := tiledProj(xn, lw.Attn.VProj)
		for r := range xd {
			rope(q[r], lw.Attn.InvFreq.buf, nH, positions[r])
			rope(k[r], lw.Attn.InvFreq.buf, nKV, positions[r])
			enc.CopyBufferToBuffer(k[r], 0, lw.Attn.KCache.buf, uint64(positions[r]*kvDim*4), uint64(kvDim*4))
			enc.CopyBufferToBuffer(v[r], 0, lw.Attn.VCache.buf, uint64(positions[r]*kvDim*4), uint64(kvDim*4))
		}
		// All rows' K/V are now in the cache; each row attends to its causal prefix
		// (including earlier rows of this block).
		ctxv := make([]*wgpu.Buffer, M)
		for r := range xd {
			cv := storF(nH * hd)
			ap := uni([]uint32{uint32(nH), uint32(nKV), uint32(hd), uint32(positions[r] + 1), uint32(start), uint32(nH / nKV), f32bits(scale), 0})
			disp(c.attnPipeline, bind(c.attnLayout, q[r], lw.Attn.KCache.buf, lw.Attn.VCache.buf, cv, ap), uint32(nH), 1)
			ctxv[r] = cv
		}
		attnOut := tiledProj(ctxv, lw.Attn.OProj)
		for r := range xd {
			residual(xd[r], attnOut[r])
		}
		// MLP: batched gate/up, per-row SwiGLU, batched down.
		xn2 := make([]*wgpu.Buffer, M)
		for r := range xn2 {
			xn2[r] = rms(xd[r], lw.MLPNorm.buf)
		}
		gate := tiledProj(xn2, lw.Gate)
		up := tiledProj(xn2, lw.Up)
		mid := make([]*wgpu.Buffer, M)
		for r := range xd {
			md := storF(inter)
			sp := uni([]uint32{uint32(inter), 0, 0, 0})
			disp(c.swigluPipeline, bind(c.swigluLayout, gate[r], up[r], md, sp), uint32(inter+63)/64, 1)
			mid[r] = md
		}
		down := tiledProj(mid, lw.Down)
		for r := range xd {
			residual(xd[r], down[r])
		}
	}

	// final norm + LM head (batched)
	xnf := make([]*wgpu.Buffer, M)
	for r := range xnf {
		xnf[r] = rms(xd[r], m.FinalNorm.buf)
	}
	logits := tiledProj(xnf, m.LMHead)

	vocab := m.LMHead.rows
	stag := make([]*wgpu.Buffer, M)
	for r := range stag {
		s, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(vocab * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
		if err != nil {
			return nil, err
		}
		keepBuf(s)
		enc.CopyBufferToBuffer(logits[r], 0, s, 0, uint64(vocab*4))
		stag[r] = s
	}
	cmd, err := enc.Finish(nil)
	if err != nil {
		return nil, err
	}
	defer cmd.Release()
	c.queue.Submit(cmd) // the ONE submit for the whole block

	statuses := make([]wgpu.BufferMapAsyncStatus, M)
	for r := range stag {
		idx := r
		if err := stag[r].MapAsync(wgpu.MapModeRead, 0, uint64(vocab*4), func(s wgpu.BufferMapAsyncStatus) { statuses[idx] = s }); err != nil {
			return nil, err
		}
	}
	c.device.Poll(true, nil) // the ONE fence for all M rows
	out := make([][]float32, M)
	for r := range stag {
		if statuses[r] != wgpu.BufferMapAsyncStatusSuccess {
			return nil, fmt.Errorf("gpu: DecodeTokenFusedBatched map[%d] failed: %v", r, statuses[r])
		}
		row := make([]float32, vocab)
		copy(row, wgpu.FromBytes[float32](stag[r].GetMappedRange(0, uint(vocab*4))))
		stag[r].Unmap()
		out[r] = row
	}
	return out, nil
}
