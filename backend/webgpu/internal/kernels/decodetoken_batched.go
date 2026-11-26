//go:build gpu

package kernels

import (
	"fmt"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// DecodeTokenFusedBatched is the Stage-B (docs/spec/07) batched verify forward: it runs M token rows (a speculative block at
// consecutive positions) through the dense W8A8 layers, with the weight-heavy PROJECTIONS done as M=K tiled GEMMs (each
// weight streamed once across all M rows) and the cheap per-row ops (rmsnorm, RoPE, attention, SwiGLU, residual, KV-store)
// looped over the rows.
//
// It is bit-equivalent to M sequential DecodeTokenFused calls at positions[0..M-1] (same int8 inputs, same int32
// accumulation; the tiled GEMM equals the GEMV): all rows' K/V are written to the shared cache before any row's attention
// reads it, so row i attends to rows 0..i-1 as sequential decode would. Gated by TestDecodeTokenFusedBatched_parity. Dense
// W8A8 only (no MoE/MLA/SSM/bias/QK-norm).
//
// NOT one command buffer end to end: Metal allocates a native MTLCommandBuffer inside every ComputePassEncoder.TryEnd() and
// MTLCommandQueue caps how many can be uncommitted at once, so a single encoder spanning all M rows x all layers blocks on
// that cap and deadlocks (Submit is unreachable). flushPasses commits periodically without a Poll (same-queue submissions
// are ordered, so KV writes still happen-before later attention reads); it exists only to stay under the cap, so keep it.
// The weight-stream-once win is across M rows within a layer, not across the whole block.
// History: docs/code-notes/gpu.md#DecodeTokenFusedBatched.
func (c *Context) DecodeTokenFusedBatched(xs [][]float32, m ModelW, hidden, nH, nKV, hd, inter int, positions []int, start int, eps, scale float32, addOne bool) ([][]float32, error) {
	M := len(xs)
	if M == 0 {
		return nil, nil
	}
	// The thin-M gemmRow kernel accumulates into a private array<i32, gemmRowMaxM> (gemm_rows.go),
	// so M rows past that silently alias the last accumulator under WGSL's robustness clamp —
	// wrong logits, no error. The one-shot MatmulW8A8GemmRow guards this and so does this path: reject rather than
	// corrupt; the caller (spec verify) chunks its block.
	if M > gemmRowMaxM {
		return nil, fmt.Errorf("gpu: DecodeTokenFusedBatched M=%d exceeds gemmRowMaxM=%d (chunk the block)", M, gemmRowMaxM)
	}
	if len(positions) != M {
		return nil, fmt.Errorf("gpu: DecodeTokenFusedBatched %d rows but %d positions", M, len(positions))
	}
	for _, f := range []func() error{c.ensureGEMV, c.ensureQuantize, c.ensureLayer, c.ensureAttn, c.ensureGemmRow} {
		if err := f(); err != nil {
			return nil, err
		}
	}

	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	// passesPerFlush is well under the smallest dispatch count (~139, the sequential per-row
	// path's total for an 8-layer decode) observed to run without hitting Metal's uncommitted-
	// command-buffer cap, so it stays safe across every M/L combination this function is called
	// with rather than depending on a per-config layer or row count.
	const passesPerFlush = 32
	fe := &fusedEnc{
		c: c, name: "DecodeTokenFusedBatched", storUsage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
		flushEvery: passesPerFlush, hidden: hidden, nH: nH, nKV: nKV, hd: hd, inter: inter, start: start, kvDim: nKV * hd,
		eps: eps, scale: scale, addOne: addOne, enc: enc,
	}
	defer fe.release()

	// Upload the M input rows.
	xd := make([]*wgpu.Buffer, M)
	for r := range xs {
		b, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(xs[r]), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
		if err != nil {
			return nil, err
		}
		fe.keepBuf(b)
		xd[r] = b
	}
	fe.noSinks = fe.storF(1)
	fe.noHasSink = fe.uni([]uint32{0, 0, 0, 0})

	for i := range m.Layers {
		fe.batchedLayer(&m.Layers[i], xd, positions)
	}

	// final norm + LM head (batched)
	xnf := make([]*wgpu.Buffer, M)
	for r := range xnf {
		xnf[r] = fe.rms(xd[r], m.FinalNorm.buf)
	}
	logits := fe.tiledProj(xnf, m.LMHead)

	// Surface any device-allocation/bind failure as an error before the readback, instead of
	// copying from uninitialised `logits` rows.
	if fe.buildErr != nil {
		return nil, fe.buildErr
	}
	return fe.readRows(logits, m.LMHead.rows)
}

// batchedLayer encodes one layer over every row: batched q/k/v projection, per-row RoPE + KV-store, per-row attention,
// batched o projection, then the MLP with batched gate/up, per-row SwiGLU and batched down.
func (fe *fusedEnc) batchedLayer(lw *LayerW, xd []*wgpu.Buffer, positions []int) {
	M, nH, nKV, kvDim := len(xd), fe.nH, fe.nKV, fe.kvDim
	xn := make([]*wgpu.Buffer, M)
	for r := range xn {
		xn[r] = fe.rms(xd[r], lw.Attn.Norm.buf)
	}
	q := fe.tiledProj(xn, lw.Attn.QProj)
	k := fe.tiledProj(xn, lw.Attn.KProj)
	v := fe.tiledProj(xn, lw.Attn.VProj)
	for r := range xd {
		fe.rope(q[r], lw.Attn.InvFreq.buf, nH, positions[r])
		fe.rope(k[r], lw.Attn.InvFreq.buf, nKV, positions[r])
		fe.cpy(k[r], 0, lw.Attn.KCache.buf, uint64(positions[r]*kvDim*4), uint64(kvDim*4))
		fe.cpy(v[r], 0, lw.Attn.VCache.buf, uint64(positions[r]*kvDim*4), uint64(kvDim*4))
	}
	// All rows' K/V are now in the cache; each row attends to its causal prefix
	// (including earlier rows of this block).
	ctxv := make([]*wgpu.Buffer, M)
	for r := range xd {
		ctxv[r] = fe.attn(q[r], lw, positions[r])
	}
	attnOut := fe.tiledProj(ctxv, lw.Attn.OProj)
	for r := range xd {
		fe.residual(xd[r], attnOut[r])
	}
	xn2 := make([]*wgpu.Buffer, M)
	for r := range xn2 {
		xn2[r] = fe.rms(xd[r], lw.MLPNorm.buf)
	}
	gate := fe.tiledProj(xn2, lw.Gate)
	up := fe.tiledProj(xn2, lw.Up)
	mid := make([]*wgpu.Buffer, M)
	for r := range xd {
		mid[r] = fe.swiglu(gate[r], up[r])
	}
	down := fe.tiledProj(mid, lw.Down)
	for r := range xd {
		fe.residual(xd[r], down[r])
	}
}

// readRows copies each row's logits to a staging buffer, submits, and reads every row back after one fence.
func (fe *fusedEnc) readRows(logits []*wgpu.Buffer, vocab int) ([][]float32, error) {
	c, M := fe.c, len(logits)
	stag := make([]*wgpu.Buffer, M)
	for r := range stag {
		s, err := c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Size: uint64(vocab * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
		if err != nil {
			return nil, err
		}
		fe.keepBuf(s)
		if err := fe.enc.TryCopyBufferToBuffer(logits[r], 0, s, 0, uint64(vocab*4)); err != nil {
			return nil, fmt.Errorf("gpu: DecodeTokenFusedBatched: logits copy[%d]: %w", r, err)
		}
		stag[r] = s
	}
	cmd, err := fe.enc.TryFinish(nil)
	if err != nil {
		return nil, err
	}
	defer cmd.Release()
	c.queue.Submit(cmd) // the FINAL submit — everything since the last flushPasses, plus the readback copies

	statuses := make([]wgpu.MapAsyncStatus, M)
	for r := range stag {
		idx := r
		if err := stag[r].TryMapAsync(wgpu.MapModeRead, 0, uint64(vocab*4), func(s wgpu.MapAsyncStatus) { statuses[idx] = s }); err != nil {
			return nil, err
		}
	}
	c.device.Poll(true, nil) // the ONE fence for all M rows
	out := make([][]float32, M)
	for r := range stag {
		if statuses[r] != wgpu.MapAsyncStatusSuccess {
			return nil, fmt.Errorf("gpu: DecodeTokenFusedBatched map[%d] failed: %v", r, statuses[r])
		}
		row := make([]float32, vocab)
		copy(row, wgpu.FromBytes[float32](stag[r].GetMappedRange(0, uint(vocab*4))))
		stag[r].TryUnmap()
		out[r] = row
	}
	return out, nil
}
