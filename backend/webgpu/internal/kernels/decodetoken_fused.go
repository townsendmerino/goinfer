//go:build gpu

package kernels

import (
	"fmt"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// DecodeTokenFused is the fast one-command-BUFFER decode forward: it records every
// dispatch of every layer into a SINGLE command encoder (one dispatch per pass, so
// pass-boundary barriers keep the data dependencies correct), with the KV-append
// copies between passes — then ONE Submit + ONE Poll, where the per-op-submit DecodeToken
// issues hundreds. Bit-exact-identical to DecodeToken (TestDecodeTokenFused_parity).
func (c *Context) DecodeTokenFused(x []float32, m ModelW, hidden, nH, nKV, hd, inter, pos, start int, eps, scale float32, addOne bool) ([]float32, error) {
	if err := c.ensureGEMV(); err != nil {
		return nil, err
	}
	if err := c.ensureQuantize(); err != nil {
		return nil, err
	}
	if err := c.ensureLayer(); err != nil {
		return nil, err
	}
	if err := c.ensureAttn(); err != nil {
		return nil, err
	}

	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	fe := &fusedEnc{
		c: c, name: "DecodeTokenFused", storUsage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
		hidden: hidden, nH: nH, nKV: nKV, hd: hd, inter: inter, start: start, kvDim: nKV * hd,
		eps: eps, scale: scale, addOne: addOne, enc: enc,
	}
	defer fe.release()

	xd, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(x), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, err
	}
	fe.keepBuf(xd)
	fe.noSinks = fe.storF(1)
	fe.noHasSink = fe.uni([]uint32{0, 0, 0, 0})

	for i := range m.Layers {
		fe.fusedLayer(&m.Layers[i], xd, pos)
	}
	// final norm + LM head
	xnf := fe.rms(xd, m.FinalNorm.buf)
	fq, fs := fe.quant(xnf, hidden)
	logits := fe.gemv(fq, fs, m.LMHead)

	// A device-allocation/bind failure anywhere above surfaces here as an error, not a panic or a
	// read of an uninitialised `logits` buffer.
	if fe.buildErr != nil {
		return nil, fe.buildErr
	}
	return fe.readOne(logits, m.LMHead.rows)
}

// fusedLayer encodes one layer of the one row at pos: the attention sub-block, then the MLP sub-block.
func (fe *fusedEnc) fusedLayer(lw *LayerW, xd *wgpu.Buffer, pos int) {
	hidden, nH, nKV, hd, inter := fe.hidden, fe.nH, fe.nKV, fe.hd, fe.inter
	xn := fe.rms(xd, lw.Attn.Norm.buf)
	aq, as := fe.quant(xn, hidden)
	q := fe.gemv(aq, as, lw.Attn.QProj)
	k := fe.gemv(aq, as, lw.Attn.KProj)
	v := fe.gemv(aq, as, lw.Attn.VProj)
	fe.rope(q, lw.Attn.InvFreq.buf, nH, pos)
	fe.rope(k, lw.Attn.InvFreq.buf, nKV, pos)
	fe.copyKV(k, lw.Attn.KCache.buf, pos)
	fe.copyKV(v, lw.Attn.VCache.buf, pos)
	ctxv := fe.attn(q, lw, pos)
	cq, cs := fe.quant(ctxv, nH*hd)
	attnOut := fe.gemv(cq, cs, lw.Attn.OProj)
	fe.residual(xd, attnOut)
	xn2 := fe.rms(xd, lw.MLPNorm.buf)
	mq, ms := fe.quant(xn2, hidden)
	gate := fe.gemv(mq, ms, lw.Gate)
	up := fe.gemv(mq, ms, lw.Up)
	mid := fe.swiglu(gate, up)
	dq, ds := fe.quant(mid, inter)
	down := fe.gemv(dq, ds, lw.Down)
	fe.residual(xd, down)
}

// readOne copies the logits to a staging buffer, submits the one command buffer and reads them back after one fence.
func (fe *fusedEnc) readOne(logits *wgpu.Buffer, vocab int) ([]float32, error) {
	c := fe.c
	stag, err := c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Size: uint64(vocab * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	if err != nil {
		return nil, err
	}
	defer releaseBuf(stag)
	if err := fe.enc.TryCopyBufferToBuffer(logits, 0, stag, 0, uint64(vocab*4)); err != nil {
		return nil, fmt.Errorf("gpu: DecodeTokenFused: logits copy: %w", err)
	}
	cmd, err := fe.enc.TryFinish(nil)
	if err != nil {
		return nil, err
	}
	defer cmd.Release()
	c.queue.Submit(cmd) // the ONE submit

	st := wgpu.MapAsyncStatus(0)
	if err := stag.TryMapAsync(wgpu.MapModeRead, 0, uint64(vocab*4), func(s wgpu.MapAsyncStatus) { st = s }); err != nil {
		return nil, err
	}
	c.device.Poll(true, nil) // the ONE fence
	if st != wgpu.MapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: DecodeTokenFused map failed: %v", st)
	}
	out := make([]float32, vocab)
	copy(out, wgpu.FromBytes[float32](stag.GetMappedRange(0, uint(vocab*4))))
	stag.TryUnmap()
	return out, nil
}
