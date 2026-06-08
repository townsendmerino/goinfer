//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// DecodeTokenFused is the fast one-command-BUFFER decode forward: it records every
// dispatch of every layer into a SINGLE command encoder (one dispatch per pass, so
// pass-boundary barriers keep the data dependencies correct), with the KV-append
// copies between passes — then ONE Submit + ONE Poll. The per-op-submit DecodeToken
// was correct but issued ~500 submits; this collapses them to one, which is what
// the §0.5 probe measured as the win. Bit-exact-identical to DecodeToken
// (TestDecodeTokenFused_parity).
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
	kvDim := nKV * hd
	var keep []func()
	rel := func() {
		for _, f := range keep {
			f()
		}
	}
	defer rel()
	keepBuf := func(b *wgpu.Buffer) { keep = append(keep, b.Release) }
	keepBG := func(b *wgpu.BindGroup) { keep = append(keep, b.Release) }

	enc, err := c.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	defer enc.Release()

	storF := func(n int) *wgpu.Buffer {
		b, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
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
	// op shorthands recording into enc:
	rms := func(in, w *wgpu.Buffer) *wgpu.Buffer {
		out := storF(hidden)
		p := uni([]uint32{uint32(hidden), f32bits(eps), boolU32(addOne), 0})
		disp(c.rmsnormPipeline, bind(c.rmsnormLayout, in, w, out, p), 64, 1)
		return out
	}
	quant := func(in *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer) {
		kp := padK(K)
		q := storF((kp / 4))
		s := storF(1)
		p := uni([]uint32{1, uint32(K), uint32(kp), 0})
		disp(c.quantizePipeline, bind(c.quantizeLayout, in, q, s, p), 1, 1)
		return q, s
	}
	gemv := func(aq, as *wgpu.Buffer, rm *ResidentW8A8) *wgpu.Buffer {
		out := storF(rm.rows)
		p := uni([]uint32{1, uint32(rm.kp), uint32(rm.rows), 0})
		bg := bind(c.gemvLayout, aq, rm.bq, as, rm.bScales, out, p)
		gx, gy := gemvGrid(rm.rows)
		disp(c.gemvPipeline, bg, gx, gy)
		return out
	}
	rope := func(vec, invFreq *wgpu.Buffer, heads int) {
		half := hd / 2
		p := uni([]uint32{uint32(heads), uint32(hd), uint32(half), uint32(pos), f32bits(1), 0, 0, 0})
		disp(c.ropePipeline, bind(c.ropeLayout, vec, invFreq, p), uint32(heads*half+63)/64, 1)
	}
	residual := func(x, y *wgpu.Buffer) {
		p := uni([]uint32{uint32(hidden), 0, 0, 0})
		disp(c.residualPipeline, bind(c.residualLayout, x, y, p), uint32(hidden+63)/64, 1)
	}
	copyKV := func(src, cache *wgpu.Buffer) {
		enc.CopyBufferToBuffer(src, 0, cache, uint64(pos*kvDim*4), uint64(kvDim*4))
	}

	xd, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(x), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, err
	}
	keepBuf(xd)

	for i := range m.Layers {
		lw := &m.Layers[i]
		// attention sub-block
		xn := rms(xd, lw.Attn.Norm.buf)
		aq, as := quant(xn, hidden)
		q := gemv(aq, as, lw.Attn.QProj)
		k := gemv(aq, as, lw.Attn.KProj)
		v := gemv(aq, as, lw.Attn.VProj)
		rope(q, lw.Attn.InvFreq.buf, nH)
		rope(k, lw.Attn.InvFreq.buf, nKV)
		copyKV(k, lw.Attn.KCache.buf)
		copyKV(v, lw.Attn.VCache.buf)
		ctxv := storF(nH * hd)
		ap := uni([]uint32{uint32(nH), uint32(nKV), uint32(hd), uint32(pos + 1), uint32(start), uint32(nH / nKV), f32bits(scale), 0})
		disp(c.attnPipeline, bind(c.attnLayout, q, lw.Attn.KCache.buf, lw.Attn.VCache.buf, ctxv, ap), uint32(nH), 1)
		cq, cs := quant(ctxv, nH*hd)
		attnOut := gemv(cq, cs, lw.Attn.OProj)
		residual(xd, attnOut)
		// MLP sub-block
		xn2 := rms(xd, lw.MLPNorm.buf)
		mq, ms := quant(xn2, hidden)
		gate := gemv(mq, ms, lw.Gate)
		up := gemv(mq, ms, lw.Up)
		mid := storF(inter)
		sp := uni([]uint32{uint32(inter), 0, 0, 0})
		disp(c.swigluPipeline, bind(c.swigluLayout, gate, up, mid, sp), uint32(inter+63)/64, 1)
		dq, ds := quant(mid, inter)
		down := gemv(dq, ds, lw.Down)
		residual(xd, down)
	}
	// final norm + LM head
	xnf := rms(xd, m.FinalNorm.buf)
	fq, fs := quant(xnf, hidden)
	logits := gemv(fq, fs, m.LMHead)

	stag, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(m.LMHead.rows * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	if err != nil {
		return nil, err
	}
	defer stag.Release()
	enc.CopyBufferToBuffer(logits, 0, stag, 0, uint64(m.LMHead.rows*4))
	cmd, err := enc.Finish(nil)
	if err != nil {
		return nil, err
	}
	defer cmd.Release()
	c.queue.Submit(cmd) // the ONE submit

	st := wgpu.BufferMapAsyncStatusUnknown
	if err := stag.MapAsync(wgpu.MapModeRead, 0, uint64(m.LMHead.rows*4), func(s wgpu.BufferMapAsyncStatus) { st = s }); err != nil {
		return nil, err
	}
	c.device.Poll(true, nil) // the ONE fence
	if st != wgpu.BufferMapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: DecodeTokenFused map failed: %v", st)
	}
	out := make([]float32, m.LMHead.rows)
	copy(out, wgpu.FromBytes[float32](stag.GetMappedRange(0, uint(m.LMHead.rows*4))))
	stag.Unmap()
	return out, nil
}
