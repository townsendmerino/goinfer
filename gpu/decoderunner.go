//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// DecodeRunner is the production one-command-buffer decode forward: it builds every
// scratch buffer + bind group ONCE (the per-token allocation that made
// DecodeTokenFused slow), then Run only WriteBuffers the input + the pos-dependent
// uniforms (RoPE pos, attention nKeys) and re-records the fixed dispatch plan into
// a fresh encoder — one Submit, one Poll, logits back. This is the GEMVRunner
// pattern applied to the whole token graph.
type DecodeRunner struct {
	c                    *Context
	steps                []runStep
	posUnis              []posUni
	xd, stag, lastLogits *wgpu.Buffer
	vocab                int
	kvDim                int
	keep                 []func()
}

type runStep struct {
	pl     *wgpu.ComputePipeline // nil ⇒ a KV-append copy
	bg     *wgpu.BindGroup
	gx, gy uint32
	src    *wgpu.Buffer // copy: source (k or v)
	cache  *wgpu.Buffer // copy: KV cache
}

type posUni struct {
	buf *wgpu.Buffer
	gen func(pos int) []uint32 // uniform contents for this pos
}

// NewDecodeRunner builds the persistent plan for a resident model.
func (c *Context) NewDecodeRunner(m ModelW, hidden, nH, nKV, hd, inter, start int, eps, scale float32, addOne bool) (*DecodeRunner, error) {
	for _, e := range []func() error{c.ensureGEMV, c.ensureQuantize, c.ensureLayer, c.ensureAttn} {
		if err := e(); err != nil {
			return nil, err
		}
	}
	r := &DecodeRunner{c: c, vocab: m.LMHead.rows, kvDim: nKV * hd}
	keepBuf := func(b *wgpu.Buffer) *wgpu.Buffer { r.keep = append(r.keep, b.Release); return b }
	keepBG := func(b *wgpu.BindGroup) *wgpu.BindGroup { r.keep = append(r.keep, b.Release); return b }
	storF := func(n int) *wgpu.Buffer {
		b, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
		return keepBuf(b)
	}
	uni := func(v []uint32) *wgpu.Buffer {
		b, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(v), Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst})
		return keepBuf(b)
	}
	bind := func(layout *wgpu.BindGroupLayout, bufs ...*wgpu.Buffer) *wgpu.BindGroup {
		es := make([]wgpu.BindGroupEntry, len(bufs))
		for i, b := range bufs {
			es[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: b, Size: b.GetSize()}
		}
		bg, e := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: es})
		if e != nil {
			r.release()
			panic(e)
		}
		return keepBG(bg)
	}
	add := func(pl *wgpu.ComputePipeline, bg *wgpu.BindGroup, gx, gy uint32) {
		r.steps = append(r.steps, runStep{pl: pl, bg: bg, gx: gx, gy: gy})
	}

	// op builders (record a step against persistent buffers):
	rms := func(in, w *wgpu.Buffer) *wgpu.Buffer {
		out := storF(hidden)
		p := uni([]uint32{uint32(hidden), f32bits(eps), boolU32(addOne), 0})
		add(c.rmsnormPipeline, bind(c.rmsnormLayout, in, w, out, p), 64, 1)
		return out
	}
	quant := func(in *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer) {
		kp := padK(K)
		q, s := storF(kp/4), storF(1)
		p := uni([]uint32{1, uint32(K), uint32(kp), 0})
		add(c.quantizePipeline, bind(c.quantizeLayout, in, q, s, p), 1, 1)
		return q, s
	}
	gemv := func(aq, as *wgpu.Buffer, rm *ResidentW8A8) *wgpu.Buffer {
		out := storF(rm.rows)
		p := uni([]uint32{1, uint32(rm.kp), uint32(rm.rows), 0})
		gx, gy := gemvGrid(rm.rows)
		add(c.gemvPipeline, bind(c.gemvLayout, aq, rm.bq, as, rm.bScales, out, p), gx, gy)
		return out
	}
	rope := func(vec, invFreq *wgpu.Buffer, heads int) {
		half := hd / 2
		p := uni([]uint32{uint32(heads), uint32(hd), uint32(half), 0, f32bits(1), 0, 0, 0})
		r.posUnis = append(r.posUnis, posUni{buf: p, gen: func(pos int) []uint32 {
			return []uint32{uint32(heads), uint32(hd), uint32(half), uint32(pos), f32bits(1), 0, 0, 0}
		}})
		add(c.ropePipeline, bind(c.ropeLayout, vec, invFreq, p), uint32(heads*half+63)/64, 1)
	}
	residual := func(x, y *wgpu.Buffer) {
		p := uni([]uint32{uint32(hidden), 0, 0, 0})
		add(c.residualPipeline, bind(c.residualLayout, x, y, p), uint32(hidden+63)/64, 1)
	}

	r.xd = keepBuf(func() *wgpu.Buffer {
		b, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(hidden * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc})
		return b
	}())

	for i := range m.Layers {
		lw := &m.Layers[i]
		xn := rms(r.xd, lw.Attn.Norm.buf)
		aq, as := quant(xn, hidden)
		q, k, v := gemv(aq, as, lw.Attn.QProj), gemv(aq, as, lw.Attn.KProj), gemv(aq, as, lw.Attn.VProj)
		rope(q, lw.Attn.InvFreq.buf, nH)
		rope(k, lw.Attn.InvFreq.buf, nKV)
		r.steps = append(r.steps, runStep{src: k, cache: lw.Attn.KCache.buf}, runStep{src: v, cache: lw.Attn.VCache.buf})
		ctxv := storF(nH * hd)
		ap := uni([]uint32{uint32(nH), uint32(nKV), uint32(hd), 0, uint32(start), uint32(nH / nKV), f32bits(scale), 0})
		r.posUnis = append(r.posUnis, posUni{buf: ap, gen: func(pos int) []uint32 {
			return []uint32{uint32(nH), uint32(nKV), uint32(hd), uint32(pos + 1), uint32(start), uint32(nH / nKV), f32bits(scale), 0}
		}})
		add(c.attnPipeline, bind(c.attnLayout, q, lw.Attn.KCache.buf, lw.Attn.VCache.buf, ctxv, ap), uint32(nH), 1)
		cq, cs := quant(ctxv, nH*hd)
		residual(r.xd, gemv(cq, cs, lw.Attn.OProj))
		xn2 := rms(r.xd, lw.MLPNorm.buf)
		mq, ms := quant(xn2, hidden)
		gate, up := gemv(mq, ms, lw.Gate), gemv(mq, ms, lw.Up)
		mid := storF(inter)
		sp := uni([]uint32{uint32(inter), 0, 0, 0})
		add(c.swigluPipeline, bind(c.swigluLayout, gate, up, mid, sp), uint32(inter+63)/64, 1)
		dq, ds := quant(mid, inter)
		residual(r.xd, gemv(dq, ds, lw.Down))
	}
	xnf := rms(r.xd, m.FinalNorm.buf)
	fq, fs := quant(xnf, hidden)
	logits := gemv(fq, fs, m.LMHead)
	r.lastLogits = logits
	stag, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(r.vocab * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	r.stag = keepBuf(stag)
	return r, nil
}

// Run executes the plan for one token at absolute position pos. x is the token's
// input embedding [hidden]; returns the logits [vocab]. One Submit + one Poll.
func (r *DecodeRunner) Run(x []float32, pos int) ([]float32, error) {
	c := r.c
	if err := c.queue.WriteBuffer(r.xd, 0, wgpu.ToBytes(x)); err != nil {
		return nil, err
	}
	for _, pu := range r.posUnis {
		if err := c.queue.WriteBuffer(pu.buf, 0, wgpu.ToBytes(pu.gen(pos))); err != nil {
			return nil, err
		}
	}
	enc, err := c.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	defer enc.Release()
	for _, s := range r.steps {
		if s.pl == nil { // KV-append copy
			enc.CopyBufferToBuffer(s.src, 0, s.cache, uint64(pos*r.kvDim*4), uint64(r.kvDim*4))
			continue
		}
		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(s.pl)
		pass.SetBindGroup(0, s.bg, nil)
		pass.DispatchWorkgroups(s.gx, s.gy, 1)
		pass.End()
		pass.Release()
	}
	enc.CopyBufferToBuffer(r.lastLogits, 0, r.stag, 0, uint64(r.vocab*4))
	cmd, err := enc.Finish(nil)
	if err != nil {
		return nil, err
	}
	defer cmd.Release()
	c.queue.Submit(cmd)
	st := wgpu.BufferMapAsyncStatusUnknown
	if err := r.stag.MapAsync(wgpu.MapModeRead, 0, uint64(r.vocab*4), func(s wgpu.BufferMapAsyncStatus) { st = s }); err != nil {
		return nil, err
	}
	c.device.Poll(true, nil)
	if st != wgpu.BufferMapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: DecodeRunner map failed: %v", st)
	}
	out := make([]float32, r.vocab)
	copy(out, wgpu.FromBytes[float32](r.stag.GetMappedRange(0, uint(r.vocab*4))))
	r.stag.Unmap()
	return out, nil
}

func (r *DecodeRunner) release() {
	for _, f := range r.keep {
		f()
	}
	r.keep = nil
}

// Release frees the runner's scratch (not the resident model).
func (r *DecodeRunner) Release() { r.release() }
