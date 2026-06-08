//go:build gpu

package gpu

import (
	"fmt"
	"time"

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

	// §5 instrumentation: wall time of each Run phase, overwritten per call.
	// Zero overhead when ignored; the decomposition test reads them.
	TWrite, TEncode, TSync time.Duration
}

type runStep struct {
	pl     *wgpu.ComputePipeline
	bg     *wgpu.BindGroup
	gx, gy uint32
}

type posUni struct {
	buf *wgpu.Buffer
	gen func(pos int) []uint32 // uniform contents for this pos
}

// NewDecodeRunner builds the persistent plan for a resident model.
func (c *Context) NewDecodeRunner(m ModelW, hidden, nH, nKV, hd, inter, start int, eps, scale float32, addOne bool) (*DecodeRunner, error) {
	for _, e := range []func() error{c.ensureGEMV, c.ensureQuantize, c.ensureLayer, c.ensureAttn, c.ensureFuse} {
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
	// rmsQuant fuses RMSNorm→quantize: one dispatch, no xn round-trip, one fewer
	// link on the serialized decode spine (§2). Bit-exact with rms→quant.
	rmsQuant := func(in, w *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer) {
		kp := padK(K)
		q, s := storF(kp/4), storF(1)
		p := uni([]uint32{uint32(K), f32bits(eps), boolU32(addOne), uint32(kp)})
		add(c.rmsQuantPipeline, bind(c.rmsQuantLayout, in, w, q, s, p), 1, 1)
		return q, s
	}
	// swigluQuant fuses SwiGLU→quantize: the inter-wide product never materializes
	// or crosses a barrier — one fewer link and the big buffer stays off the spine.
	swigluQuant := func(gate, up *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer) {
		kp := padK(K)
		q, s := storF(kp/4), storF(1)
		p := uni([]uint32{uint32(K), uint32(kp), 0, 0})
		add(c.swigluQuantPipeline, bind(c.swigluQuantLayout, gate, up, q, s, p), 1, 1)
		return q, s
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
	// gemvAdd is gemv with the residual fused into the epilogue: dst (the running
	// hidden state) gets dst[n] += result, deleting a standalone residual link.
	gemvAdd := func(aq, as *wgpu.Buffer, rm *ResidentW8A8, dst *wgpu.Buffer) {
		p := uni([]uint32{1, uint32(rm.kp), uint32(rm.rows), 1})
		gx, gy := gemvGrid(rm.rows)
		add(c.gemvPipeline, bind(c.gemvLayout, aq, rm.bq, as, rm.bScales, dst, p), gx, gy)
	}
	// §4: the per-token uniforms (rope-q, rope-store-k, v-store, attn) depend only
	// on pos, NOT on layer index — their contents are identical across all 28
	// layers. So allocate ONE buffer per type and let every layer's dispatch bind
	// it; Run then writes 4 small uniforms per token instead of ~112. The builders
	// below reference these shared buffers and no longer append per-call posUnis.
	half := hd / 2
	ropeQUni := uni([]uint32{uint32(nH), uint32(hd), uint32(half), 0, f32bits(1), 0, 0, 0})
	r.posUnis = append(r.posUnis, posUni{buf: ropeQUni, gen: func(pos int) []uint32 {
		return []uint32{uint32(nH), uint32(hd), uint32(half), uint32(pos), f32bits(1), 0, 0, 0}
	}})
	ropeKUni := uni([]uint32{uint32(nKV), uint32(hd), uint32(half), 0, f32bits(1), 0, 0, 0})
	r.posUnis = append(r.posUnis, posUni{buf: ropeKUni, gen: func(pos int) []uint32 {
		return []uint32{uint32(nKV), uint32(hd), uint32(half), uint32(pos), f32bits(1), uint32(pos * r.kvDim), 0, 0}
	}})
	vStoreUni := uni([]uint32{uint32(r.kvDim), 0, 0, 0})
	r.posUnis = append(r.posUnis, posUni{buf: vStoreUni, gen: func(pos int) []uint32 {
		return []uint32{uint32(r.kvDim), uint32(pos * r.kvDim), 0, 0}
	}})
	attnUni := uni([]uint32{uint32(nH), uint32(nKV), uint32(hd), 0, uint32(start), uint32(nH / nKV), f32bits(scale), 0})
	r.posUnis = append(r.posUnis, posUni{buf: attnUni, gen: func(pos int) []uint32 {
		return []uint32{uint32(nH), uint32(nKV), uint32(hd), uint32(pos + 1), uint32(start), uint32(nH / nKV), f32bits(scale), 0}
	}})
	rope := func(vec, invFreq *wgpu.Buffer) {
		add(c.ropePipeline, bind(c.ropeLayout, vec, invFreq, ropeQUni), uint32(nH*half+63)/64, 1)
	}
	// ropeStore rotates src (the K projection) and writes it straight into the KV
	// cache at pos*kvDim — replacing the K CopyBufferToBuffer append so the token
	// stays one compute pass. base rides the shared ropeKUni.
	ropeStore := func(src, invFreq, cache *wgpu.Buffer) {
		add(c.ropeStorePipeline, bind(c.ropeStoreLayout, src, invFreq, cache, ropeKUni), uint32(nKV*half+63)/64, 1)
	}
	// vStore copies src (the V projection) into the V cache at pos*kvDim.
	vStore := func(src, cache *wgpu.Buffer) {
		add(c.kvStorePipeline, bind(c.kvStoreLayout, src, cache, vStoreUni), uint32(r.kvDim+63)/64, 1)
	}

	r.xd = keepBuf(func() *wgpu.Buffer {
		b, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(hidden * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc})
		return b
	}())

	for i := range m.Layers {
		lw := &m.Layers[i]
		aq, as := rmsQuant(r.xd, lw.Attn.Norm.buf, hidden)
		q, k, v := gemv(aq, as, lw.Attn.QProj), gemv(aq, as, lw.Attn.KProj), gemv(aq, as, lw.Attn.VProj)
		rope(q, lw.Attn.InvFreq.buf)
		ropeStore(k, lw.Attn.InvFreq.buf, lw.Attn.KCache.buf) // rotate K + append into cache
		vStore(v, lw.Attn.VCache.buf)                         // append V into cache
		ctxv := storF(nH * hd)
		add(c.attnPipeline, bind(c.attnLayout, q, lw.Attn.KCache.buf, lw.Attn.VCache.buf, ctxv, attnUni), uint32(nH), 1)
		cq, cs := quant(ctxv, nH*hd)
		gemvAdd(cq, cs, lw.Attn.OProj, r.xd) // o-proj + residual into xd
		mq, ms := rmsQuant(r.xd, lw.MLPNorm.buf, hidden)
		gate, up := gemv(mq, ms, lw.Gate), gemv(mq, ms, lw.Up)
		dq, ds := swigluQuant(gate, up, inter)
		gemvAdd(dq, ds, lw.Down, r.xd) // down-proj + residual into xd
	}
	fq, fs := rmsQuant(r.xd, m.FinalNorm.buf, hidden)
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
	tw := time.Now()
	if err := c.queue.WriteBuffer(r.xd, 0, wgpu.ToBytes(x)); err != nil {
		return nil, err
	}
	for _, pu := range r.posUnis {
		if err := c.queue.WriteBuffer(pu.buf, 0, wgpu.ToBytes(pu.gen(pos))); err != nil {
			return nil, err
		}
	}
	r.TWrite = time.Since(tw)
	te := time.Now()
	enc, err := c.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	defer enc.Release()
	// One compute pass for the whole token: WebGPU runs the dispatches in record
	// order and the backend inserts the minimal storage-buffer barriers between
	// data-dependent dispatches. The KV appends are now compute kernels (rope-store
	// / kv-store), so nothing forces a pass break.
	pass := enc.BeginComputePass(nil)
	for _, s := range r.steps {
		pass.SetPipeline(s.pl)
		pass.SetBindGroup(0, s.bg, nil)
		pass.DispatchWorkgroups(s.gx, s.gy, 1)
	}
	pass.End()
	pass.Release()
	enc.CopyBufferToBuffer(r.lastLogits, 0, r.stag, 0, uint64(r.vocab*4))
	cmd, err := enc.Finish(nil)
	if err != nil {
		return nil, err
	}
	defer cmd.Release()
	r.TEncode = time.Since(te)
	ts := time.Now()
	c.queue.Submit(cmd)
	st := wgpu.BufferMapAsyncStatusUnknown
	if err := r.stag.MapAsync(wgpu.MapModeRead, 0, uint64(r.vocab*4), func(s wgpu.BufferMapAsyncStatus) { st = s }); err != nil {
		return nil, err
	}
	c.device.Poll(true, nil)
	r.TSync = time.Since(ts)
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
