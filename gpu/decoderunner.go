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

// runLayer / runModel are the DecodeRunner's precision-agnostic view of a resident
// model: the f32 buffers (norms, RoPE freqs, KV caches) plus the projection
// weights as decodeWeight (W8A8 or W4A8). The public constructors adapt a concrete
// ModelW / ModelW4 into this; the builder below works the same for either.
type runLayer struct {
	attnNorm, invFreq, kCache, vCache, mlpNorm *wgpu.Buffer
	kScale, vScale                             *wgpu.Buffer // int8-KV per-(pos,head) scales; nil unless kvI8
	q, k, v, o, gate, up, down                 decodeWeight
	qBias, kBias, vBias                        *wgpu.Buffer // optional (Qwen2); nil ⇒ no bias
}

type runModel struct {
	layers    []runLayer
	finalNorm *wgpu.Buffer
	lmHead    decodeWeight
	kvF16     bool // KCache/VCache are f16-packed (NewKVCacheF16) → use the f16 kernels
	kvI8      bool // KCache/VCache are int8-packed (NewKVCacheI8) + scales → int8 kernels
}

// w8Model adapts the W8A8 ModelW into the precision-agnostic runModel.
func w8Model(m ModelW) runModel {
	rm := runModel{finalNorm: m.FinalNorm.buf, lmHead: m.LMHead}
	for i := range m.Layers {
		lw := &m.Layers[i]
		rm.layers = append(rm.layers, runLayer{
			attnNorm: lw.Attn.Norm.buf, invFreq: lw.Attn.InvFreq.buf,
			kCache: lw.Attn.KCache.buf, vCache: lw.Attn.VCache.buf, mlpNorm: lw.MLPNorm.buf,
			q: lw.Attn.QProj, k: lw.Attn.KProj, v: lw.Attn.VProj, o: lw.Attn.OProj,
			gate: lw.Gate, up: lw.Up, down: lw.Down,
		})
	}
	return rm
}

// NewDecodeRunner builds the persistent plan for a resident W8A8 model.
func (c *Context) NewDecodeRunner(m ModelW, hidden, nH, nKV, hd, inter, start int, eps, scale float32, addOne bool) (*DecodeRunner, error) {
	return c.newDecodeRunner(w8Model(m), hidden, nH, nKV, hd, inter, start, eps, scale, addOne)
}

// newDecodeRunner builds the persistent decode plan for either precision.
func (c *Context) newDecodeRunner(m runModel, hidden, nH, nKV, hd, inter, start int, eps, scale float32, addOne bool) (*DecodeRunner, error) {
	for _, e := range []func() error{c.ensureGEMV, c.ensureQuantize, c.ensureLayer, c.ensureAttn, c.ensureFuse, c.ensureGEMVW4} {
		if err := e(); err != nil {
			return nil, err
		}
	}
	r := &DecodeRunner{c: c, vocab: m.lmHead.nRows(), kvDim: nKV * hd}
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
	// gemv records a projection matmul against any resident precision (W8A8 or
	// W4A8 — both expose the same 6-binding gemv + addResidual via decodeWeight).
	gemv := func(aq, as *wgpu.Buffer, w decodeWeight) *wgpu.Buffer {
		out := storF(w.nRows())
		p := uni([]uint32{1, uint32(w.kPad()), uint32(w.nRows()), 0})
		gx, gy := gemvGrid(w.nRows())
		add(w.gPipe(c), bind(w.gLayout(c), aq, w.wbuf(), as, w.sbuf(), out, p), gx, gy)
		return out
	}
	// gemvAdd is gemv with the residual fused into the epilogue: dst (the running
	// hidden state) gets dst[n] += result, deleting a standalone residual link.
	gemvAdd := func(aq, as *wgpu.Buffer, w decodeWeight, dst *wgpu.Buffer) {
		p := uni([]uint32{1, uint32(w.kPad()), uint32(w.nRows()), 1})
		gx, gy := gemvGrid(w.nRows())
		add(w.gPipe(c), bind(w.gLayout(c), aq, w.wbuf(), as, w.sbuf(), dst, p), gx, gy)
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
	// slot 6 carries nKV for the int8 ropeStore (it indexes scales[pos*nKV+head]);
	// the f32/f16 ropeStore ignore it (it's their unused _b pad), so it rides along.
	ropeKUni := uni([]uint32{uint32(nKV), uint32(hd), uint32(half), 0, f32bits(1), 0, uint32(nKV), 0})
	r.posUnis = append(r.posUnis, posUni{buf: ropeKUni, gen: func(pos int) []uint32 {
		return []uint32{uint32(nKV), uint32(hd), uint32(half), uint32(pos), f32bits(1), uint32(pos * r.kvDim), uint32(nKV), 0}
	}})
	vStoreUni := uni([]uint32{uint32(r.kvDim), 0, 0, 0})
	r.posUnis = append(r.posUnis, posUni{buf: vStoreUni, gen: func(pos int) []uint32 {
		return []uint32{uint32(r.kvDim), uint32(pos * r.kvDim), 0, 0}
	}})
	// int8 V store needs its own (differently-laid-out) per-token uniform:
	// {heads=nKV, headDim=hd, base=pos*kvDim, pos, nKV}. Only allocated for kvI8.
	var vStoreI8Uni *wgpu.Buffer
	if m.kvI8 {
		vStoreI8Uni = uni([]uint32{uint32(nKV), uint32(hd), 0, 0, uint32(nKV), 0, 0, 0})
		r.posUnis = append(r.posUnis, posUni{buf: vStoreI8Uni, gen: func(pos int) []uint32 {
			return []uint32{uint32(nKV), uint32(hd), uint32(pos * r.kvDim), uint32(pos), uint32(nKV), 0, 0, 0}
		}})
	}
	attnUni := uni([]uint32{uint32(nH), uint32(nKV), uint32(hd), 0, uint32(start), uint32(nH / nKV), f32bits(scale), 0})
	r.posUnis = append(r.posUnis, posUni{buf: attnUni, gen: func(pos int) []uint32 {
		return []uint32{uint32(nH), uint32(nKV), uint32(hd), uint32(pos + 1), uint32(start), uint32(nH / nKV), f32bits(scale), 0}
	}})
	rope := func(vec, invFreq *wgpu.Buffer) {
		add(c.ropePipeline, bind(c.ropeLayout, vec, invFreq, ropeQUni), uint32(nH*half+63)/64, 1)
	}
	// ropeStore rotates src (the K projection) and writes it straight into the KV
	// cache at pos*kvDim — replacing the K CopyBufferToBuffer append so the token
	// stays one compute pass. base rides the shared ropeKUni. The f16 variant packs
	// 2 rotated elems/word (one thread per word = nKV*half, same dispatch count).
	ropeStore := func(src, invFreq, cache, scale *wgpu.Buffer) {
		if m.kvI8 {
			// one thread per KV head: per-head absmax → scale → quantize + pack 4/word.
			add(c.ropeStoreI8Pipeline, bind(c.ropeStoreI8Layout, src, invFreq, cache, scale, ropeKUni), uint32(nKV+63)/64, 1)
			return
		}
		pl, ly := c.ropeStorePipeline, c.ropeStoreLayout
		if m.kvF16 {
			pl, ly = c.ropeStoreF16Pipeline, c.ropeStoreF16Layout
		}
		add(pl, bind(ly, src, invFreq, cache, ropeKUni), uint32(nKV*half+63)/64, 1)
	}
	// vStore copies src (the V projection) into the V cache at pos*kvDim. The f16
	// variant packs 2 elems/word, so it dispatches half as many threads (one/word);
	// the int8 variant is one thread per KV head (per-head absmax → scale → pack).
	vStore := func(src, cache, scale *wgpu.Buffer) {
		if m.kvI8 {
			add(c.kvStoreI8Pipeline, bind(c.kvStoreI8Layout, src, cache, scale, vStoreI8Uni), uint32(nKV+63)/64, 1)
			return
		}
		if m.kvF16 {
			words := r.kvDim / 2
			add(c.kvStoreF16Pipeline, bind(c.kvStoreF16Layout, src, cache, vStoreUni), uint32(words+63)/64, 1)
			return
		}
		add(c.kvStorePipeline, bind(c.kvStoreLayout, src, cache, vStoreUni), uint32(r.kvDim+63)/64, 1)
	}
	// biasAdd adds a per-output bias into a projection result (Qwen2 q/k/v bias),
	// reusing the residual kernel (vec[i] += bias[i]); n is the projection width.
	biasAdd := func(vec, bias *wgpu.Buffer, n int) {
		p := uni([]uint32{uint32(n), 0, 0, 0})
		add(c.residualPipeline, bind(c.residualLayout, vec, bias, p), uint32(n+63)/64, 1)
	}

	r.xd = keepBuf(func() *wgpu.Buffer {
		b, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(hidden * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc})
		return b
	}())

	for i := range m.layers {
		lw := &m.layers[i]
		aq, as := rmsQuant(r.xd, lw.attnNorm, hidden)
		q, k, v := gemv(aq, as, lw.q), gemv(aq, as, lw.k), gemv(aq, as, lw.v)
		if lw.qBias != nil { // Qwen2 q/k/v bias, added before RoPE (matches CPU attention)
			biasAdd(q, lw.qBias, nH*hd)
			biasAdd(k, lw.kBias, r.kvDim)
			biasAdd(v, lw.vBias, r.kvDim)
		}
		rope(q, lw.invFreq)
		ropeStore(k, lw.invFreq, lw.kCache, lw.kScale) // rotate K + append into cache
		vStore(v, lw.vCache, lw.vScale)                // append V into cache
		ctxv := storF(nH * hd)
		if m.kvI8 {
			// attnI8 reads packed int8 K/V + the per-(pos,head) scale side buffers.
			add(c.attnI8Pipeline, bind(c.attnI8Layout, q, lw.kCache, lw.vCache, lw.kScale, lw.vScale, ctxv, attnUni), uint32(nH), 1)
		} else {
			attnPl, attnLy := c.attnPipeline, c.attnLayout
			if m.kvF16 {
				attnPl, attnLy = c.attnF16Pipeline, c.attnF16Layout
			}
			add(attnPl, bind(attnLy, q, lw.kCache, lw.vCache, ctxv, attnUni), uint32(nH), 1)
		}
		cq, cs := quant(ctxv, nH*hd)
		gemvAdd(cq, cs, lw.o, r.xd) // o-proj + residual into xd
		mq, ms := rmsQuant(r.xd, lw.mlpNorm, hidden)
		gate, up := gemv(mq, ms, lw.gate), gemv(mq, ms, lw.up)
		dq, ds := swigluQuant(gate, up, inter)
		gemvAdd(dq, ds, lw.down, r.xd) // down-proj + residual into xd
	}
	fq, fs := rmsQuant(r.xd, m.finalNorm, hidden)
	logits := gemv(fq, fs, m.lmHead)
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
