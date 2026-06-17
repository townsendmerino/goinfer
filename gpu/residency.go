//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// GPU full-residency bridge: builds a resident DecodeRunner from a loaded
// decoder.Model (dense Qwen2/Llama) so decoder.Generate's per-token forward runs
// entirely on the device. The webgpuBackend satisfies decoder.ResidencyBackend;
// the decoder calls BuildResident when the arch is eligible, then routes the
// per-token forward through the returned ResidentForward (decoder/residency.go).

// The decoder's projections are now linalg.WeightMat (the aikit consolidation), so
// the bridge pulls a projection's resident arrays via its exported accessors
// (Kind/Rows/Cols/Int4/Int8) directly — no goinfer-local interface needed.

// uploadProj uploads one projection to the device at its native precision,
// returning a decodeWeight the DecodeRunner can GEMV. int4 and int8 only (the
// .giw cases); f32 is unsupported here (caller falls back).
func (c *Context) uploadProj(w *linalg.WeightMat) (decodeWeight, error) {
	N, K := w.Rows(), w.Cols()
	switch w.Kind() {
	case "int4":
		q4, q4s, group, _ := w.Int4()
		if group != w4a8GroupSize {
			return nil, fmt.Errorf("gpu: residency int4 group %d != %d", group, w4a8GroupSize)
		}
		// decoder packs 2 nibbles/byte (elem k → byte k>>1, low nibble if even);
		// UploadW4A8 takes one nibble (0..15) per element and re-packs to the GPU
		// layout. Unpack here — values (and so nibble−8) are preserved.
		nib := make([]uint8, N*K)
		for r := 0; r < N; r++ {
			row := q4[r*((K+1)/2):]
			dst := nib[r*K : r*K+K]
			for k := 0; k < K; k++ {
				b := row[k>>1]
				if k&1 == 0 {
					dst[k] = b & 0x0F
				} else {
					dst[k] = b >> 4
				}
			}
		}
		return c.UploadW4A8(nib, q4s, N, K)
	case "int8":
		q8, scales, _, _ := w.Int8()
		return c.UploadW8A8(q8, scales, N, K)
	default:
		return nil, fmt.Errorf("gpu: residency unsupported projection precision %q", w.Kind())
	}
}

// residentDecoder is the gpu side of decoder.ResidentForward: a persistent
// DecodeRunner + the runModel (for KV upload), built once per model.
type residentDecoder struct {
	c       *Context
	runner  *DecodeRunner
	rm      runModel
	nKV, hd int      // KV grouping — UploadKV needs it to per-head quantize int8
	keep    []func() // release the resident buffers (norms, biases, KV, projections)

	// Batched verify (ForwardN): extra DecodeRunner instances sharing rm (the same
	// resident weights + KV caches) with their own scratch/uniforms, built lazily up
	// to the largest K seen. batch[0] aliases runner. newRunner builds one more.
	newRunner func() (*DecodeRunner, error)
	batch     []*DecodeRunner
}

// BuildResident builds a resident DecodeRunner from m, or (nil,false,nil) when
// the arch is ineligible / a projection is f32 (caller uses the staged path).
func (b *webgpuBackend) BuildResident(m *decoder.Model) (decoder.ResidentForward, bool, error) {
	if !m.DecodeRunnerEligible() {
		return nil, false, nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	c := b.ctx
	w := m.Weights()
	hidden, _, nH, nKV, hd, inter, vocab := m.Dims() // arch-backed (Cfg may be zero for GGUF/.giw)
	eps := m.NormEps()
	// f32 KV caps context at 16k (the proven 8 GB fit); f16 halves per-token KV
	// bytes → 32k (task-gpu-f16-kv.md); int8 quarters them → ~64k (task-gpu-kv-i8.md).
	kvF16 := m.KVCacheF16()
	kvI8 := m.KVCacheI8()
	ctxCap := 16384
	switch {
	case kvI8:
		ctxCap = 65536
	case kvF16:
		ctxCap = 32768
	}
	kvDim := nKV * hd

	rd := &residentDecoder{c: c, nKV: nKV, hd: hd}
	keepF := func(f func()) { rd.keep = append(rd.keep, f) }
	up32 := func(v []float32) (*wgpu.Buffer, error) {
		d, err := c.UploadF32(v)
		if err != nil {
			return nil, err
		}
		keepF(d.Release)
		return d.buf, nil
	}
	proj := func(pw *linalg.WeightMat) (decodeWeight, error) {
		dw, err := c.uploadProj(pw)
		if err != nil {
			return nil, err
		}
		switch t := dw.(type) {
		case *ResidentW4A8:
			keepF(t.Release)
		case *ResidentW8A8:
			keepF(t.Release)
		}
		return dw, nil
	}

	fail := func(err error) (decoder.ResidentForward, bool, error) { rd.release(); return nil, false, err }

	invD, err := up32(m.RopeInvFreq())
	if err != nil {
		return fail(err)
	}
	finalNorm, err := up32(w.FinalNorm)
	if err != nil {
		return fail(err)
	}
	// LM head: tied (LMHead empty → the Embed matrix is the head) or separate.
	headW := &w.LMHead
	if w.LMHead.Rows() == 0 {
		headW = &w.Embed
	}
	lmHead, err := proj(headW)
	if err != nil {
		return fail(err)
	}
	rd.rm = runModel{finalNorm: finalNorm, lmHead: lmHead}

	// MoE (Lever C3c): Mixtral-class models route to stacked int8 experts on-device.
	// moeOK gates the per-layer FFN build below; the params are model-level.
	nExp, topK, moeInter, shInter, sig, normTopK, shUngated, rScale, moeOK := m.MoEResidentParams()
	if moeOK {
		rd.rm.moe = &moeRunParams{
			nE: nExp, k: topK, inter: moeInter, sigmoid: sig, norm: normTopK, scale: float32(rScale),
			sharedInter: shInter, sharedUngated: shUngated,
		}
	}
	// buildStacked packs one projection (gate/up/down) across all nE experts into a
	// resident stacked int8 buffer the indexed expert GEMV reads. int8 only — int4
	// experts aren't stacked yet (returns an error → silent staged fallback).
	buildStacked := func(lw *decoder.LayerWeights, get func(e int) *linalg.WeightMat) (*ResidentStackedW8A8, error) {
		nE := len(lw.Experts)
		w0 := get(0)
		N, K := w0.Rows(), w0.Cols()
		q8 := make([][]int8, nE)
		sc := make([][]float32, nE)
		for e := 0; e < nE; e++ {
			w := get(e)
			if w.Kind() != "int8" {
				return nil, fmt.Errorf("gpu: MoE residency expert %d kind %q (int8 only)", e, w.Kind())
			}
			q, s, _, _ := w.Int8()
			q8[e], sc[e] = q, s
		}
		return c.UploadStackedExperts(q8, sc, nE, N, K)
	}

	for i := range w.Layers {
		lw := &w.Layers[i]
		rl := runLayer{}
		var e error
		if rl.attnNorm, e = up32(lw.PreAttnNorm); e != nil {
			return fail(e)
		}
		if rl.mlpNorm, e = up32(lw.PreMLPNorm); e != nil {
			return fail(e)
		}
		rl.invFreq = invD
		var kc, vc *DeviceBuffer
		var e1, e2 error
		switch {
		case kvI8:
			var ks, vs *DeviceBuffer
			kc, ks, e1 = c.NewKVCacheI8(nil, ctxCap*kvDim, nKV, hd)
			vc, vs, e2 = c.NewKVCacheI8(nil, ctxCap*kvDim, nKV, hd)
			if e1 == nil && e2 == nil {
				keepF(ks.Release)
				keepF(vs.Release)
				rl.kScale, rl.vScale = ks.buf, vs.buf
			}
		case kvF16:
			kc, e1 = c.NewKVCacheF16(nil, ctxCap*kvDim)
			vc, e2 = c.NewKVCacheF16(nil, ctxCap*kvDim)
		default:
			kc, e1 = c.NewKVCache(nil, ctxCap*kvDim)
			vc, e2 = c.NewKVCache(nil, ctxCap*kvDim)
		}
		if e1 != nil || e2 != nil {
			return fail(fmt.Errorf("gpu: residency KV alloc (layer %d): %v %v", i, e1, e2))
		}
		keepF(kc.Release)
		keepF(vc.Release)
		rl.kCache, rl.vCache = kc.buf, vc.buf
		if rl.q, e = proj(&lw.QProj); e != nil {
			return fail(e)
		}
		if rl.k, e = proj(&lw.KProj); e != nil {
			return fail(e)
		}
		if rl.v, e = proj(&lw.VProj); e != nil {
			return fail(e)
		}
		if rl.o, e = proj(&lw.OProj); e != nil {
			return fail(e)
		}
		if moeOK && len(lw.Experts) > 0 { // Mixtral-class sparse MoE FFN
			rl.isMoE = true
			if rl.router, e = proj(&lw.Router); e != nil {
				return fail(e)
			}
			if len(lw.RouterBias) > 0 {
				if rl.routerBias, e = up32(lw.RouterBias); e != nil {
					return fail(e)
				}
			}
			if rl.expGate, e = buildStacked(lw, func(x int) *linalg.WeightMat { return &lw.Experts[x].Gate }); e != nil {
				return fail(e)
			}
			keepF(rl.expGate.Release)
			if rl.expUp, e = buildStacked(lw, func(x int) *linalg.WeightMat { return &lw.Experts[x].Up }); e != nil {
				return fail(e)
			}
			keepF(rl.expUp.Release)
			if rl.expDown, e = buildStacked(lw, func(x int) *linalg.WeightMat { return &lw.Experts[x].Down }); e != nil {
				return fail(e)
			}
			keepF(rl.expDown.Release)
			if shInter > 0 { // always-on shared expert (qwen2_moe gated / GLM ungated)
				if rl.shGate, e = proj(&lw.SharedExpert.Gate); e != nil {
					return fail(e)
				}
				if rl.shUp, e = proj(&lw.SharedExpert.Up); e != nil {
					return fail(e)
				}
				if rl.shDown, e = proj(&lw.SharedExpert.Down); e != nil {
					return fail(e)
				}
				if !shUngated { // qwen2_moe: the [1,hidden] sigmoid gate
					if rl.shGateW, e = proj(&lw.SharedGate); e != nil {
						return fail(e)
					}
				}
			}
		} else {
			if rl.gate, e = proj(&lw.GateProj); e != nil {
				return fail(e)
			}
			if rl.up, e = proj(&lw.UpProj); e != nil {
				return fail(e)
			}
			if rl.down, e = proj(&lw.DownProj); e != nil {
				return fail(e)
			}
		}
		if len(lw.QBias) > 0 { // Qwen2 q/k/v bias
			if rl.qBias, e = up32(lw.QBias); e != nil {
				return fail(e)
			}
			if rl.kBias, e = up32(lw.KBias); e != nil {
				return fail(e)
			}
			if rl.vBias, e = up32(lw.VBias); e != nil {
				return fail(e)
			}
		}
		if len(lw.QNorm) > 0 { // Qwen3/GLM per-head QK-norm (applied before RoPE)
			if rl.qNorm, e = up32(lw.QNorm); e != nil {
				return fail(e)
			}
			if rl.kNorm, e = up32(lw.KNorm); e != nil {
				return fail(e)
			}
		}
		rd.rm.layers = append(rd.rm.layers, rl)
	}
	_ = vocab // logits length is lmHead.nRows()

	rd.rm.kvF16 = kvF16 // the runner picks the f16 attn/store kernels off this
	rd.rm.kvI8 = kvI8   // …or the int8 kernels + scale binds
	scale, addOne := m.AttnScale(), m.RMSAddOne()
	rd.newRunner = func() (*DecodeRunner, error) {
		return c.newDecodeRunner(rd.rm, hidden, nH, nKV, hd, inter, 0, eps, scale, addOne)
	}
	runner, err := rd.newRunner()
	if err != nil {
		return fail(err)
	}
	rd.runner = runner
	return rd, true, nil
}

func (rd *residentDecoder) Forward(embedding []float32, pos int) ([]float32, error) {
	return rd.runner.Run(embedding, pos)
}

// ForwardN runs K tokens at startPos..startPos+K-1 in one command buffer. It lazily
// grows a pool of K DecodeRunners that share rd.rm (the resident weights + KV caches)
// but own their scratch/uniforms, then records all K into a single submit (runBatch).
// Causal across the shared KV: row i's kv-store precedes row i+1's attention read.
func (rd *residentDecoder) ForwardN(embeddings [][]float32, startPos int) ([][]float32, error) {
	n := len(embeddings)
	if n == 0 {
		return nil, nil
	}
	if len(rd.batch) == 0 {
		rd.batch = append(rd.batch, rd.runner) // batch[0] aliases the M=1 runner
	}
	for len(rd.batch) < n {
		r, err := rd.newRunner()
		if err != nil {
			return nil, err
		}
		rd.batch = append(rd.batch, r)
	}
	return runBatch(rd.c, rd.batch, embeddings, startPos)
}

// TruncateTo is a no-op on the resident cache: it is positional and Forward sets
// nKeys=pos+1, so entries past pos are never read and get overwritten next round
// (see the ResidentForward.TruncateTo contract).
func (rd *residentDecoder) TruncateTo(pos int) {}

// UploadKV writes a layer's post-RoPE K and raw V (positions 0..n-1) into the
// resident caches — the prefill bridge. keys/vals are [n*kvDim] f32, packed to the
// cache's precision (f32 raw, f16 via packF16Pairs, int8 per-head + scales) so the
// upload matches what an on-device decode would have written. Currently unused (the
// stateless prefill seeds the caches via sequential Forward); kept for prefix-reuse.
func (rd *residentDecoder) UploadKV(layer int, keys, vals []float32) error {
	if layer < 0 || layer >= len(rd.rm.layers) {
		return fmt.Errorf("gpu: UploadKV layer %d out of range", layer)
	}
	l := rd.rm.layers[layer]
	switch {
	case rd.rm.kvI8:
		kw, ks := packKVInt8(keys, rd.nKV, rd.hd)
		vw, vs := packKVInt8(vals, rd.nKV, rd.hd)
		if err := rd.c.queue.WriteBuffer(l.kCache, 0, wgpu.ToBytes(kw)); err != nil {
			return err
		}
		if err := rd.c.queue.WriteBuffer(l.kScale, 0, wgpu.ToBytes(ks)); err != nil {
			return err
		}
		if err := rd.c.queue.WriteBuffer(l.vCache, 0, wgpu.ToBytes(vw)); err != nil {
			return err
		}
		return rd.c.queue.WriteBuffer(l.vScale, 0, wgpu.ToBytes(vs))
	case rd.rm.kvF16:
		if err := rd.c.queue.WriteBuffer(l.kCache, 0, wgpu.ToBytes(packF16Pairs(keys))); err != nil {
			return err
		}
		return rd.c.queue.WriteBuffer(l.vCache, 0, wgpu.ToBytes(packF16Pairs(vals)))
	default:
		if err := rd.c.queue.WriteBuffer(l.kCache, 0, wgpu.ToBytes(keys)); err != nil {
			return err
		}
		return rd.c.queue.WriteBuffer(l.vCache, 0, wgpu.ToBytes(vals))
	}
}

func (rd *residentDecoder) Reset() {} // positions overwritten on next decode; caller tracks pos

func (rd *residentDecoder) Close() error { rd.release(); return nil }

func (rd *residentDecoder) release() {
	// batch[0] aliases rd.runner; release the extra verify runners (batch[1:]) — they
	// own scratch but share rm's weights/KV (freed via rd.keep below).
	for i := 1; i < len(rd.batch); i++ {
		rd.batch[i].Release()
	}
	rd.batch = nil
	if rd.runner != nil {
		rd.runner.Release()
		rd.runner = nil
	}
	for i := len(rd.keep) - 1; i >= 0; i-- {
		rd.keep[i]()
	}
	rd.keep = nil
}
