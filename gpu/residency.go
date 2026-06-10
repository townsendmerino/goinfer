//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// GPU full-residency bridge: builds a resident DecodeRunner from a loaded
// decoder.Model (dense Qwen2/Llama) so decoder.Generate's per-token forward runs
// entirely on the device. The webgpuBackend satisfies decoder.ResidencyBackend;
// the decoder calls BuildResident when the arch is eligible, then routes the
// per-token forward through the returned ResidentForward (decoder/residency.go).

// projWeight matches the read-only accessors decoder.weightMat exposes, so the
// bridge can pull a projection's resident arrays without naming the unexported
// type. *decoder.weightMat (via &model.Weights().Layers[i].QProj) satisfies it.
type projWeight interface {
	Kind() string
	Rows() int
	Cols() int
	Int4() (q4 []byte, q4s []float32, group int, ok bool)
	Int8() (q8 []int8, scales []float32, ok bool)
}

// uploadProj uploads one projection to the device at its native precision,
// returning a decodeWeight the DecodeRunner can GEMV. int4 and int8 only (the
// .giw cases); f32 is unsupported here (caller falls back).
func (c *Context) uploadProj(w projWeight) (decodeWeight, error) {
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
		q8, scales, _ := w.Int8()
		return c.UploadW8A8(q8, scales, N, K)
	default:
		return nil, fmt.Errorf("gpu: residency unsupported projection precision %q", w.Kind())
	}
}

// residentDecoder is the gpu side of decoder.ResidentForward: a persistent
// DecodeRunner + the runModel (for KV upload), built once per model.
type residentDecoder struct {
	c      *Context
	runner *DecodeRunner
	rm     runModel
	keep   []func() // release the resident buffers (norms, biases, KV, projections)
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
	// bytes, so the same VRAM holds 32k (task-gpu-f16-kv.md).
	kvF16 := m.KVCacheF16()
	ctxCap := 16384
	if kvF16 {
		ctxCap = 32768
	}
	kvDim := nKV * hd

	rd := &residentDecoder{c: c}
	keepF := func(f func()) { rd.keep = append(rd.keep, f) }
	up32 := func(v []float32) (*wgpu.Buffer, error) {
		d, err := c.UploadF32(v)
		if err != nil {
			return nil, err
		}
		keepF(d.Release)
		return d.buf, nil
	}
	proj := func(pw projWeight) (decodeWeight, error) {
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
	headW := projWeight(&w.LMHead)
	if w.LMHead.Rows() == 0 {
		headW = &w.Embed
	}
	lmHead, err := proj(headW)
	if err != nil {
		return fail(err)
	}
	rd.rm = runModel{finalNorm: finalNorm, lmHead: lmHead}

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
		if kvF16 {
			kc, e1 = c.NewKVCacheF16(nil, ctxCap*kvDim)
			vc, e2 = c.NewKVCacheF16(nil, ctxCap*kvDim)
		} else {
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
		if rl.gate, e = proj(&lw.GateProj); e != nil {
			return fail(e)
		}
		if rl.up, e = proj(&lw.UpProj); e != nil {
			return fail(e)
		}
		if rl.down, e = proj(&lw.DownProj); e != nil {
			return fail(e)
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
		rd.rm.layers = append(rd.rm.layers, rl)
	}
	_ = vocab // logits length is lmHead.nRows()

	rd.rm.kvF16 = kvF16 // the runner picks the f16 attn/store kernels off this
	runner, err := c.newDecodeRunner(rd.rm, hidden, nH, nKV, hd, inter, 0, eps, m.AttnScale(), m.RMSAddOne())
	if err != nil {
		return fail(err)
	}
	rd.runner = runner
	return rd, true, nil
}

func (rd *residentDecoder) Forward(embedding []float32, pos int) ([]float32, error) {
	return rd.runner.Run(embedding, pos)
}

// UploadKV writes a layer's post-RoPE K and raw V (positions 0..n-1) into the
// resident caches — the prefill bridge. keys/vals are [n*kvDim] f32.
func (rd *residentDecoder) UploadKV(layer int, keys, vals []float32) error {
	if layer < 0 || layer >= len(rd.rm.layers) {
		return fmt.Errorf("gpu: UploadKV layer %d out of range", layer)
	}
	l := rd.rm.layers[layer]
	if err := rd.c.queue.WriteBuffer(l.kCache, 0, wgpu.ToBytes(keys)); err != nil {
		return err
	}
	return rd.c.queue.WriteBuffer(l.vCache, 0, wgpu.ToBytes(vals))
}

func (rd *residentDecoder) Reset() {} // positions overwritten on next decode; caller tracks pos

func (rd *residentDecoder) Close() error { rd.release(); return nil }

func (rd *residentDecoder) release() {
	if rd.runner != nil {
		rd.runner.Release()
		rd.runner = nil
	}
	for i := len(rd.keep) - 1; i >= 0; i-- {
		rd.keep[i]()
	}
	rd.keep = nil
}
