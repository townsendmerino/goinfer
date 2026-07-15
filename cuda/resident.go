//go:build cuda

package cuda

import (
	"context"
	"fmt"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// cudaResident is the production resident decode runner: the parity-green cgo-free forward
// (TestRealForwardParity), promoted from the test harness. All CUDA state is owned by a
// single LockOSThread-pinned executor goroutine (guardrail #3); Forward routes one channel
// round-trip per token. Dense residency only (Qwen2/Llama, DecodeRunnerEligible), mixed
// int4/int8/f32 weights as the real q4_k_m checkpoint stores them.
var _ decoder.ResidentForward = (*cudaResident)(nil)

const cudaCtxCap = 4096 // resident KV capacity (positions); staged path handles longer.

// cudaWQ is a device projection weight in whatever precision the checkpoint stored it.
type cudaWQ struct {
	kind string
	W    *gc.Buffer[uint32]  // packed weights (int4 fast-layout nibbles, or int8x4)
	ws   *gc.Buffer[float32] // scales: int4 group (N*K/32) / int8 row (N)
	N, K int
}

type cudaLayer struct {
	q, k, v, o, g, u, d cudaWQ
	qb, kb, vb          *gc.Buffer[float32] // QKV bias (nil ⇒ none)
	preNorm, postNorm   *gc.Buffer[float32]
	hasBias             bool
}

type cudaResident struct {
	reqCh chan func() error
	ackCh chan error

	hidden, nLayers, nH, nKV, hd, inter, vocab int
	qDim, kvDim, half                          int
	eps, attnScale                             float32

	// gocudrv state — touched ONLY on the executor thread.
	cx                                                         *gc.Context
	stream                                                     *gc.Stream
	gemvW4, gemvW8, kvStore, fRms, fQ, fRope, fAttn, fSw, fRes *gc.Function

	layers          []cudaLayer
	lmW             cudaWQ
	finalNorm, invF *gc.Buffer[float32]

	// per-token scratch + KV caches (device).
	x, aSc, qB, kB, vB, cctx, cSc, oO, mSc, gO, uO, dSc, dScr, dO, logits *gc.Buffer[float32]
	aq, cq, mq, dq                                                        *gc.Buffer[int32]
	kc, vc                                                                []*gc.Buffer[float32]

	logitsHost []float32 // reused across Forward calls (decode consumes each before the next)
	setupErr   error     // first alloc/upload error during BuildResident's setup job
}

// alloc/upload helpers — called ONLY inside the setup job (r.cx current on the executor
// thread). They record the first failure into setupErr so BuildResident can decline
// gracefully (→ staged fallback) instead of proceeding with nil buffers.
func (r *cudaResident) af(n int) *gc.Buffer[float32] {
	b, e := gc.Alloc[float32](r.cx, n)
	if e != nil && r.setupErr == nil {
		r.setupErr = e
	}
	return b
}
func (r *cudaResident) ai(n int) *gc.Buffer[int32] {
	b, e := gc.Alloc[int32](r.cx, n)
	if e != nil && r.setupErr == nil {
		r.setupErr = e
	}
	return b
}
func (r *cudaResident) up32(v []float32) *gc.Buffer[float32] {
	b := r.af(len(v))
	if b != nil {
		_ = gc.CopyHtoD(context.Background(), b, v)
	}
	return b
}
func (r *cudaResident) upu32(v []uint32) *gc.Buffer[uint32] {
	b, e := gc.Alloc[uint32](r.cx, len(v))
	if e != nil && r.setupErr == nil {
		r.setupErr = e
	}
	if b != nil {
		_ = gc.CopyHtoD(context.Background(), b, v)
	}
	return b
}
func (r *cudaResident) upW(h hostW) cudaWQ {
	return cudaWQ{h.kind, r.upu32(h.wpk), r.up32(h.ws), h.N, h.K}
}

func (r *cudaResident) do(j func() error) error { r.reqCh <- j; return <-r.ackCh }

// Forward runs one token at absolute position pos and returns logits[vocab].
func (r *cudaResident) Forward(embedding []float32, pos int) ([]float32, error) {
	var out []float32
	err := r.do(func() error {
		o, e := r.step(embedding, pos)
		out = o
		return e
	})
	return out, err
}

// ForwardN runs K tokens at consecutive positions. Correctness-first: sequential steps in a
// single executor round-trip (bit-identical to K Forward calls; amortizes the channel hop).
func (r *cudaResident) ForwardN(embeddings [][]float32, startPos int) ([][]float32, error) {
	out := make([][]float32, len(embeddings))
	err := r.do(func() error {
		for i, emb := range embeddings {
			l, e := r.step(emb, startPos+i)
			if e != nil {
				return e
			}
			out[i] = append([]float32(nil), l...) // each row kept, so copy off the reused host buf
		}
		return nil
	})
	return out, err
}

// UploadKV writes a layer's post-RoPE K and raw V into the resident caches from position 0
// (prefill bridge, same packed layout the kernels read: [pos*kvDim + head*hd + d]).
func (r *cudaResident) UploadKV(layer int, keys, vals []float32) error {
	return r.do(func() error {
		bg := context.Background()
		if e := r.kc[layer].CopyFromAt(bg, 0, keys); e != nil {
			return e
		}
		return r.vc[layer].CopyFromAt(bg, 0, vals)
	})
}

// TruncateTo is a no-op: KV is positional and Forward sets nKeys=pos+1, so entries past pos
// are never read and get overwritten (matches the WebGPU path).
func (r *cudaResident) TruncateTo(pos int) {}

// Reset clears resident KV for a fresh generation (positions are overwritten on write).
func (r *cudaResident) Reset() {}

// Close shuts down the executor goroutine (and unpins its OS thread). Device buffers are
// freed by primary-context teardown at process exit; a per-buffer free is unnecessary for
// the single-model serve lifetime.
func (r *cudaResident) Close() error {
	if r.reqCh != nil {
		close(r.reqCh)
		r.reqCh = nil
	}
	return nil
}

// --- launch helpers (executor-thread only) ---

func g1cfg(n, b int) gc.LaunchConfig {
	return gc.LaunchConfig{GridX: uint32((n + b - 1) / b), GridY: 1, GridZ: 1, BlockX: uint32(b), BlockY: 1, BlockZ: 1}
}
func onecfg(b, sh int) gc.LaunchConfig {
	return gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: uint32(b), BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(sh)}
}

func (r *cudaResident) launch(f *gc.Function, cfg gc.LaunchConfig, args ...gc.KernelArg) error {
	return f.LaunchOn(context.Background(), r.stream, cfg, args...)
}

func (r *cudaResident) rms(src, nrm *gc.Buffer[float32], qOut *gc.Buffer[int32], sOut *gc.Buffer[float32]) error {
	return r.launch(r.fRms, onecfg(256, (r.hidden+256)*4),
		gc.Arg(src), gc.Arg(nrm), gc.ArgValue(int32(r.hidden)), gc.ArgValue(r.eps), gc.Arg(qOut), gc.Arg(sOut))
}

func (r *cudaResident) doG(wt cudaWQ, a *gc.Buffer[int32], as *gc.Buffer[float32], bias gc.KernelArg, dst *gc.Buffer[float32]) error {
	cfg := gc.LaunchConfig{GridX: uint32((wt.N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
	if wt.kind == "int4" {
		return r.launch(r.gemvW4, cfg, gc.Arg(wt.W), gc.Arg(a), gc.Arg(wt.ws), gc.Arg(as), bias,
			gc.ArgValue(int32(wt.N)), gc.ArgValue(int32(wt.K/8)), gc.ArgValue(int32(wt.K/32)), gc.Arg(dst))
	}
	return r.launch(r.gemvW8, cfg, gc.Arg(wt.W), gc.Arg(a), gc.Arg(wt.ws), gc.Arg(as), bias,
		gc.ArgValue(int32(wt.N)), gc.ArgValue(int32(wt.K/4)), gc.Arg(dst))
}

// step is one token's forward on the executor thread; returns logits (reused host buffer).
func (r *cudaResident) step(emb []float32, pos int) ([]float32, error) {
	bg := context.Background()
	nullBias := gc.ArgDevicePtr(0)
	if e := gc.CopyHtoD(bg, r.x, emb); e != nil {
		return nil, e
	}
	for l := 0; l < r.nLayers; l++ {
		Ly := &r.layers[l]
		if e := r.rms(r.x, Ly.preNorm, r.aq, r.aSc); e != nil {
			return nil, e
		}
		qb, kb, vb := nullBias, nullBias, nullBias
		if Ly.hasBias {
			qb, kb, vb = gc.Arg(Ly.qb), gc.Arg(Ly.kb), gc.Arg(Ly.vb)
		}
		if e := r.doG(Ly.q, r.aq, r.aSc, qb, r.qB); e != nil {
			return nil, e
		}
		_ = r.doG(Ly.k, r.aq, r.aSc, kb, r.kB)
		_ = r.doG(Ly.v, r.aq, r.aSc, vb, r.vB)
		_ = r.launch(r.fRope, g1cfg(r.nH*r.half, 256), gc.Arg(r.qB), gc.Arg(r.invF), gc.ArgValue(int32(r.nH)), gc.ArgValue(int32(r.hd)), gc.ArgValue(int32(pos)))
		_ = r.launch(r.fRope, g1cfg(r.nKV*r.half, 64), gc.Arg(r.kB), gc.Arg(r.invF), gc.ArgValue(int32(r.nKV)), gc.ArgValue(int32(r.hd)), gc.ArgValue(int32(pos)))
		_ = r.launch(r.kvStore, g1cfg(r.kvDim, 128), gc.Arg(r.kB), gc.Arg(r.kc[l]), gc.ArgValue(int32(pos)), gc.ArgValue(int32(r.kvDim)))
		_ = r.launch(r.kvStore, g1cfg(r.kvDim, 128), gc.Arg(r.vB), gc.Arg(r.vc[l]), gc.ArgValue(int32(pos)), gc.ArgValue(int32(r.kvDim)))
		nKeys := pos + 1
		_ = r.launch(r.fAttn, gc.LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((nKeys + 128) * 4)},
			gc.Arg(r.qB), gc.Arg(r.kc[l]), gc.Arg(r.vc[l]), gc.ArgValue(int32(r.nH)), gc.ArgValue(int32(r.nKV)), gc.ArgValue(int32(r.hd)), gc.ArgValue(int32(nKeys)), gc.ArgValue(r.attnScale), gc.Arg(r.cctx))
		_ = r.launch(r.fQ, onecfg(256, 256*4), gc.Arg(r.cctx), gc.ArgValue(int32(r.qDim)), gc.Arg(r.cq), gc.Arg(r.cSc))
		_ = r.doG(Ly.o, r.cq, r.cSc, nullBias, r.oO)
		_ = r.launch(r.fRes, g1cfg(r.hidden, 256), gc.Arg(r.x), gc.Arg(r.oO), gc.ArgValue(int32(r.hidden)))
		_ = r.rms(r.x, Ly.postNorm, r.mq, r.mSc)
		_ = r.doG(Ly.g, r.mq, r.mSc, nullBias, r.gO)
		_ = r.doG(Ly.u, r.mq, r.mSc, nullBias, r.uO)
		_ = r.launch(r.fSw, onecfg(256, 256*4), gc.Arg(r.gO), gc.Arg(r.uO), gc.ArgValue(int32(r.inter)), gc.Arg(r.dq), gc.Arg(r.dSc), gc.Arg(r.dScr))
		_ = r.doG(Ly.d, r.dq, r.dSc, nullBias, r.dO)
		if e := r.launch(r.fRes, g1cfg(r.hidden, 256), gc.Arg(r.x), gc.Arg(r.dO), gc.ArgValue(int32(r.hidden))); e != nil {
			return nil, e
		}
	}
	if e := r.rms(r.x, r.finalNorm, r.aq, r.aSc); e != nil {
		return nil, e
	}
	if e := r.doG(r.lmW, r.aq, r.aSc, nullBias, r.logits); e != nil {
		return nil, e
	}
	if e := r.stream.Synchronize(bg); e != nil {
		return nil, e
	}
	if e := gc.CopyDtoH(bg, r.logitsHost, r.logits); e != nil {
		return nil, e
	}
	return r.logitsHost, nil
}

// --- host-side weight packing (CPU; runs before any CUDA) ---

type hostW struct {
	kind string
	wpk  []uint32
	ws   []float32
	N, K int
}

func packI8(q8 []int8, N, K int) []uint32 {
	p := make([]uint32, N*(K/4))
	for i := range p {
		p[i] = uint32(uint8(q8[i*4])) | uint32(uint8(q8[i*4+1]))<<8 | uint32(uint8(q8[i*4+2]))<<16 | uint32(uint8(q8[i*4+3]))<<24
	}
	return p
}

// packWeight quantizes/repacks a projection into the resident device layout, or errors
// (so BuildResident can decline gracefully → staged fallback) for shapes it can't handle.
func packWeight(w *linalg.WeightMat) (hostW, error) {
	N, K := w.Rows(), w.Cols()
	if K%4 != 0 {
		return hostW{}, fmt.Errorf("cuda: K=%d not a multiple of 4", K)
	}
	switch w.Kind() {
	case "int4":
		if K%32 != 0 {
			return hostW{}, fmt.Errorf("cuda: int4 K=%d not a multiple of 32", K)
		}
		q4, sc, _, _ := w.Int4()
		wpk := make([]uint32, N*(K/8))
		for i := range wpk {
			b := q4[i*4 : i*4+4]
			wpk[i] = permuteFast(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
		}
		return hostW{"int4", wpk, sc, N, K}, nil
	case "int8":
		q8, sc, _, _ := w.Int8()
		return hostW{"int8", packI8(q8, N, K), sc, N, K}, nil
	case "f32":
		f32, _ := w.F32()
		q8, sc := linalg.QuantizeRowsInt8(f32, N, K)
		return hostW{"int8", packI8(q8, N, K), sc, N, K}, nil
	default:
		return hostW{}, fmt.Errorf("cuda: unsupported projection kind %q", w.Kind())
	}
}
