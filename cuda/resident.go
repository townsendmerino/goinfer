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
var (
	_ decoder.ResidentForward = (*cudaResident)(nil)
	_ decoder.ResidentGreedy  = (*cudaResident)(nil)
)

const cudaCtxCap = 4096 // resident KV capacity (positions); staged path handles longer.

// cudaWQ is a device projection weight in whatever precision the checkpoint stored it.
type cudaWQ struct {
	kind string
	W    *gc.Buffer[uint32]  // packed weights (int4 fast-layout nibbles, or int8x4)
	ws   *gc.Buffer[float32] // int8 row scales (N)
	ws16 *gc.Buffer[uint16]  // int4 group scales as f16 (N*K/32) — f32 would be 20% of the
	//                          int4 byte stream; f16 halves that (decode is byte-bound).
	N, K int
}

type cudaLayer struct {
	q, k, v, o, g, u, d cudaWQ
	qb, kb, vb          *gc.Buffer[float32] // QKV bias (nil ⇒ none)
	qNorm, kNorm        *gc.Buffer[float32] // per-head QK-norm weights (nil ⇒ arch has none)
	window              int32               // sliding-window span for THIS layer; 0 = full causal
	preNorm, postNorm   *gc.Buffer[float32]
	// Gemma sandwich norms (nil unless NormSandwich4): applied to the SUBLAYER OUTPUT before
	// the residual add, not to a GEMV input.
	postAttnNorm, postMLPNorm *gc.Buffer[float32]
	invF                      *gc.Buffer[float32] // per-layer RoPE inv-freq (local vs global base)
	hasBias                   bool
}

type cudaResident struct {
	reqCh chan func() error
	ackCh chan error

	hidden, nLayers, nH, nKV, hd, inter, vocab int
	qDim, kvDim, half                          int
	eps, attnScale                             float32
	qkNorm                                     bool  // arch needs per-head Q/K RMSNorm before RoPE
	rmsAddOne                                  bool  // (1+w) offset — false for Qwen3/Llama
	act                                        int32 // gated MLP activation, decoder.ActKind (0=gelu-tanh, 1=silu)
	sandwich                                   bool  // Gemma 4-norm sandwich: extra post-attn / post-MLP norms

	// gocudrv state — touched ONLY on the executor thread.
	cx                                                                                                 *gc.Context
	stream                                                                                             *gc.Stream
	gemvW4, gemvW8, kvStore, ropeKV, fRms, fRmsF32, fQ, fRope, fAttn, fSw, fRes, fArg, fQKV, fGU, fQKN *gc.Function

	fuseQKV   bool // all of Q/K/V int4 ⇒ K1 super-kernel is usable
	layers    []cudaLayer
	lmW       cudaWQ
	finalNorm *gc.Buffer[float32]

	// per-token scratch + KV caches (device).
	x, aSc, qB, kB, vB, cctx, cSc, oO, mSc, gO, uO, dSc, dScr, dO, logits *gc.Buffer[float32]
	aq, cq, mq, dq, argIdx                                                *gc.Buffer[int32]
	argVal                                                                *gc.Buffer[float32]
	kc, vc                                                                []*gc.Buffer[float32]

	// logitsPinned is PAGE-LOCKED host memory for the per-token logits readback. A pageable
	// D2H of 594 KB measured only ~1.26 GB/s (it stages through a driver bounce buffer);
	// pinned memory DMAs straight out. Slice() is a zero-copy view, so Forward still returns
	// without an extra copy. Reused across calls (decode consumes each before the next).
	logitsPinned *gc.HostBuffer[float32]
	logitsHost   []float32 // zero-copy view of logitsPinned
	setupErr     error     // first alloc/upload error during BuildResident's setup job
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
func (r *cudaResident) upu16(v []uint16) *gc.Buffer[uint16] {
	b, e := gc.Alloc[uint16](r.cx, len(v))
	if e != nil && r.setupErr == nil {
		r.setupErr = e
	}
	if b != nil {
		_ = gc.CopyHtoD(context.Background(), b, v)
	}
	return b
}
func (r *cudaResident) upW(h hostW) cudaWQ {
	w := cudaWQ{kind: h.kind, W: r.upu32(h.wpk), N: h.N, K: h.K}
	if h.kind == "int4" {
		w.ws16 = r.upu16(h.ws16)
	} else {
		w.ws = r.up32(h.ws)
	}
	return w
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
	if r.logitsPinned != nil {
		_ = r.logitsPinned.Close() // page-locked host memory must be freed before the ctx
		r.logitsPinned, r.logitsHost = nil, nil
	}
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

// addOneArg is the (1+w) RMS selector as the kernels take it (Architecture.RMSAddOne).
// Gemma stores norm weights as deviations from 1.0; Llama/Qwen scale by w directly.
func (r *cudaResident) addOneArg() int32 {
	if r.rmsAddOne {
		return 1
	}
	return 0
}

func (r *cudaResident) rms(src, nrm *gc.Buffer[float32], qOut *gc.Buffer[int32], sOut *gc.Buffer[float32]) error {
	return r.launch(r.fRms, onecfg(256, (r.hidden+256)*4),
		gc.Arg(src), gc.Arg(nrm), gc.ArgValue(int32(r.hidden)), gc.ArgValue(r.eps),
		gc.ArgValue(r.addOneArg()), gc.Arg(qOut), gc.Arg(sOut))
}

// normF32 is Gemma's sandwich post-norm: a plain in-place RMSNorm of a SUBLAYER OUTPUT
// (no quant — it lands straight in the f32 residual stream). No-op when the arch has no
// sandwich norms, so non-Gemma families pay nothing.
func (r *cudaResident) normF32(x, w *gc.Buffer[float32]) error {
	if w == nil {
		return nil
	}
	return r.launch(r.fRmsF32, onecfg(256, 256*4),
		gc.Arg(x), gc.Arg(w), gc.ArgValue(int32(r.hidden)), gc.ArgValue(r.eps), gc.ArgValue(r.addOneArg()))
}

// doG launches the projection GEMV. accum=1 makes the epilogue do dst[n] += result, which
// absorbs the separate `residual` launch (bit-identical: same operands, same rounding, just
// no round-trip through a temp buffer). Only lane 0 of the row's warp touches dst[n], and the
// GEMV's input activation is never x, so accumulating straight into the residual stream is
// race-free.
func (r *cudaResident) doG(wt cudaWQ, a *gc.Buffer[int32], as *gc.Buffer[float32], bias gc.KernelArg, dst *gc.Buffer[float32], accum int32) error {
	cfg := gc.LaunchConfig{GridX: uint32((wt.N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
	if wt.kind == "int4" {
		return r.launch(r.gemvW4, cfg, gc.Arg(wt.W), gc.Arg(a), gc.Arg(wt.ws16), gc.Arg(as), bias,
			gc.ArgValue(int32(wt.N)), gc.ArgValue(int32(wt.K/8)), gc.ArgValue(int32(wt.K/32)), gc.Arg(dst), gc.ArgValue(accum))
	}
	return r.launch(r.gemvW8, cfg, gc.Arg(wt.W), gc.Arg(a), gc.Arg(wt.ws), gc.Arg(as), bias,
		gc.ArgValue(int32(wt.N)), gc.ArgValue(int32(wt.K/4)), gc.Arg(dst), gc.ArgValue(accum))
}

// launchToken issues one token's whole kernel chain, leaving logits[vocab] on the device.
func (r *cudaResident) launchToken(emb []float32, pos int) error {
	bg := context.Background()
	nullBias := gc.ArgDevicePtr(0)
	if e := gc.CopyHtoD(bg, r.x, emb); e != nil {
		return e
	}
	for l := 0; l < r.nLayers; l++ {
		Ly := &r.layers[l]
		qb, kb, vb := nullBias, nullBias, nullBias
		if Ly.hasBias {
			qb, kb, vb = gc.Arg(Ly.qb), gc.Arg(Ly.kb), gc.Arg(Ly.vb)
		}
		if r.fuseQKV {
			// K1: rmsnorm+quant redundantly per block + this block's Q/K/V rows — one
			// launch instead of four, and the GridX:1 rmsnorm disappears.
			nrows := r.qDim + 2*r.kvDim
			cfg := gc.LaunchConfig{GridX: uint32((nrows + 7) / 8), GridY: 1, GridZ: 1,
				BlockX: 256, BlockY: 1, BlockZ: 1,
				SharedMemBytes: uint32((r.hidden + 256 + r.hidden/4) * 4)}
			if e := r.launch(r.fQKV, cfg,
				gc.Arg(r.x), gc.Arg(Ly.preNorm), gc.ArgValue(int32(r.hidden)), gc.ArgValue(r.eps),
				gc.ArgValue(r.addOneArg()),
				gc.Arg(Ly.q.W), gc.Arg(Ly.q.ws16), qb,
				gc.Arg(Ly.k.W), gc.Arg(Ly.k.ws16), kb,
				gc.Arg(Ly.v.W), gc.Arg(Ly.v.ws16), vb,
				gc.ArgValue(int32(r.qDim)), gc.ArgValue(int32(r.kvDim)),
				gc.ArgValue(int32(r.hidden/8)), gc.ArgValue(int32(r.hidden/32)),
				gc.Arg(r.qB), gc.Arg(r.kB), gc.Arg(r.vB)); e != nil {
				return e
			}
		} else {
			if e := r.rms(r.x, Ly.preNorm, r.aq, r.aSc); e != nil {
				return e
			}
			if e := r.doG(Ly.q, r.aq, r.aSc, qb, r.qB, 0); e != nil {
				return e
			}
			_ = r.doG(Ly.k, r.aq, r.aSc, kb, r.kB, 0)
			_ = r.doG(Ly.v, r.aq, r.aSc, vb, r.vB, 0)
		}
		// QK-norm (Qwen3/GLM/Mellum): per-head RMSNorm of Q and K over head_dim, BEFORE RoPE
		// (decoder/attention.go:94-96). One block per head; only dispatched for archs that
		// need it, so plain-dense models pay nothing.
		if r.qkNorm {
			addOne := int32(0)
			if r.rmsAddOne {
				addOne = 1
			}
			if e := r.launch(r.fQKN, gc.LaunchConfig{GridX: uint32(r.nH + r.nKV), GridY: 1, GridZ: 1,
				BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 8}, // f64 reduction
				gc.Arg(r.qB), gc.Arg(r.kB), gc.Arg(Ly.qNorm), gc.Arg(Ly.kNorm),
				gc.ArgValue(int32(r.nH)), gc.ArgValue(int32(r.nKV)), gc.ArgValue(int32(r.hd)),
				gc.ArgValue(r.eps), gc.ArgValue(addOne)); e != nil {
				return e
			}
		}
		// fused rope(q)+rope(k)+kv_store(k)+kv_store(v): 4 launches → 1 (same math/order).
		_ = r.launch(r.ropeKV, g1cfg(r.nH*r.half+r.nKV*r.half, 256),
			gc.Arg(r.qB), gc.Arg(r.kB), gc.Arg(r.vB), gc.Arg(Ly.invF), gc.Arg(r.kc[l]), gc.Arg(r.vc[l]),
			gc.ArgValue(int32(r.nH)), gc.ArgValue(int32(r.nKV)), gc.ArgValue(int32(r.hd)), gc.ArgValue(int32(pos)))
		nKeys := pos + 1
		// Sliding window (per layer: Mistral is all-local, Mellum interleaves). Shared is sized
		// to the ATTENDED span, so a windowed layer's request stays bounded as context grows.
		nWin := nKeys
		if Ly.window > 0 && nKeys > int(Ly.window) {
			nWin = int(Ly.window)
		}
		_ = r.launch(r.fAttn, gc.LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((nWin + 128) * 4)},
			gc.Arg(r.qB), gc.Arg(r.kc[l]), gc.Arg(r.vc[l]), gc.ArgValue(int32(r.nH)), gc.ArgValue(int32(r.nKV)), gc.ArgValue(int32(r.hd)), gc.ArgValue(int32(nKeys)), gc.ArgValue(r.attnScale), gc.ArgValue(Ly.window), gc.Arg(r.cctx))
		_ = r.launch(r.fQ, onecfg(256, 256*4), gc.Arg(r.cctx), gc.ArgValue(int32(r.qDim)), gc.Arg(r.cq), gc.Arg(r.cSc))
		// Normally the out-proj accumulates straight into the residual stream (accum=1),
		// absorbing the `residual` launch. Gemma's sandwich norm CANNOT use that epilogue: it
		// must normalize the sublayer output BETWEEN the projection and the residual add
		// (decoder/model.go: `a = attn(...); if sandwich { a = rmsNorm(a, PostAttnNorm) }; h += a`).
		// So the sandwich path projects into a scratch buffer, norms it, then adds.
		if r.sandwich {
			// oO/dO are the pre-existing [hidden] sublayer-output buffers, dead since the
			// accum=1 epilogue absorbed the residual launch — the sandwich path needs exactly
			// them back.
			_ = r.doG(Ly.o, r.cq, r.cSc, nullBias, r.oO, 0)
			_ = r.normF32(r.oO, Ly.postAttnNorm)
			_ = r.launch(r.fRes, g1cfg(r.hidden, 256), gc.Arg(r.x), gc.Arg(r.oO), gc.ArgValue(int32(r.hidden)))
		} else {
			_ = r.doG(Ly.o, r.cq, r.cSc, nullBias, r.x, 1)
		}
		if r.fuseQKV { // same guard: all layer projections int4
			// K3a: pre-MLP rmsnorm+quant redundantly per block + this block's gate/up rows —
			// one launch instead of three; the layer's second GridX:1 rmsnorm disappears.
			cfg := gc.LaunchConfig{GridX: uint32((2*r.inter + 63) / 64), GridY: 1, GridZ: 1, // 64 rows/block
				BlockX: 256, BlockY: 1, BlockZ: 1,
				SharedMemBytes: uint32((r.hidden + 256 + r.hidden/4) * 4)}
			if e := r.launch(r.fGU, cfg,
				gc.Arg(r.x), gc.Arg(Ly.postNorm), gc.ArgValue(int32(r.hidden)), gc.ArgValue(r.eps),
				gc.ArgValue(r.addOneArg()),
				gc.Arg(Ly.g.W), gc.Arg(Ly.g.ws16),
				gc.Arg(Ly.u.W), gc.Arg(Ly.u.ws16),
				gc.ArgValue(int32(r.inter)), gc.ArgValue(int32(r.hidden/8)), gc.ArgValue(int32(r.hidden/32)),
				gc.Arg(r.gO), gc.Arg(r.uO)); e != nil {
				return e
			}
		} else {
			_ = r.rms(r.x, Ly.postNorm, r.mq, r.mSc)
			_ = r.doG(Ly.g, r.mq, r.mSc, nullBias, r.gO, 0)
			_ = r.doG(Ly.u, r.mq, r.mSc, nullBias, r.uO, 0)
		}
		_ = r.launch(r.fSw, onecfg(256, 256*4), gc.Arg(r.gO), gc.Arg(r.uO), gc.ArgValue(int32(r.inter)),
			gc.ArgValue(r.act), gc.Arg(r.dq), gc.Arg(r.dSc), gc.Arg(r.dScr))
		if r.sandwich {
			if e := r.doG(Ly.d, r.dq, r.dSc, nullBias, r.dO, 0); e != nil {
				return e
			}
			_ = r.normF32(r.dO, Ly.postMLPNorm)
			_ = r.launch(r.fRes, g1cfg(r.hidden, 256), gc.Arg(r.x), gc.Arg(r.dO), gc.ArgValue(int32(r.hidden)))
		} else if e := r.doG(Ly.d, r.dq, r.dSc, nullBias, r.x, 1); e != nil {
			return e
		}
	}
	if e := r.rms(r.x, r.finalNorm, r.aq, r.aSc); e != nil {
		return e
	}
	return r.doG(r.lmW, r.aq, r.aSc, nullBias, r.logits, 0)
}

// step returns full logits — the general contract (sampler / constrained decode / logprobs).
// Costs a vocab*4 B D2H every token (594 KB at a 151936 vocab).
func (r *cudaResident) step(emb []float32, pos int) ([]float32, error) {
	bg := context.Background()
	if e := r.launchToken(emb, pos); e != nil {
		return nil, e
	}
	if e := r.stream.Synchronize(bg); e != nil {
		return nil, e
	}
	if e := r.logits.CopyToHost(bg, r.logitsPinned); e != nil {
		return nil, e
	}
	return r.logitsHost, nil
}

// ForwardArgmax is the greedy fast path (decoder.ResidentGreedy): reduce the argmax on-device
// and read back 4 B instead of the whole logits vector. Same kernel chain, same numerics —
// only the readback differs, so the id equals argmax(Forward(...)) exactly.
func (r *cudaResident) ForwardArgmax(embedding []float32, pos int) (int, error) {
	var id int
	err := r.do(func() error {
		bg := context.Background()
		if e := r.launchToken(embedding, pos); e != nil {
			return e
		}
		if e := r.launch(r.fArg, onecfg(256, 256*4+256*4), gc.Arg(r.logits),
			gc.ArgValue(int32(r.vocab)), gc.Arg(r.argIdx), gc.Arg(r.argVal)); e != nil {
			return e
		}
		if e := r.stream.Synchronize(bg); e != nil {
			return e
		}
		out := make([]int32, 1)
		if e := gc.CopyDtoH(bg, out, r.argIdx); e != nil {
			return e
		}
		id = int(out[0])
		return nil
	})
	return id, err
}

// --- host-side weight packing (CPU; runs before any CUDA) ---

type hostW struct {
	kind string
	wpk  []uint32
	ws   []float32 // int8 row scales
	ws16 []uint16  // int4 group scales (f16)
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
		gs := make([]uint16, len(sc))
		for i, v := range sc {
			gs[i] = f32tof16(v)
		}
		return hostW{kind: "int4", wpk: wpk, ws16: gs, N: N, K: K}, nil
	case "int8":
		q8, sc, _, _ := w.Int8()
		return hostW{kind: "int8", wpk: packI8(q8, N, K), ws: sc, N: N, K: K}, nil
	case "f32":
		f32, _ := w.F32()
		q8, sc := linalg.QuantizeRowsInt8(f32, N, K)
		return hostW{kind: "int8", wpk: packI8(q8, N, K), ws: sc, N: N, K: K}, nil
	default:
		return hostW{}, fmt.Errorf("cuda: unsupported projection kind %q", w.Kind())
	}
}
