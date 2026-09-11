//go:build gpu

package gpu

import (
	"fmt"
	"sync"

	"github.com/townsendmerino/aikit/encoder"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// init registers the WebGPU backend under the name "webgpu" with BOTH the
// goinfer decoder and the aikit encoder. Their Backend interfaces have the
// same method set, so one *webgpuBackend satisfies both — a `-tags gpu` build
// that blank-imports this package gains GPU matmul on either side without the
// core modules ever importing github.com/cogentcore/webgpu.
func init() {
	// Return a LITERAL nil interface on failure, not the (nil, err) *webgpuBackend
	// newWebGPUBackend hands back: a typed-nil pointer auto-converts to a NON-nil
	// Backend interface, which defeats the caller's "no device → fall back to CPU"
	// path — Model.withResidency would type-assert it and call BuildResident on a
	// nil receiver (panic). With a literal nil, Load keeps the CPU backend cleanly.
	decoder.RegisterBackend("webgpu", func() (decoder.Backend, error) {
		b, err := newWebGPUBackend("decoder")
		if err != nil {
			return nil, err
		}
		return b, nil
	})
	encoder.RegisterBackend("webgpu", func() (encoder.Backend, error) {
		b, err := newWebGPUBackend("encoder")
		if err != nil {
			return nil, err
		}
		return b, nil
	})
}

// webgpuBackend runs MatmulBT on a WebGPU adapter (Vulkan / Metal / D3D12) via
// the Context foundation in this package. Built only under -tags gpu.
//
// Resident weights: a model weight matrix is constant across every token, so
// the first MatmulBT for a given weight uploads it to a GPU storage buffer and
// caches the handle (keyed by the slice's backing pointer); every later call
// only uploads the (small) activation and reads the result back. This removes
// the catastrophic per-token re-upload of the weights (the LM head alone is
// the ~671 MB embedding). On any per-call GPU error it falls back to the CPU
// matmul, so results are always correct.
//
// Still naive beyond that: each matmul is its own synchronous dispatch +
// readback, so decode is latency-bound on per-matmul round-trips. Keeping the
// activations resident on-device across a layer's matmuls (and porting
// norms/rope/softmax to WGSL) is the remaining work for a GPU-fast forward.
type webgpuBackend struct {
	ctx  *Context
	name string

	mu         sync.Mutex // Context is not goroutine-safe
	resident   map[*float32]*ResidentMatrix
	qresident  map[*int8]*qResident  // W8A8 weights kept resident (+ a decode runner)
	q4resident map[*byte]*q4Resident // G6 (docs/task-gpu-paths-2026-09.md): W4A8 twin — the "staged int4" item
	fallbacks  int
}

// qResident is a resident int8 weight plus its cached decode (GEMV) runner.
type qResident struct {
	rm     *ResidentW8A8
	runner *GEMVRunner // lazily built on the first M=1 call
}

// q4Resident is qResident's int4 (W4A8) twin.
type q4Resident struct {
	rm     *ResidentW4A8
	runner *GEMVRunner
}

// newWebGPUBackend initializes a WebGPU context for the named consumer
// ("decoder"/"encoder", used only in the backend's reported Name). It returns
// a CPU-equivalent error (not a hard failure) when no adapter is present, so a
// `--backend webgpu` selection still runs on a headless machine.
func newWebGPUBackend(who string) (*webgpuBackend, error) {
	ctx, err := New()
	if err != nil {
		// No adapter (headless / no driver): surface as an error so the
		// caller's registry falls back to CPU with a note.
		return nil, fmt.Errorf("gpu: no WebGPU adapter for %s (%v)", who, err)
	}
	return &webgpuBackend{
		ctx:        ctx,
		name:       "webgpu:" + ctx.Backend(),
		resident:   make(map[*float32]*ResidentMatrix),
		qresident:  make(map[*int8]*qResident),
		q4resident: make(map[*byte]*q4Resident),
	}, nil
}

func (b *webgpuBackend) Name() string { return b.name }

// MatmulW8A8 runs the int8×int8 weight matmul on the GPU: the weight is uploaded
// once (resident, keyed by &bQ[0]); each call quantizes the activation and
// dispatches the coalesced GEMV (decode, M=1) or the tiled GEMM (prefill, M>1).
// Returns false on any GPU error so weightMat falls back to the CPU kernel.
func (b *webgpuBackend) MatmulW8A8(a []float32, bQ []int8, bScales []float32, dst []float32, M, K, N int) bool {
	if len(bQ) == 0 {
		return false
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	key := &bQ[0]
	qr := b.qresident[key]
	if qr == nil {
		rm, err := b.ctx.UploadW8A8(bQ, bScales, N, K)
		if err != nil {
			b.fallbacks++
			return false
		}
		qr = &qResident{rm: rm}
		b.qresident[key] = qr
	}
	aq, aScales := linalg.QuantizeRowsInt8(a, M, K)
	if M == 1 {
		if qr.runner == nil {
			r, err := b.ctx.NewGEMVRunner(qr.rm)
			if err != nil {
				b.fallbacks++
				return false
			}
			qr.runner = r
		}
		out, err := qr.runner.Run(aq, aScales[0])
		if err != nil {
			b.fallbacks++
			return false
		}
		copy(dst, out)
		return true
	}
	out, err := b.ctx.MatmulW8A8Tiled(aq, aScales, qr.rm, M)
	if err != nil {
		b.fallbacks++
		return false
	}
	copy(dst, out)
	return true
}

// MatmulW4A8 is MatmulW8A8's int4 (W4A8) twin — G6 (docs/task-gpu-paths-2026-09.md), the "staged
// int4" item: decoder/weightmat.go's matmulInto never consulted a backend for int4 before this
// (its int8 branch already did, via MatmulW8A8/QuantBackend), so an int4-quantized model on the
// STAGED (non-resident) path ran every projection on the CPU regardless of which backend was
// active — gpu/gemv_w4a8.go's kernel, upload paths and decodeWeight interface already existed,
// but only gpu/residency.go's RESIDENT uploadProj used them.
//
// M=1 only (decode): the M>1 (prefill/tiled) case has no int4 GEMM kernel on this backend yet —
// declines, so matmulInto's caller falls back to the CPU W4A8 kernel exactly as it did before
// this method existed. bQ4 is decoder's native on-disk packed layout (2 nibbles/byte); group is
// always w4a8GroupSize (32) for goinfer's models (matches uploadProj's own assumption).
func (b *webgpuBackend) MatmulW4A8(a []float32, bQ4 []byte, bScales []float32, group int, dst []float32, M, K, N int) bool {
	if len(bQ4) == 0 || M != 1 || group != w4a8GroupSize {
		return false
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	qr, ok := b.residentW4A8For(bQ4, bScales, N, K)
	if !ok {
		return false
	}
	aq, aScales := linalg.QuantizeRowsInt8(a, M, K)
	if qr.runner == nil {
		r, err := b.ctx.NewGEMVRunner(qr.rm)
		if err != nil {
			b.fallbacks++
			return false
		}
		qr.runner = r
	}
	out, err := qr.runner.Run(aq, aScales[0])
	if err != nil {
		b.fallbacks++
		return false
	}
	copy(dst, out)
	return true
}

// residentW4A8For resolves (or uploads and caches) the resident W4A8 weight for one packed
// int4 buffer, keyed by its backing pointer — shared by MatmulW4A8 and MatmulW4A8Batch (P-16,
// audit-2026-09-10) so the upload/unpack logic exists exactly once. Caller holds b.mu. ok=false
// means upload failed; the caller counts the fallback.
func (b *webgpuBackend) residentW4A8For(bQ4 []byte, bScales []float32, N, K int) (*q4Resident, bool) {
	key := &bQ4[0]
	if qr := b.q4resident[key]; qr != nil {
		return qr, true
	}
	var rm *ResidentW4A8
	var err error
	if K%w4a8GroupSize == 0 && !int4SlowPath {
		// Fast path: decoder's 2-nibble/byte int4 is byte-identical to the GPU packed
		// layout when K%32==0 (TestInt4LayoutMatch) — upload the bytes straight, mirroring
		// uploadProj's own fast path.
		rm, err = b.ctx.UploadW4A8Packed(bQ4, bScales, N, K)
	} else {
		// Fallback (K not a multiple of 32 → row padding differs): unpack 2-nibble/byte to
		// one nibble (0..15) per element and let UploadW4A8 re-pack. Values preserved —
		// identical unpack loop to uploadProj's own fallback.
		nib := make([]uint8, N*K)
		for r := range N {
			row := bQ4[r*((K+1)/2):]
			dstRow := nib[r*K : r*K+K]
			for k := range K {
				v := row[k>>1]
				if k&1 == 0 {
					dstRow[k] = v & 0x0F
				} else {
					dstRow[k] = v >> 4
				}
			}
		}
		rm, err = b.ctx.UploadW4A8(nib, bScales, N, K)
	}
	if err != nil {
		b.fallbacks++
		return nil, false
	}
	qr := &q4Resident{rm: rm}
	b.q4resident[key] = qr
	return qr, true
}

// MatmulW4A8Batch runs several W4A8 GEMVs that share one activation (fused q/k/v or gate/up,
// M=1 decode) as ONE GPU submit — quantize once, dispatch all, sync once. P-16's int4 twin of
// MatmulW8A8Batch: staged int4 previously had no batch dispatch on ANY GPU backend, so a fused
// call on an int4 model paid one sync PER PROJECTION (three for q/k/v, two for gate/up) instead
// of one for the whole group — exactly the per-dispatch overhead MatmulW8A8Batch exists to
// remove for int8, never extended to int4. Falls back (returns false) for M>1, a group other
// than w4a8GroupSize, or any GPU error — matmulW4A8Batch's caller then uses the CPU batch kernel,
// the same decline contract MatmulW4A8/MatmulW8A8Batch already use.
func (b *webgpuBackend) MatmulW4A8Batch(a []float32, M, K, group int, ops []linalg.W4A8Op) bool {
	if M != 1 || len(ops) == 0 || group != w4a8GroupSize {
		return false
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	rms := make([]decodeWeight, len(ops))
	for i, op := range ops {
		if len(op.W4) == 0 {
			return false
		}
		qr, ok := b.residentW4A8For(op.W4, op.Scales, op.N, K)
		if !ok {
			return false
		}
		rms[i] = qr.rm
	}
	aq, aScales := linalg.QuantizeRowsInt8(a, M, K)
	outs, err := b.ctx.BatchGEMV(aq, aScales[0], rms)
	if err != nil {
		b.fallbacks++
		return false
	}
	for i := range ops {
		copy(ops[i].Dst, outs[i])
	}
	return true
}

// MatmulBT computes dst[M,N] = a · bMatᵀ, with bMat (the weight) uploaded once
// and reused. bMat's identity is its backing pointer — the forward pass passes
// the same weight slice every token.
func (b *webgpuBackend) MatmulBT(a, bMat, dst []float32, M, K, N int) {
	b.mu.Lock()
	out, err := b.matmulLocked(a, bMat, M, K, N)
	if err != nil {
		b.fallbacks++ // N-06: under the lock, like the other fallbacks counters (was racy post-Unlock, tripping -race)
	}
	b.mu.Unlock()
	if err != nil {
		linalg.MatmulBT(a, bMat, dst, M, K, N) // correctness over speed
		return
	}
	copy(dst, out)
}

// matmulLocked uploads (or reuses) the resident weight and dispatches. Caller
// holds b.mu.
func (b *webgpuBackend) matmulLocked(a, bMat []float32, M, K, N int) ([]float32, error) {
	if len(bMat) == 0 {
		return nil, fmt.Errorf("gpu: empty weight")
	}
	key := &bMat[0]
	rm := b.resident[key]
	if rm == nil {
		var err error
		if rm, err = b.ctx.UploadMatrix(bMat, N, K); err != nil {
			return nil, err
		}
		b.resident[key] = rm
	}
	return b.ctx.MatmulBTResident(a, rm, M)
}

// MatmulW8A8Batch runs the fused qkv / gate-up projections (shared activation,
// M=1 decode) as one GPU submit — quantize once, dispatch all, sync once. Falls
// back (returns false) for M>1 or any GPU error.
func (b *webgpuBackend) MatmulW8A8Batch(a []float32, M, K int, ops []linalg.W8A8Op) bool {
	if M < 1 || len(ops) == 0 {
		return false
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	rms := make([]*ResidentW8A8, len(ops))
	for i, op := range ops {
		if len(op.BQ) == 0 {
			return false
		}
		qr := b.qresident[&op.BQ[0]]
		if qr == nil {
			rm, err := b.ctx.UploadW8A8(op.BQ, op.Scales, op.N, K)
			if err != nil {
				b.fallbacks++
				return false
			}
			qr = &qResident{rm: rm}
			b.qresident[&op.BQ[0]] = qr
		}
		rms[i] = qr.rm
	}
	aq, aScales := linalg.QuantizeRowsInt8(a, M, K)
	var outs [][]float32
	var err error
	if M == 1 { // coalesced GEMV (decode); else tiled GEMM (prefill) — both one sync
		dw := make([]decodeWeight, len(rms)) // BatchGEMV is precision-agnostic (P-16); wrap the concrete type
		for i, rm := range rms {
			dw[i] = rm
		}
		outs, err = b.ctx.BatchGEMV(aq, aScales[0], dw)
	} else {
		outs, err = b.ctx.BatchTiled(aq, aScales, M, rms)
	}
	if err != nil {
		b.fallbacks++
		return false
	}
	for i := range ops {
		copy(ops[i].Dst, outs[i])
	}
	return true
}

func (b *webgpuBackend) Close() error {
	b.mu.Lock()
	for _, rm := range b.resident {
		rm.Close()
	}
	b.resident = nil
	for _, qr := range b.qresident {
		if qr.runner != nil {
			qr.runner.Close()
		}
		qr.rm.Close()
	}
	b.qresident = nil
	for _, qr := range b.q4resident {
		if qr.runner != nil {
			qr.runner.Close()
		}
		qr.rm.Close()
	}
	b.q4resident = nil
	b.mu.Unlock()
	b.ctx.Close()
	return nil
}
