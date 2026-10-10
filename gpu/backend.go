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
// core modules ever importing github.com/oliverbestmann/webgpu.
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

// webgpuBackend runs the decoder's matmuls on a WebGPU adapter (Vulkan / Metal / D3D12) via the Context foundation in this
// package. Built only under -tags gpu.
//
// Resident weights: a model weight matrix is constant across every token, so the first matmul for a given weight uploads it to
// a GPU storage buffer and caches the handle, keyed by the slice's backing pointer; later calls upload only the (small)
// activation. On any per-call GPU error MatmulBT falls back to the CPU matmul and the W8A8/W4A8 methods return false so the
// caller uses its CPU kernel, so results are always correct. Each matmul here is its own synchronous dispatch and readback;
// the device-resident forward is BuildResident (residency.go). History: docs/code-notes/gpu.md#webgpuBackend.
type webgpuBackend struct {
	ctx  *Context
	name string

	mu         sync.Mutex // Context is not goroutine-safe
	resident   map[*float32]*ResidentMatrix
	qresident  map[*int8]*qResident  // W8A8 weights kept resident (+ a decode runner)
	q4resident map[*byte]*q4Resident // G6 (docs/tasks/task-gpu-paths-2026-09.md): W4A8 twin — the "staged int4" item
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

// MatmulW4A8 is MatmulW8A8's int4 (W4A8) twin for the STAGED (non-resident) path: without it an int4 model's staged
// projections ran on the CPU whichever backend was active. M=1 only (decode); M>1 has no int4 GEMM kernel on this backend and
// declines, so the caller falls back to the CPU W4A8 kernel. bQ4 is decoder's packed layout (2 nibbles/byte); group must be
// w4a8GroupSize (32). History: docs/code-notes/gpu.md#webgpuBackend.MatmulW4A8.
func (b *webgpuBackend) MatmulW4A8(a []float32, bQ4 []byte, bScales16 []uint16, group int, dst []float32, M, K, N int) bool {
	if len(bQ4) == 0 || M != 1 || group != w4a8GroupSize {
		return false
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	qr, ok := b.residentW4A8For(bQ4, bScales16, nil, N, K)
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

// residentW4A8For resolves (or uploads and caches) the resident W4A8 weight for one packed int4 buffer, keyed by its backing
// pointer, shared by MatmulW4A8 and MatmulW4A8Batch so the upload/unpack logic exists exactly once. Caller holds b.mu.
// ok=false means upload failed; the caller counts the fallback.
//
// The scales arrive as binary16 (scales16, a WeightMat's storage) or, from an op built against f32,
// as scales32. They are read only on a cache miss: widened (exactly) for the f32 upload path, which
// converts back to the same binary16 bits the WeightMat holds.
func (b *webgpuBackend) residentW4A8For(bQ4 []byte, scales16 []uint16, scales32 []float32, N, K int) (*q4Resident, bool) {
	key := &bQ4[0]
	if qr := b.q4resident[key]; qr != nil {
		return qr, true
	}
	bScales := scales32
	if scales16 != nil {
		bScales = make([]float32, len(scales16))
		linalg.F16ToF32Slice(bScales, scales16)
	}
	var rm *ResidentW4A8
	var err error
	if K%w4a8GroupSize == 0 {
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

// MatmulW4A8Batch runs several W4A8 GEMVs that share one activation (fused q/k/v or gate/up, M=1 decode) as ONE GPU submit:
// quantize once, dispatch all, sync once. The int4 twin of MatmulW8A8Batch. Returns false for M>1, a group other than
// w4a8GroupSize, or any GPU error, and the caller then uses the CPU batch kernel.
// History: docs/code-notes/gpu.md#webgpuBackend.MatmulW4A8Batch.
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
		qr, ok := b.residentW4A8For(op.W4, op.ScalesF16, op.Scales, op.N, K)
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

// The staged int4 interfaces are met by a runtime type assertion (decoder's matmul/matmulInto), so a
// signature drift would silently route every int4 projection to the CPU; these make it a build error.
var (
	_ decoder.QuantBackend4      = (*webgpuBackend)(nil)
	_ decoder.QuantBatchBackend4 = (*webgpuBackend)(nil)
)
