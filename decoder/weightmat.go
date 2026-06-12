package decoder

import "github.com/townsendmerino/aikit/linalg"

// Weight matrices are linalg.WeightMat (aikit): one type that hides f32 / per-row
// int8 / group-wise int4 storage behind uniform accessors + the linalg kernels.
// goinfer keeps the model POLICY here — which table gets which precision
// (quantMode), the int4 group size, and the matmul backend routing (the staged
// GPU hooks below) — while the storage wrapper, quantize primitives, and Row
// dequant live in linalg. (Consolidates the wrapper formerly open-coded as
// decoder.weightMat; see aikit linalg.WeightMat.)

// quantMode selects the resident weight precision the loader streams into (see
// loadWeights). The f32 path keeps the widened weights; int8 is per-row symmetric
// (¼ f32); int4 is group-wise symmetric (~⅛ f32).
type quantMode uint8

const (
	quantNone quantMode = iota
	quantInt8
	quantInt4
	quantInt8I8 // weights int8 (as quantInt8) but matmul is full int8×int8 (W8A8)
)

// embedding returns the precision to use for the token-embedding table (and the
// LM head, tied or not). These are logit-critical: at int4 they flip the argmax
// and tank the cosine (the tied head dots every logit against them), so int4
// mode keeps them at int8 — mirroring how GGUF Q4_K_M keeps token_embd/output at
// Q6_K while the projections go 4-bit. int8 and f32 modes use themselves.
func (q quantMode) embedding() quantMode {
	if q == quantInt4 {
		return quantInt8
	}
	return q
}

// int4GroupSize is the number of consecutive input features that share one f32
// scale in the int4 path. 32 matches GGUF Q4_K's sub-block granularity — small
// enough to keep 4-bit accuracy, large enough that the per-group scale overhead
// stays ~0.125 byte/element.
const int4GroupSize = 32

// streamQuantized builds a [rows, cols] linalg.WeightMat in the target precision
// by dequantizing each row through rowInto (into a reused cols-wide scratch) and
// quantizing it straight into the resident arrays — never materializing the whole
// [rows*cols] f32. This is the load-time memory-bandwidth win: a big GGUF tensor's
// f32 intermediate stays one row wide (in cache) instead of streaming to DRAM and
// back. Bit-identical to quantizeWM(WrapF32(fullF32), mode): the per-row primitives
// here are exactly the ones linalg.Quantize{Rows,Groups} call internally, just
// driven one row at a time; the result is WrapInt8/WrapInt4/WrapF32'd (no copy).
func streamQuantized(rows, cols int, mode quantMode, rowInto func(r int, dst []float32) error) (linalg.WeightMat, error) {
	scratch := make([]float32, cols)
	switch mode {
	case quantInt8, quantInt8I8:
		q8 := make([]int8, rows*cols)
		scales := make([]float32, rows)
		for r := range rows {
			if err := rowInto(r, scratch); err != nil {
				return linalg.WeightMat{}, err
			}
			scales[r] = linalg.QuantizeRowInt8(scratch, q8[r*cols:(r+1)*cols])
		}
		return linalg.WrapInt8(q8, scales, rows, cols, mode == quantInt8I8), nil
	case quantInt4:
		const group = int4GroupSize
		nGroups := (cols + group - 1) / group
		bpr := (cols + 1) / 2
		q4 := make([]byte, rows*bpr)
		q4s := make([]float32, rows*nGroups)
		for r := range rows {
			if err := rowInto(r, scratch); err != nil {
				return linalg.WeightMat{}, err
			}
			linalg.QuantizeGroupInt4Row(scratch, cols, group, q4[r*bpr:(r+1)*bpr], q4s[r*nGroups:(r+1)*nGroups])
		}
		return linalg.WrapInt4(q4, q4s, rows, cols, group), nil
	default: // quantNone — no quant target, keep the full f32
		f32 := make([]float32, rows*cols)
		for r := range rows {
			if err := rowInto(r, f32[r*cols:(r+1)*cols]); err != nil {
				return linalg.WeightMat{}, err
			}
		}
		return linalg.WrapF32(f32, rows, cols), nil
	}
}

// quantizeWM returns w streamed to the requested resident precision — the
// immutable-WeightMat replacement for the old in-place weightMat.quantize (the
// freed-f32 memory win comes from dropping the old f32 reference at the call site).
// No-op for quantNone or if w isn't f32-resident (already quantized / empty).
func quantizeWM(w linalg.WeightMat, mode quantMode) linalg.WeightMat {
	f32, ok := w.F32()
	if !ok {
		return w
	}
	switch mode {
	case quantInt8:
		return linalg.QuantizeInt8(f32, w.Rows(), w.Cols(), false)
	case quantInt8I8:
		return linalg.QuantizeInt8(f32, w.Rows(), w.Cols(), true)
	case quantInt4:
		return linalg.QuantizeInt4(f32, w.Rows(), w.Cols(), int4GroupSize)
	default:
		return w
	}
}

// isW8A8 reports whether w uses the int8×int8 (W8A8) path — the only one with a
// zero-alloc Workspace + batched-dispatch kernel.
func isW8A8(w *linalg.WeightMat) bool {
	_, _, w8a8, ok := w.Int8()
	return ok && w8a8
}

// wmInt8 / wmScales pull the int8 codes / per-row scales out of a WeightMat for the
// batched W8A8 ops (the forward's only sites that read the raw arrays directly, to
// fuse several matrices into one matmulW8A8Batch dispatch). Both assume isW8A8(w).
func wmInt8(w *linalg.WeightMat) []int8      { q8, _, _, _ := w.Int8(); return q8 }
func wmScales(w *linalg.WeightMat) []float32 { _, s, _, _ := w.Int8(); return s }

// matmul computes dst[M, rows] = a[M, cols] · wᵀ, dispatching on w's precision
// with goinfer's backend routing: the f32 and W8A8 paths can run on a GPU backend
// (be.MatmulBT / QuantBackend.MatmulW8A8); int4 (W4A8) and weight-only int8 (Q8)
// stay CPU. (The old weightMat.matmul, now a free function over linalg.WeightMat.)
func matmul(be Backend, w *linalg.WeightMat, a, dst []float32, M int) {
	if q4, q4s, group, ok := w.Int4(); ok {
		// int4 weights run the int8-activation W4A8 integer kernel at EVERY M (decode
		// AND prefill): it stays integer (int4 weight × int8 activation) and benchmarks
		// faster than the dequant-to-f32 Q4 path at every M, and its per-output result
		// is M-independent so batched prefill is bit-identical to sequential decode.
		linalg.MatmulBTW4A8(a, q4, q4s, dst, M, w.Cols(), w.Rows(), group)
		return
	}
	if q8, scales, w8a8, ok := w.Int8(); ok {
		if w8a8 {
			if qb, ok := be.(QuantBackend); ok && qb.MatmulW8A8(a, q8, scales, dst, M, w.Cols(), w.Rows()) {
				return
			}
			linalg.MatmulBTW8A8(a, q8, scales, dst, M, w.Cols(), w.Rows())
			return
		}
		linalg.MatmulBTQ8(a, q8, scales, dst, M, w.Cols(), w.Rows())
		return
	}
	f32, _ := w.F32()
	be.MatmulBT(a, f32, dst, M, w.Cols(), w.Rows())
}

// matmulInto is matmul using a Workspace for the W8A8 path, so steady-state decode
// quantizes the activation once into reusable scratch (no per-call allocation).
// Other quant paths ignore ws and use matmul.
func matmulInto(ws *linalg.Workspace, be Backend, w *linalg.WeightMat, a, dst []float32, M int) {
	if isW8A8(w) {
		q8, scales, _, _ := w.Int8()
		if qb, ok := be.(QuantBackend); ok && qb.MatmulW8A8(a, q8, scales, dst, M, w.Cols(), w.Rows()) {
			return
		}
		linalg.MatmulBTW8A8Into(ws, a, q8, scales, dst, M, w.Cols(), w.Rows())
		return
	}
	matmul(be, w, a, dst, M)
}
