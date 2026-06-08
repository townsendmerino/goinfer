package decoder

import "github.com/townsendmerino/aikit/linalg"

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

// weightMat is one [rows, cols] = [out, in] weight matrix in one of three
// precisions: f32 (default), per-row int8 (quantizeInt8), or group-wise int4
// (quantizeInt4). It hides the storage choice from the forward pass: every
// matmul site calls matmul, and the tied embedding lookup calls embedRow,
// regardless of precision.
type weightMat struct {
	f32    []float32 // [rows*cols], set unless quantized
	q8     []int8    // [rows*cols], set for int8
	scales []float32 // [rows], per-row scale for int8
	q4     []byte    // [rows*((cols+1)/2)] packed nibbles, set for int4
	q4s    []float32 // [rows*nGroups], per-group scale for int4
	group  int       // int4 group size (0 unless int4)
	w8a8   bool      // int8 weights but full int8×int8 matmul (quantInt8I8)
	rows   int       // out features (N)
	cols   int       // in features (K)
}

func newWeightMat(f32 []float32, rows, cols int) weightMat {
	return weightMat{f32: f32, rows: rows, cols: cols}
}

// streamQuantized builds a [rows, cols] weightMat in the target precision by
// dequantizing each row through rowInto (into a reused cols-wide scratch) and
// quantizing it straight into the resident arrays — never materializing the whole
// [rows*cols] f32. This is the load-time memory-bandwidth win: a big GGUF tensor's
// f32 intermediate stays one row wide (in cache) instead of streaming to DRAM and
// back. The result is bit-identical to newWeightMat(fullF32).quantize(mode): the
// per-row primitives here are exactly the ones QuantizeRowsInt8 / QuantizeGroupsInt4
// call internally, just driven one row at a time.
func streamQuantized(rows, cols int, mode quantMode, rowInto func(r int, dst []float32) error) (weightMat, error) {
	w := weightMat{rows: rows, cols: cols}
	scratch := make([]float32, cols)
	switch mode {
	case quantInt8, quantInt8I8:
		w.q8 = make([]int8, rows*cols)
		w.scales = make([]float32, rows)
		w.w8a8 = mode == quantInt8I8
		for r := range rows {
			if err := rowInto(r, scratch); err != nil {
				return weightMat{}, err
			}
			w.scales[r] = linalg.QuantizeRowInt8(scratch, w.q8[r*cols:(r+1)*cols])
		}
	case quantInt4:
		const group = int4GroupSize
		nGroups := (cols + group - 1) / group
		bpr := (cols + 1) / 2
		w.q4 = make([]byte, rows*bpr)
		w.q4s = make([]float32, rows*nGroups)
		w.group = group
		for r := range rows {
			if err := rowInto(r, scratch); err != nil {
				return weightMat{}, err
			}
			linalg.QuantizeGroupInt4Row(scratch, cols, group, w.q4[r*bpr:(r+1)*bpr], w.q4s[r*nGroups:(r+1)*nGroups])
		}
	default: // quantNone — no quant target, keep the full f32
		w.f32 = make([]float32, rows*cols)
		for r := range rows {
			if err := rowInto(r, w.f32[r*cols:(r+1)*cols]); err != nil {
				return weightMat{}, err
			}
		}
	}
	return w, nil
}

// quantize streams this matrix to the requested resident precision, freeing the
// f32 backing. No-op for quantNone or if already quantized.
func (w *weightMat) quantize(mode quantMode) {
	switch mode {
	case quantInt8:
		w.quantizeInt8()
	case quantInt8I8:
		w.quantizeInt8()
		w.w8a8 = true
	case quantInt4:
		w.quantizeInt4(int4GroupSize)
	}
}

// quantizeInt8 converts f32 → per-row int8 in place and frees the f32 backing
// (the M8 memory win). No-op if already quantized.
func (w *weightMat) quantizeInt8() {
	if w.q8 != nil || w.q4 != nil || w.f32 == nil {
		return
	}
	w.q8, w.scales = linalg.QuantizeRowsInt8(w.f32, w.rows, w.cols)
	w.f32 = nil
}

// quantizeInt4 converts f32 → group-wise int4 in place and frees the f32
// backing (~⅛ f32). No-op if already quantized.
func (w *weightMat) quantizeInt4(group int) {
	if w.q4 != nil || w.q8 != nil || w.f32 == nil {
		return
	}
	w.q4, w.q4s = linalg.QuantizeGroupsInt4(w.f32, w.rows, w.cols, group)
	w.group = group
	w.f32 = nil
}

// matmul computes dst[M, rows] = a[M, cols] · this[rows, cols]ᵀ, dispatching to
// the int4, int8, or f32 kernel. The f32 path uses the backend (so a GPU backend
// can substitute); the quantized paths are CPU-only for now.
func (w *weightMat) matmul(be Backend, a, dst []float32, M int) {
	switch {
	case w.q4 != nil:
		linalg.MatmulBTQ4(a, w.q4, w.q4s, dst, M, w.cols, w.rows, w.group)
	case w.q8 != nil && w.w8a8:
		if qb, ok := be.(QuantBackend); ok && qb.MatmulW8A8(a, w.q8, w.scales, dst, M, w.cols, w.rows) {
			return
		}
		linalg.MatmulBTW8A8(a, w.q8, w.scales, dst, M, w.cols, w.rows)
	case w.q8 != nil:
		linalg.MatmulBTQ8(a, w.q8, w.scales, dst, M, w.cols, w.rows)
	default:
		be.MatmulBT(a, w.f32, dst, M, w.cols, w.rows)
	}
}

// isW8A8 reports whether this matrix uses the int8×int8 (W8A8) path — the only
// one with a zero-alloc Workspace + batched-dispatch kernel.
func (w *weightMat) isW8A8() bool { return w.q8 != nil && w.w8a8 }

// matmulInto is matmul using a Workspace for the W8A8 path, so steady-state
// decode quantizes the activation once into reusable scratch (no per-call
// allocation). Other quant paths ignore ws and use matmul.
func (w *weightMat) matmulInto(ws *linalg.Workspace, be Backend, a, dst []float32, M int) {
	if w.isW8A8() {
		if qb, ok := be.(QuantBackend); ok && qb.MatmulW8A8(a, w.q8, w.scales, dst, M, w.cols, w.rows) {
			return
		}
		linalg.MatmulBTW8A8Into(ws, a, w.q8, w.scales, dst, M, w.cols, w.rows)
		return
	}
	w.matmul(be, a, dst, M)
}

// embedRow writes row id (one token's embedding) into dst[:cols], dequantizing
// on the fly when the table is quantized.
func (w *weightMat) embedRow(id int, dst []float32) {
	switch {
	case w.q4 != nil:
		bpr := (w.cols + 1) / 2
		nGroups := (w.cols + w.group - 1) / w.group
		linalg.DequantizeRowInt4(w.q4[id*bpr:(id+1)*bpr], w.q4s[id*nGroups:(id+1)*nGroups], w.group, w.cols, dst)
	case w.q8 != nil:
		lo := id * w.cols
		linalg.DequantizeRowInt8(w.q8[lo:lo+w.cols], w.scales[id], dst)
	default:
		copy(dst, w.f32[id*w.cols:id*w.cols+w.cols])
	}
}
