package vision

import "github.com/townsendmerino/aikit/linalg"

// qmat is a matmul weight held either as f32 or as per-row int8 (W8A8). Its
// matmul computes dst[M, rows] = a[M, cols] · weightᵀ — i.e. MatmulBT with the
// weight as a [rows, cols] = [out, in] matrix. The int8 form quantizes the weight
// once at load (per-row symmetric int8 + scales, via aikit's QuantizeRowsInt8)
// and runs the integer W8A8 kernel — ~2–4× the f32 throughput on the matmul-bound
// SigLIP tower, at the cost of cosine ~0.999 vs the f32 path's 1.0.
type qmat struct {
	rows, cols int
	f32        []float32 // non-nil ⇒ f32 path
	q          []int8    // non-nil ⇒ int8 (W8A8) path
	scales     []float32 // per-row weight scales (int8 path)
}

// newQMat wraps an f32 [rows, cols] weight, quantizing to int8 when quant is set.
// The source slice is copied out (f32) or consumed into int8, so the caller's
// mmap'd tensor can be released after.
func newQMat(w []float32, rows, cols int, quant bool) qmat {
	if quant {
		q, s := linalg.QuantizeRowsInt8(w, rows, cols)
		return qmat{rows: rows, cols: cols, q: q, scales: s}
	}
	return qmat{rows: rows, cols: cols, f32: append([]float32(nil), w...)}
}

// matmul computes dst[M, rows] = a[M, cols] · weightᵀ (CPU; an int8 weight uses
// the W8A8 kernel, f32 uses MatmulBT). The GPU path is the resident encoder in
// the gpu module, not a per-call offload here.
func (m qmat) matmul(a, dst []float32, M int) {
	if m.q != nil {
		linalg.MatmulBTW8A8(a, m.q, m.scales, dst, M, m.cols, m.rows)
		return
	}
	linalg.MatmulBT(a, m.f32, dst, M, m.cols, m.rows)
}
