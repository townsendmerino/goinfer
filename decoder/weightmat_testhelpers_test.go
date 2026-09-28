package decoder

import "github.com/townsendmerino/aikit/linalg"

// Test-only accessors that reach a linalg.WeightMat's stored arrays/flags through
// its exported accessors — the tests inspect resident precision + raw arrays
// (aliasing, quant-equality) that used to read the unexported weightMat fields
// directly. Production code uses Kind()/Int8()/Int4()/F32() at the call site.
func tF32(w *linalg.WeightMat) []float32    { f, _ := w.F32(); return f }
func tQ8(w *linalg.WeightMat) []int8        { q, _, _, _ := w.Int8(); return q }
func tScales(w *linalg.WeightMat) []float32 { _, s, _, _ := w.Int8(); return s }
func tW8A8(w *linalg.WeightMat) bool        { _, _, b, _ := w.Int8(); return b }
func tGroup(w *linalg.WeightMat) int        { _, _, g, _ := Int4F32(w); return g }

// Tests written against f32 int4 scales. aikit v1.50.0 stores them as binary16 and deprecates the f32
// accessors; these do exactly what those did — widen into a new slice, or convert with linalg.F32ToF16 —
// so a test's meaning is unchanged. (Int4F32, the production helper, covers Int4.)
func int4Row4F32(w *linalg.WeightMat) (packed4 []byte, scales4 []float32, ok bool) {
	packed4, s16, ok := w.Int4Row4F16()
	if s16 != nil {
		scales4 = make([]float32, len(s16))
		linalg.F16ToF32Slice(scales4, s16)
	}
	return packed4, scales4, ok
}

func wrapInt4F32(q4 []byte, q4s []float32, rows, cols, group int) linalg.WeightMat {
	return linalg.WrapInt4F16(q4, linalg.F32ToF16Scales(q4s), rows, cols, group)
}

func wrapInt4Row4F32(q4 []byte, q4s []float32, rows, cols, group int, q4Row4 []byte, q4Row4Scales []float32) linalg.WeightMat {
	return linalg.WrapInt4Row4F16(q4, linalg.F32ToF16Scales(q4s), rows, cols, group, q4Row4, linalg.F32ToF16Scales(q4Row4Scales))
}
