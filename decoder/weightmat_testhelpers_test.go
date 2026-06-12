package decoder

import "github.com/townsendmerino/aikit/linalg"

// Test-only accessors that reach a linalg.WeightMat's stored arrays/flags through
// its exported accessors — the tests inspect resident precision + raw arrays
// (aliasing, quant-equality) that used to read the unexported weightMat fields
// directly. Production code uses Kind()/Int8()/Int4()/F32() at the call site.
func tF32(w *linalg.WeightMat) []float32    { f, _ := w.F32(); return f }
func tQ8(w *linalg.WeightMat) []int8        { q, _, _, _ := w.Int8(); return q }
func tScales(w *linalg.WeightMat) []float32 { _, s, _, _ := w.Int8(); return s }
func tQ4(w *linalg.WeightMat) []byte        { q, _, _, _ := w.Int4(); return q }
func tW8A8(w *linalg.WeightMat) bool        { _, _, b, _ := w.Int8(); return b }
func tGroup(w *linalg.WeightMat) int        { _, _, g, _ := w.Int4(); return g }
