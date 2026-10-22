package decoder

import (
	"math"

	"github.com/townsendmerino/aikit/linalg"
)

// islandMat holds an f32 island tensor [rows, cols] (a family's projections kept at full precision) as the bf16 WeightMat kind
// when every value is exactly a bf16 value, as a bf16 checkpoint widened by its loader is, and as f32 otherwise. The bf16 kind's
// matmul widens a tile and runs the f32 kernel, so the forward is bit-identical either way at half the bytes (MEM11-bf16,
// docs/tasks/task-memory-campaign-2026-10.md). f is not retained on the bf16 path.
func islandMat(f []float32, rows, cols int) linalg.WeightMat {
	for _, x := range f {
		if math.Float32bits(linalg.BF16ToF32(linalg.F32ToBF16(x))) != math.Float32bits(x) {
			return linalg.WrapF32(f, rows, cols)
		}
	}
	return linalg.QuantizeBF16(f, rows, cols)
}

// islandF32 is an island's values as f32: the slice itself for the f32 kind, a widened copy for bf16. The .giw writes islands
// as f32 (islandMat makes them bf16 again on load), so the format is the same whichever kind a load holds.
func islandF32(w *linalg.WeightMat) []float32 {
	if f, ok := w.F32(); ok {
		return f
	}
	u, _, _ := w.Half16()
	f := make([]float32, len(u))
	for i, b := range u {
		f[i] = linalg.BF16ToF32(b)
	}
	return f
}
