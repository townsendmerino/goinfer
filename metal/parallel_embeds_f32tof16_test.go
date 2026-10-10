//go:build darwin

package metal

import (
	"math/rand"
	"testing"
)

// TestParallelEmbedsF32ToF16_matchesSerial pins P-14 (docs/audit-2026-09-10.md): PrefillLast's host f32→f16 conversion, on the
// TTFT path, is split by row across up to 8 workers by parallelEmbedsF32ToF16. f32ToF16 is a pure per-element function (no
// shared state, no reduction to reassociate), so the split must be byte-identical to the serial reference, not merely close;
// asserted directly because this function is row-parallel over [][]float32, unlike parallelF32ToF16's flat []float32.
func TestParallelEmbedsF32ToF16_matchesSerial(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, shape := range []struct{ M, H int }{
		{1, 8},       // below the parallel threshold entirely
		{4, 16},      // small, still serial (M*H < 8192)
		{64, 128},    // 8192 exactly at the boundary
		{2048, 3584}, // the audit's own cited TTFT shape
	} {
		embs := make([][]float32, shape.M)
		for m := range embs {
			embs[m] = make([]float32, shape.H)
			for i := range embs[m] {
				embs[m][i] = rng.Float32()*20 - 10
			}
		}

		want := make([]uint16, shape.M*shape.H)
		for m := range shape.M {
			for i := range shape.H {
				want[m*shape.H+i] = f32ToF16(embs[m][i])
			}
		}

		got := make([]uint16, shape.M*shape.H)
		parallelEmbedsF32ToF16(got, embs, shape.H)

		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("shape M=%d H=%d: got[%d] = %#04x, want %#04x (serial reference)",
					shape.M, shape.H, i, got[i], want[i])
			}
		}
	}
}
