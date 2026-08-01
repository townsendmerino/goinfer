package decoder

import (
	"bytes"
	"math"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// testWeights builds a deterministic [rows*cols] f32 matrix with per-group range
// asymmetry and a few outliers, so int4 rounding is actually exercised.
func testWeights(rows, cols int) []float32 {
	w := make([]float32, rows*cols)
	for i := range w {
		v := math.Sin(float64(i)*0.7) * float64(1+i%5)
		if i%37 == 0 {
			v *= 6 // outliers → widen the per-group range
		}
		w[i] = float32(v)
	}
	return w
}

// TestFakeQuantSymMatchesRuntimeInt4 is the harness's FIDELITY GATE: the whole int4 matrix
// in docs/task-gemma4-moe.md is only trustworthy because its "sym" control reproduces the
// real-int4 behavior. This asserts the harness's `sym` scheme is element-wise identical to
// aikit's runtime int4 quantizer (QuantizeGroupsInt4 → DequantizeRowInt4). If this fails,
// the "sym" cell no longer stands in for the shipping int4 path and no other cell is
// interpretable.
func TestFakeQuantSymMatchesRuntimeInt4(t *testing.T) {
	const rows, cols, group = 3, 96, int4GroupSize // 96 = 3 groups of 32
	w := testWeights(rows, cols)

	// aikit runtime int4 → dequant, per row (exactly how streamQuantized stores/reads it)
	packed, scales := linalg.QuantizeGroupsInt4(w, rows, cols, group)
	nGroups := (cols + group - 1) / group
	bpr := (cols + 1) / 2
	ref := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		linalg.DequantizeRowInt4(packed[r*bpr:(r+1)*bpr], scales[r*nGroups:(r+1)*nGroups], group, cols, ref[r*cols:(r+1)*cols])
	}

	got := fakeQuantInt4("sym", w, rows, cols, group)
	for i := range ref {
		if d := math.Abs(float64(ref[i] - got[i])); d > 1e-6 {
			t.Fatalf("sym harness diverges from runtime int4 at %d: ref=%g got=%g (Δ=%g)", i, ref[i], got[i], d)
		}
	}
}

// TestFakeQuantOffBitIdentical proves the load-time hook is a strict no-op when
// GOINFER_FAKEQUANT is unset (the default): the int4 quantizeWM path is byte-for-byte
// identical to aikit's QuantizeInt4, so every existing int4 consumer is unaffected by the
// diagnostic's presence in the binary.
func TestFakeQuantOffBitIdentical(t *testing.T) {
	if fakeQuantScheme != "" {
		t.Skip("GOINFER_FAKEQUANT is set; the off-path identity check only holds when unset")
	}
	const rows, cols = 4, 64
	w := testWeights(rows, cols)

	got := quantizeWM(linalg.WrapF32(append([]float32(nil), w...), rows, cols), quantInt4)
	want := linalg.QuantizeInt4(append([]float32(nil), w...), rows, cols, int4GroupSize)

	gq, gs, gg, gok := got.Int4()
	wq, ws, wg, wok := want.Int4()
	if !gok || !wok {
		t.Fatalf("expected int4 resident (got ok=%v want ok=%v)", gok, wok)
	}
	if gg != wg {
		t.Fatalf("group size mismatch: %d vs %d", gg, wg)
	}
	if !bytes.Equal(gq, wq) {
		t.Fatal("packed nibbles differ — default int4 path is NOT bit-identical with the hook present")
	}
	for i := range ws {
		if gs[i] != ws[i] {
			t.Fatalf("scale mismatch at group %d: %g vs %g", i, gs[i], ws[i])
		}
	}
}
