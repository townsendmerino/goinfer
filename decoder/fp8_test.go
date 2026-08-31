package decoder

import (
	"math"
	"testing"
)

// TestFP8E4M3_matchesReference checks the derivation in fp8.go against every one of the 256
// possible byte patterns, using the committed PyTorch-generated oracle.
//
// EXHAUSTIVE ON PURPOSE. The domain is 256 values, so there is no reason to sample it and no
// excuse for a spot-check: a dequant table that is right on the common patterns and wrong on
// subnormals, on -0, or on the no-infinity exponent produces finite, plausibly-scaled,
// completely wrong weights. Nothing downstream errors on that — it is the exact failure mode
// mxfp4.go's header calls out ("would have produced finite, plausibly-scaled, completely wrong
// weights"), and the only defence is comparing against a reference rather than against
// intuition.
func TestFP8E4M3_matchesReference(t *testing.T) {
	for i := range 256 {
		got := fp8E4M3ToF32(byte(i))
		want := fp8E4M3RefTable[i]

		if math.IsNaN(want) {
			if !math.IsNaN(float64(got)) {
				t.Errorf("0x%02X: got %v, reference is NaN", i, got)
			}
			continue
		}
		if float64(got) != want {
			t.Errorf("0x%02X: got %v, reference %v", i, got, want)
		}
		// Signed zero is a real distinction here (0x80 is -0.0) and == would not catch a sign
		// flip on it, so check the bit pattern for the zeros specifically.
		if want == 0 && math.Signbit(float64(got)) != math.Signbit(want) {
			t.Errorf("0x%02X: zero sign differs: got %v, reference %v", i, got, want)
		}
	}

	// Anchors worth naming, because each is a place a plausible implementation goes wrong.
	if v := fp8E4M3ToF32(0x7E); v != 448 {
		t.Errorf("max finite: got %v, want 448 — e4m3fn has NO infinities, so the all-ones exponent is a normal number", v)
	}
	if !math.IsNaN(float64(fp8E4M3ToF32(0x7F))) || !math.IsNaN(float64(fp8E4M3ToF32(0xFF))) {
		t.Error("0x7F/0xFF must be the only NaNs (S.1111.111)")
	}
	if v := fp8E4M3ToF32(0x01); v != 0.001953125 {
		t.Errorf("smallest subnormal: got %v, want 2^-9 = 0.001953125", v)
	}

	// The table must agree with the function it is built from — otherwise the fast path and
	// the derivation could drift, and only the slow one is tested above.
	for i := range 256 {
		f, tv := fp8E4M3ToF32(byte(i)), fp8E4M3Table[i]
		if math.IsNaN(float64(f)) != math.IsNaN(float64(tv)) || (!math.IsNaN(float64(f)) && f != tv) {
			t.Fatalf("0x%02X: table %v != function %v", i, tv, f)
		}
	}
}

// TestFP8DequantBlocked checks the block-scale composition, which is the piece with no
// precedent in this tree: MXFP4 carries its scale INLINE, one per 32-element block, while fp8
// keeps a separate 2-D grid (`weight_scale_inv`, 128x128 for DeepSeek and Qwen3-FP8). Getting
// the block indexing wrong is silent — every element still gets *a* scale, just occasionally
// the wrong neighbour's.
func TestFP8DequantBlocked(t *testing.T) {
	// 3x4 weights, 2x2 blocks -> a 2x2 scale grid, with the last block row/col partial. The
	// ragged edge is the point: 3 is not a multiple of 2.
	const rows, cols, bR, bC = 3, 4, 2, 2
	w := make([]byte, rows*cols)
	for i := range w {
		w[i] = 0x38 // e4m3 1.0 — so the output IS the scale, and an indexing bug is readable
	}
	scales := []float32{10, 20, 30, 40} // [2,2] grid, row-major

	got, err := fp8DequantBlocked(w, scales, rows, cols, bR, bC)
	if err != nil {
		t.Fatalf("dequant: %v", err)
	}
	want := []float32{
		10, 10, 20, 20,
		10, 10, 20, 20,
		30, 30, 40, 40, // row 2 is in block-row 1 even though that block is half height
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("element %d (r=%d c=%d): got %v want %v — block indexing is wrong",
				i, i/cols, i%cols, got[i], want[i])
		}
	}

	if _, err := fp8DequantBlocked(w, scales[:3], rows, cols, bR, bC); err == nil {
		t.Error("a short scale grid must be refused, not silently indexed past")
	}
	if _, err := fp8DequantBlocked(w[:5], scales, rows, cols, bR, bC); err == nil {
		t.Error("a short payload must be refused")
	}
}
