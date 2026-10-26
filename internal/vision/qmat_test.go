package vision

import (
	"math"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestQMat_migrationIsBitIdentical is the gate for the vision→WeightMat migration
// (roadmap §2.8, the last of the three open-coded wrappers). The requirement is
// not "close" but BIT-IDENTICAL: WeightMat hides storage, it does not change
// arithmetic, so routing the tower's weights through it must reproduce the
// previous open-coded `qmat` kernel calls exactly.
//
// The tower's own parity tests need checkpoints and skip without them, so they
// cannot witness this. This one runs anywhere: it reconstructs the OLD code path
// against the new one over deterministic pseudo-random weights.
func TestQMat_migrationIsBitIdentical(t *testing.T) {
	// Deterministic, no math/rand dependency — a cheap LCG.
	seq := func(n int, seed uint32) []float32 {
		out := make([]float32, n)
		s := seed
		for i := range out {
			s = s*1664525 + 1013904223
			out[i] = float32(int32(s>>8)) / float32(1<<23) // ~[-1,1)
		}
		return out
	}

	shapes := []struct{ rows, cols, M int }{
		{8, 16, 1},   // single row (decode-shaped)
		{16, 8, 4},   // wide-M
		{64, 32, 7},  // non-power-of-two M
		{3, 5, 2},    // ragged, exercises tail handling
		{128, 64, 3}, // larger, crosses kernel blocking
	}

	for _, quant := range []bool{false, true} {
		for _, sh := range shapes {
			w := seq(sh.rows*sh.cols, 0x9e3779b9)
			a := seq(sh.M*sh.cols, 0x85ebca6b)

			// OLD path: exactly what vision/qmat.go used to do.
			want := make([]float32, sh.M*sh.rows)
			if quant {
				q, s := linalg.QuantizeRowsInt8(w, sh.rows, sh.cols)
				linalg.MatmulBTW8A8(a, q, s, want, sh.M, sh.cols, sh.rows)
			} else {
				f := append([]float32(nil), w...)
				linalg.MatmulBT(a, f, want, sh.M, sh.cols, sh.rows)
			}

			// NEW path: through WeightMat.
			m := newQMat(w, sh.rows, sh.cols, quant)
			got := make([]float32, sh.M*sh.rows)
			m.MatmulBT(a, got, sh.M)

			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("quant=%v rows=%d cols=%d M=%d: element %d is %v, want %v (must be bit-identical)",
						quant, sh.rows, sh.cols, sh.M, i, got[i], want[i])
				}
			}
			if m.Rows() != sh.rows || m.Cols() != sh.cols {
				t.Errorf("quant=%v: shape reported %dx%d, want %dx%d", quant, m.Rows(), m.Cols(), sh.rows, sh.cols)
			}
		}
	}
}

// TestQMat_f32DoesNotAliasSource pins the load-time contract the migration had to
// preserve: WrapF32 aliases its argument, but the tower's weights come from an
// mmap that is released after load, so newQMat must copy. If this regresses the
// symptom is not a compile error — it is garbage weights after the mapping goes.
func TestQMat_f32DoesNotAliasSource(t *testing.T) {
	const rows, cols, M = 4, 4, 1
	src := make([]float32, rows*cols)
	for i := range src {
		src[i] = float32(i + 1)
	}
	m := newQMat(src, rows, cols, false)

	a := make([]float32, M*cols)
	for i := range a {
		a[i] = 1
	}
	before := make([]float32, M*rows)
	m.MatmulBT(a, before, M)

	for i := range src { // scribble over the "mmap"
		src[i] = -999
	}
	after := make([]float32, M*rows)
	m.MatmulBT(a, after, M)

	for i := range before {
		if before[i] != after[i] {
			t.Fatalf("f32 weight aliases its source: output changed %v → %v after the source was overwritten",
				before, after)
		}
	}
}

// TestQMat_int8ExportMatchesGPUContract guards the other consumer of the old
// struct fields: GPUWeights() used to read m.q/m.scales directly and now goes
// through WeightMat.Int8(). The exported codes must be the same ones the CPU
// kernel runs on, or the GPU tower would silently diverge from the CPU one.
func TestQMat_int8ExportMatchesGPUContract(t *testing.T) {
	const rows, cols = 6, 8
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(i%13) - 6
	}
	m := newQMat(w, rows, cols, true)

	q, scales, w8a8, ok := m.Int8()
	if !ok {
		t.Fatal("quantized weight does not report int8 storage")
	}
	if !w8a8 {
		t.Error("vision quantizes for the W8A8 kernel; Int8() reports w8a8=false")
	}
	wantQ, wantS := linalg.QuantizeRowsInt8(w, rows, cols)
	if len(q) != len(wantQ) || len(scales) != len(wantS) {
		t.Fatalf("exported %d codes/%d scales, want %d/%d", len(q), len(scales), len(wantQ), len(wantS))
	}
	for i := range wantQ {
		if q[i] != wantQ[i] {
			t.Fatalf("exported code %d = %d, want %d", i, q[i], wantQ[i])
		}
	}
	for i := range wantS {
		if math.Float32bits(scales[i]) != math.Float32bits(wantS[i]) {
			t.Fatalf("exported scale %d = %v, want %v", i, scales[i], wantS[i])
		}
	}
	// An f32 weight must NOT claim int8 storage — that is what makes
	// GPUWeights() refuse a non-quantized tower instead of exporting nil codes.
	plain := newQMat(w, rows, cols, false)
	if _, _, _, ok := plain.Int8(); ok {
		t.Error("f32 weight reports int8 storage; GPUWeights() would export nil codes")
	}
}
