package decoder

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestRequantBar_DoubleQuantCost measures what Metal's DOUBLE quantization costs, per model.
//
// Metal's BuildResident cannot consume the decoder's int4 — it requires an int8 load and
// re-quantizes to its own W4A8, so a weight travels f32 → int8 (per-row, scale=max|row|/127)
// → int4 (group-32, scale=maxabs/7). CUDA and WebGPU upload the decoder's int4 byte-identically
// and pay the int4 step ONCE. That asymmetry has always been known; what was never measured is
// whether it MATTERS, and the assumption was that it is a rounding detail.
//
// It should not be a detail for a model with high per-row dynamic range, and the mechanism is
// specific: the int8 step's scale is set by the row's LARGEST element. A group whose own maxabs
// is far below the row max therefore lands on only a handful of int8 levels BEFORE int4 ever
// sees it — the group scale then spreads 15 int4 levels over a signal already crushed to 3 or 4.
// Where a row is flat (row max ≈ group max) the int8 grid is finer than the int4 one everywhere
// and the first step is nearly free. So the cost is a property of the WEIGHTS, not the kernel,
// and it predicts exactly the split measured on Metal: gemma3 loses 0.104 of logit cosine
// against its own CPU-int4 twin while the qwen control loses nothing.
//
// Two errors are compared against the same reference (the int8 row — what Metal actually gets):
//
//	direct: dequant4(decoder's own int4)      -- one quantization, what CUDA/WebGPU ship
//	metal:  dequant4(quant4(dequant8(int8)))  -- two, what Metal ships
//
// If the double step is benign, they match. If metal >> direct on gemma and not on the control,
// the parity gap has a mechanism and a fix, and it is not a kernel bug.
func TestRequantBar_DoubleQuantCost(t *testing.T) {
	if testing.Short() {
		t.Skip("loads real models twice each")
	}
	for _, tc := range []struct {
		what string
		path string
	}{
		{"dense control (qwen2.5-1.5b)", os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")},
		{"gemma3-4b", os.ExpandEnv("$HOME/models/gemma-3-4b-it-Q4_K_M.gguf")},
	} {
		t.Run(tc.what, func(t *testing.T) {
			if _, err := os.Stat(tc.path); err != nil {
				t.Skipf("no checkpoint at %s", tc.path)
			}
			m8, err := Load(tc.path, Options{Quant: "int8int8"})
			if err != nil {
				t.Fatalf("load int8: %v", err)
			}
			m4, err := Load(tc.path, Options{Quant: "int4"})
			if err != nil {
				t.Fatalf("load int4: %v", err)
			}
			w8, w4 := m8.w.matmulWeights(), m4.w.matmulWeights()

			var sumDirect, sumMetal, sumRange float64
			var n int
			var worstRatio float64
			for ti := range w8 {
				a8, a4 := w8[ti], w4[ti]
				if a8.Rows() == 0 || a4.Kind() != "int4" { // embed/head are int8-pinned in both
					continue
				}
				cols := a8.Cols()
				ref := make([]float32, cols) // the int8 row: Metal's actual input
				dir := make([]float32, cols) // decoder's int4, dequantized
				packed := make([]byte, (cols+1)/2)
				scales := make([]float32, (cols+int4GroupSize-1)/int4GroupSize)
				rq := make([]float32, cols)

				step := 1 + a8.Rows()/8 // sample rows; the statistic is stable well before all of them
				for r := 0; r < a8.Rows(); r += step {
					a8.Row(r, ref)
					a4.Row(r, dir)
					linalg.QuantizeGroupInt4Row(ref, cols, int4GroupSize, packed, scales)
					linalg.DequantizeRowInt4(packed, scales, int4GroupSize, cols, rq)

					sumDirect += relErrVec(dir, ref)
					sumMetal += relErrVec(rq, ref)
					sumRange += rowRange(ref)
					n++
				}
				if n > 0 {
					if got := sumMetal / math.Max(sumDirect, 1e-12); got > worstRatio {
						worstRatio = got
					}
				}
			}
			if n == 0 {
				t.Skip("no int4 tensors")
			}
			t.Logf("%s: direct-int4 err %.4f%% | metal double-quant err %.4f%% | %.2fx worse | mean row dynamic range %.1fx",
				tc.what, 100*sumDirect/float64(n), 100*sumMetal/float64(n),
				sumMetal/sumDirect, sumRange/float64(n))
			_, _ = m8, m4
		})
	}
}

// relErrVec is ||a-b|| / ||b||.
func relErrVec(a, b []float32) float64 {
	var num, den float64
	for i := range b {
		d := float64(a[i] - b[i])
		num += d * d
		den += float64(b[i]) * float64(b[i])
	}
	if den == 0 {
		return 0
	}
	return math.Sqrt(num / den)
}

// rowRange is the ratio that drives the double-quant cost: how far the row's largest element
// (which sets the int8 scale for the WHOLE row) sits above a typical group's own maxabs.
// Near 1 ⇒ the int8 grid is fine everywhere and the first step is free. Large ⇒ quiet groups are
// crushed to a few int8 levels before int4 sees them.
func rowRange(row []float32) float64 {
	var rowMax float64
	var groupMaxes []float64
	for g := 0; g < len(row); g += int4GroupSize {
		end := g + int4GroupSize
		if end > len(row) {
			end = len(row)
		}
		var gm float64
		for _, v := range row[g:end] {
			if a := math.Abs(float64(v)); a > gm {
				gm = a
			}
		}
		groupMaxes = append(groupMaxes, gm)
		if gm > rowMax {
			rowMax = gm
		}
	}
	var sum float64
	for _, gm := range groupMaxes {
		sum += gm
	}
	mean := sum / float64(len(groupMaxes))
	if mean == 0 {
		return 1
	}
	return rowMax / mean
}
