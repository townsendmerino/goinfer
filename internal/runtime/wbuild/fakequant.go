package wbuild

import (
	"math"

	"github.com/townsendmerino/aikit/linalg"
)

// FakeInt4WM fake-quantizes an f32 WeightMat with a 4-bit scheme (grouped along cols at
// Int4GroupSize, matching aikit's int4 + scripts/gemma4_quant_recon.py) then stores the
// reconstruction at int8 — W8A8 (int8 activations) by default, or Q8 (f32 activations) under
// GOINFER_FAKEQUANT_ACT=f32. The int8 re-quant is transparent (0.99995), so the 4-bit error
// survives; the activation precision is the only other variable.
//
// fakeQuantPerRow (set by tests) forces the group to the full row, one scale per output row (the per-row/IMMA
// granularity), so the probe can measure per-row vs per-group forward quality without a new kernel. Default off: the
// shipped 32-element grouping, so the fakequant-off invariant is untouched.
func (h *Hooks) FakeInt4WM(f32 []float32, rows, cols int, scheme string) linalg.WeightMat {
	group := Int4GroupSize
	if h.fakeQuantPerRow() {
		group = cols
	}
	fq := FakeQuantInt4(scheme, f32, rows, cols, group)
	return linalg.QuantizeInt8(fq, rows, cols, !h.fakeQuantF32Act())
}

// FakeQuantInt4 quantize→dequantizes a [rows, cols] f32 matrix with a 4-bit scheme,
// grouped along cols (input features) at `group`. Returns the reconstructed f32.
//
//	sym    — symmetric maxabs/7, codes [-7,7] (aikit's runtime int4; the control)
//	symmse — symmetric with an MSE-optimal per-group scale (grid over maxabs·[0.55,1])
//	affine — per-group zero-point over [min,max]/15, codes [0,15] (MLX's scheme)
func FakeQuantInt4(scheme string, w []float32, rows, cols, group int) []float32 {
	out := make([]float32, rows*cols)
	for r := range rows {
		row := w[r*cols : (r+1)*cols]
		dst := out[r*cols : (r+1)*cols]
		for gs := 0; gs < cols; gs += group {
			ge := min(gs+group, cols)
			fakeQuantGroup(scheme, row[gs:ge], dst[gs:ge])
		}
	}
	return out
}

func fakeQuantGroup(scheme string, g, dst []float32) {
	switch scheme {
	case "affine":
		lo, hi := g[0], g[0]
		for _, v := range g {
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
		s := float32(1)
		if hi > lo {
			s = (hi - lo) / 15
		}
		for i, v := range g {
			code := math.Max(0, math.Min(15, math.Round(float64((v-lo)/s))))
			dst[i] = float32(code)*s + lo
		}
	case "symmse":
		mx := maxAbs(g)
		bestS, bestErr := mx/7, float32(math.Inf(1))
		if mx == 0 {
			bestS = 1
		}
		for f := 0.55; f <= 1.0+1e-9; f += 0.025 {
			s := mx * float32(f) / 7
			if s <= 0 {
				continue
			}
			if e := symSSE(g, s); e < bestErr {
				bestErr, bestS = e, s
			}
		}
		symDequant(g, dst, bestS)
	default: // "sym" — aikit's runtime int4 (the control)
		mx := maxAbs(g)
		s := float32(1)
		if mx > 0 {
			s = mx / 7
		}
		symDequant(g, dst, s)
	}
}

func maxAbs(g []float32) float32 {
	var mx float32
	for _, v := range g {
		if a := float32(math.Abs(float64(v))); a > mx {
			mx = a
		}
	}
	return mx
}

func symDequant(g, dst []float32, s float32) {
	if s <= 0 {
		s = 1
	}
	for i, v := range g {
		q := math.Max(-7, math.Min(7, math.Round(float64(v/s))))
		dst[i] = float32(q) * s
	}
}

func symSSE(g []float32, s float32) float32 {
	var e float32
	for _, v := range g {
		q := math.Max(-7, math.Min(7, math.Round(float64(v/s))))
		d := float32(q)*s - v
		e += d * d
	}
	return e
}
