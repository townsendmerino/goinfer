package decoder

import (
	"math"
	"os"

	"github.com/townsendmerino/aikit/linalg"
)

// fakeQuantScheme (GOINFER_FAKEQUANT=affine|sym|symmse) is a DIAGNOSTIC: when set, the
// int4 quantization path (quantizeWM's quantInt4 case) instead quantize→dequantizes each
// weight with the named 4-bit scheme and stores the result at int8 (W8A8). This isolates
// the 4-bit WEIGHT-scheme quality without any packed-nibble format, dequant routine, W4A8
// kernel, or .giw change — the int8 path is essentially transparent (0.99995 reconstruction),
// so the fake-int4 error survives to the forward. Lands ~26 GB (fits a 64 GB box). "sym" is
// the CONTROL: it must reproduce the real-int4 garbage, or the harness is lying — asserted
// bit-for-bit by TestFakeQuantSymMatchesRuntimeInt4. Not for prod.
//
// It built the int4-quality matrix in docs/task-gemma4-moe.md (sym/affine × int8/f32 act ×
// full/experts-only). The env is read ONCE here at package load; default (unset) is a strict
// no-op → every int4 consumer stays bit-identical (TestFakeQuantOffBitIdentical).
var (
	fakeQuantScheme      = os.Getenv("GOINFER_FAKEQUANT")
	fakeQuantF32Act      = os.Getenv("GOINFER_FAKEQUANT_ACT") == "f32"
	fakeQuantExpertsOnly = os.Getenv("GOINFER_FAKEQUANT_EXPERTS") == "1"
	fakeQuantPerRow      = os.Getenv("GOINFER_FAKEQUANT_PERROW") == "1" // §7 Phase 0b: one scale per row (test sets this directly)
)

// fakeQuantF32Act (GOINFER_FAKEQUANT_ACT=f32) stores the fake-quant reconstruction at
// weight-only int8 (Q8, f32 activations) instead of W8A8 (int8 activations) — so the probe
// can test a scheme against BOTH activation precisions. MLX's coherent config is affine
// weights + f16 activations, so affine+f32-act is the direct comparison.
//
// fakeQuantExpertsOnly (GOINFER_FAKEQUANT_EXPERTS=1) restricts the fake-quant to the MoE
// experts (streamExperts) — load at int8int8 so attention/dense/embed stay int8 and only
// the experts take the 4-bit scheme. Tests whether the experts alone tolerate 4-bit when
// the router's input hidden state is kept accurate (the ~14.5 GB config).

// fakeInt4WM fake-quantizes an f32 WeightMat with a 4-bit scheme (grouped along cols at
// int4GroupSize, matching aikit's int4 + scripts/gemma4_quant_recon.py) then stores the
// reconstruction at int8 — W8A8 (int8 activations) by default, or Q8 (f32 activations) under
// GOINFER_FAKEQUANT_ACT=f32. The int8 re-quant is transparent (0.99995), so the 4-bit error
// survives; the activation precision is the only other variable.
//
// fakeQuantPerRow (GOINFER_FAKEQUANT_PERROW=1, read fresh here so a test can set it around a load)
// forces the group to the FULL row — one scale per output row, the §7 per-row/IMMA granularity — so
// the probe can measure per-row vs per-group forward quality (Phase 0b) without a new kernel. Default
// off ⇒ the shipped 32-elem grouping, so the fakequant-off invariant is untouched.
func fakeInt4WM(f32 []float32, rows, cols int, scheme string) linalg.WeightMat {
	group := int4GroupSize
	if fakeQuantPerRow {
		group = cols
	}
	fq := fakeQuantInt4(scheme, f32, rows, cols, group)
	return linalg.QuantizeInt8(fq, rows, cols, !fakeQuantF32Act)
}

// fakeQuantInt4 quantize→dequantizes a [rows, cols] f32 matrix with a 4-bit scheme,
// grouped along cols (input features) at `group`. Returns the reconstructed f32.
//
//	sym    — symmetric maxabs/7, codes [-7,7] (aikit's runtime int4; the control)
//	symmse — symmetric with an MSE-optimal per-group scale (grid over maxabs·[0.55,1])
//	affine — per-group zero-point over [min,max]/15, codes [0,15] (MLX's scheme)
func fakeQuantInt4(scheme string, w []float32, rows, cols, group int) []float32 {
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
