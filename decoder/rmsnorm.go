package decoder

import "math"

// rmsNorm applies RMSNorm in place over each row of x ([rows, dim]).
//
//	rms      = sqrt(mean(x²) + eps)
//	x[i]     = (x[i] / rms) * scale[i]
//
// addOne selects the scale: Gemma stores weights as deviations from 1.0 and
// scales by (1 + weight); Llama/Qwen scale by weight directly. Using the wrong
// one silently zeroes/shifts every layer — it's a per-family knob
// (Architecture.RMSAddOne), one of the package's carry-over invariants (doc.go).
//
// Accumulate the sum of squares in float64 — the dim can be small (640) but
// the f32 round-off still matters at the ≥1−1e-4 parity bar, mirroring the
// float64-accumulation discipline embed/ and encoder/ already rely on.
func rmsNorm(x, weight []float32, rows, dim int, eps float64, addOne bool) {
	rmsNormInto(x, x, weight, rows, dim, eps, addOne)
}

// rmsNormInto applies RMSNorm from src into dst over each row ([rows, dim]).
// It supports dst == src (in place) as well as dst != src (out of place).
//
// The sum-of-squares reduction below is four parallel partial sums (ss0..ss3),
// not one sequential accumulator — this changes the floating-point addition
// order and is therefore NOT bit-identical to a strictly sequential sum,
// though it is within the ≥1−1e-4 parity bar every family test enforces.
func rmsNormInto(dst, src, weight []float32, rows, dim int, eps float64, addOne bool) {
	if dim <= 0 {
		return
	}
	for r := range rows {
		dstRow := dst[r*dim : (r+1)*dim]
		srcRow := src[r*dim : (r+1)*dim]
		w := weight[:dim]
		_ = dstRow[dim-1]
		_ = srcRow[dim-1]
		_ = w[dim-1]

		var ss0, ss1, ss2, ss3 float64
		i := 0
		for ; i+3 < len(srcRow); i += 4 {
			_ = srcRow[i+3]
			v0 := float64(srcRow[i])
			v1 := float64(srcRow[i+1])
			v2 := float64(srcRow[i+2])
			v3 := float64(srcRow[i+3])
			ss0 += v0 * v0
			ss1 += v1 * v1
			ss2 += v2 * v2
			ss3 += v3 * v3
		}
		var ssTail float64
		for ; i < len(srcRow); i++ {
			v := float64(srcRow[i])
			ssTail += v * v
		}
		ss := (ss0 + ss1) + (ss2 + ss3) + ssTail

		inv := float32(1.0 / math.Sqrt(ss/float64(dim)+eps))
		if addOne {
			i = 0
			for ; i+3 < len(srcRow); i += 4 {
				_ = dstRow[i+3]
				_ = srcRow[i+3]
				_ = w[i+3]
				dstRow[i] = (srcRow[i] * inv) * (1 + w[i])
				dstRow[i+1] = (srcRow[i+1] * inv) * (1 + w[i+1])
				dstRow[i+2] = (srcRow[i+2] * inv) * (1 + w[i+2])
				dstRow[i+3] = (srcRow[i+3] * inv) * (1 + w[i+3])
			}
			for ; i < len(srcRow); i++ {
				dstRow[i] = (srcRow[i] * inv) * (1 + w[i])
			}
		} else {
			i = 0
			for ; i+3 < len(srcRow); i += 4 {
				_ = dstRow[i+3]
				_ = srcRow[i+3]
				_ = w[i+3]
				dstRow[i] = (srcRow[i] * inv) * w[i]
				dstRow[i+1] = (srcRow[i+1] * inv) * w[i+1]
				dstRow[i+2] = (srcRow[i+2] * inv) * w[i+2]
				dstRow[i+3] = (srcRow[i+3] * inv) * w[i+3]
			}
			for ; i < len(srcRow); i++ {
				dstRow[i] = (srcRow[i] * inv) * w[i]
			}
		}
	}
}

// layerNorm applies LayerNorm in place over each row of x ([rows, dim]) — the
// GPT-2/NeoX/Cohere normalization. Unlike RMSNorm it subtracts the mean, and
// adds a bias when present:
//
//	mean = mean(x)
//	var  = mean((x-mean)²)
//	x[i] = (x[i]-mean)/sqrt(var+eps) * weight[i] (+ bias[i])
//
// bias is nil for bias-free LayerNorm families (Cohere/Command-R); GPT-2 passes
// its ln_1/ln_2 bias. Mean and variance accumulate in float64, matching
// rmsNorm's parity discipline.
func layerNorm(x, weight, bias []float32, rows, dim int, eps float64) {
	layerNormInto(x, x, weight, bias, rows, dim, eps)
}

// layerNormInto applies LayerNorm from src into dst over each row ([rows, dim]).
// It supports dst == src (in place) as well as dst != src (out of place).
//
// Mean and variance below use the same four-way parallel-partial-sum pattern
// as rmsNormInto's ss reduction (see its comment) — not bit-identical to a
// sequential sum, within tolerance.
func layerNormInto(dst, src, weight, bias []float32, rows, dim int, eps float64) {
	if dim <= 0 {
		return
	}
	for r := range rows {
		dstRow := dst[r*dim : (r+1)*dim]
		srcRow := src[r*dim : (r+1)*dim]
		w := weight[:dim]
		_ = dstRow[dim-1]
		_ = srcRow[dim-1]
		_ = w[dim-1]
		var b []float32
		if bias != nil {
			b = bias[:dim]
			_ = b[dim-1]
		}

		var m0, m1, m2, m3 float64
		i := 0
		for ; i+3 < len(srcRow); i += 4 {
			_ = srcRow[i+3]
			m0 += float64(srcRow[i])
			m1 += float64(srcRow[i+1])
			m2 += float64(srcRow[i+2])
			m3 += float64(srcRow[i+3])
		}
		var mTail float64
		for ; i < len(srcRow); i++ {
			mTail += float64(srcRow[i])
		}
		mean := ((m0 + m1) + (m2 + m3) + mTail) / float64(dim)

		var v0, v1, v2, v3 float64
		i = 0
		for ; i+3 < len(srcRow); i += 4 {
			_ = srcRow[i+3]
			d0 := float64(srcRow[i]) - mean
			d1 := float64(srcRow[i+1]) - mean
			d2 := float64(srcRow[i+2]) - mean
			d3 := float64(srcRow[i+3]) - mean
			v0 += d0 * d0
			v1 += d1 * d1
			v2 += d2 * d2
			v3 += d3 * d3
		}
		var vTail float64
		for ; i < len(srcRow); i++ {
			d := float64(srcRow[i]) - mean
			vTail += d * d
		}
		variance := ((v0 + v1) + (v2 + v3) + vTail) / float64(dim)
		inv := 1.0 / math.Sqrt(variance+eps)

		if b != nil {
			i = 0
			for ; i+3 < len(srcRow); i += 4 {
				_ = dstRow[i+3]
				_ = srcRow[i+3]
				_ = w[i+3]
				_ = b[i+3]
				dstRow[i] = float32((float64(srcRow[i])-mean)*inv)*w[i] + b[i]
				dstRow[i+1] = float32((float64(srcRow[i+1])-mean)*inv)*w[i+1] + b[i+1]
				dstRow[i+2] = float32((float64(srcRow[i+2])-mean)*inv)*w[i+2] + b[i+2]
				dstRow[i+3] = float32((float64(srcRow[i+3])-mean)*inv)*w[i+3] + b[i+3]
			}
			for ; i < len(srcRow); i++ {
				dstRow[i] = float32((float64(srcRow[i])-mean)*inv)*w[i] + b[i]
			}
		} else {
			i = 0
			for ; i+3 < len(srcRow); i += 4 {
				_ = dstRow[i+3]
				_ = srcRow[i+3]
				_ = w[i+3]
				dstRow[i] = float32((float64(srcRow[i]) - mean) * inv * float64(w[i]))
				dstRow[i+1] = float32((float64(srcRow[i+1]) - mean) * inv * float64(w[i+1]))
				dstRow[i+2] = float32((float64(srcRow[i+2]) - mean) * inv * float64(w[i+2]))
				dstRow[i+3] = float32((float64(srcRow[i+3]) - mean) * inv * float64(w[i+3]))
			}
			for ; i < len(srcRow); i++ {
				dstRow[i] = float32((float64(srcRow[i]) - mean) * inv * float64(w[i]))
			}
		}
	}
}

// silu is x·sigmoid(x), the SwiGLU activation Llama/Mistral/Qwen use (computed
// in float64 like geluTanh for parity). Gemma uses geluTanh; this is here for
// the SwiGLU families.
func silu(x float32) float32 {
	x64 := float64(x)
	return float32(x64 / (1 + math.Exp(-x64)))
}

// relu2 is ReLU-squared (relu(x)²), Nemotron-H's non-gated MLP activation.
func relu2(x float32) float32 {
	if x <= 0 {
		return 0
	}
	return x * x
}

// geluErf is the EXACT GELU — x·Φ(x) with the true Gaussian CDF, HF's "gelu"
// (ACT2FN["gelu"] = GELUActivation). It is a DIFFERENT FUNCTION from geluTanh, not a
// spelling of it: they differ by up to 4.73e-4 (worst at x ≈ -2.7).
//
// That gap is small — well under int8 quantization error — which is exactly why the
// conflation survived. goinfer previously accepted `activation_function: "gelu"` for GPT-2
// and ran geluTanh regardless, so a checkpoint asking for the exact function silently got
// the approximation. aikit hit the mirror image of this on its encoder side (three tanh
// names routed through erf) and fixed it in v1.19.0; this is the decoder-side counterpart.
//
// In float64 for the same reason geluTanh is: parity with the reference implementation.
func geluErf(x float32) float32 {
	v := float64(x)
	return float32(0.5 * v * (1 + math.Erf(v/math.Sqrt2)))
}

// geluTanh is the tanh-approximate GELU Gemma's GeGLU MLP uses
// ("gelu_pytorch_tanh"). Provided here so mlp.go (stub) and tests have the
// activation ready.
//
//	0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 x³) ))
func geluTanh(x float32) float32 {
	const c = 0.7978845608028654 // sqrt(2/π)
	x64 := float64(x)
	inner := c * (x64 + 0.044715*x64*x64*x64)
	return float32(0.5 * x64 * (1 + math.Tanh(inner)))
}
