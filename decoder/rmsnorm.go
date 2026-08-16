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
	for r := range rows {
		row := x[r*dim : r*dim+dim]
		var ss float64
		for _, v := range row {
			ss += float64(v) * float64(v)
		}
		inv := float32(1.0 / math.Sqrt(ss/float64(dim)+eps))
		if addOne {
			for i, v := range row {
				row[i] = (v * inv) * (1 + weight[i])
			}
		} else {
			for i, v := range row {
				row[i] = (v * inv) * weight[i]
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
	for r := range rows {
		row := x[r*dim : r*dim+dim]
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(dim)
		var variance float64
		for _, v := range row {
			d := float64(v) - mean
			variance += d * d
		}
		variance /= float64(dim)
		inv := 1.0 / math.Sqrt(variance+eps)
		if bias != nil {
			for i, v := range row {
				row[i] = float32((float64(v)-mean)*inv)*weight[i] + bias[i]
			}
		} else {
			for i, v := range row {
				row[i] = float32((float64(v) - mean) * inv * float64(weight[i]))
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
