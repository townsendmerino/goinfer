//go:build gpu

package gpu

import (
	"math"
	"testing"
)

// TestRoPE_parity checks the GPU RoPE against the CPU applyRoPE math (f64 ref).
func TestRoPE_parity(t *testing.T) {
	ctx, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer ctx.Close()

	const heads, hd, pos = 12, 128, 37
	half := hd / 2
	vec := randMat(heads*hd, 5)
	invFreq := make([]float32, half)
	for d := range invFreq {
		invFreq[d] = float32(1.0 / math.Pow(1e6, float64(2*d)/float64(hd))) // typical rope base
	}
	const scale = 1.0

	ref := append([]float32(nil), vec...)
	for d := range half {
		theta := float64(pos) * float64(invFreq[d])
		c := math.Cos(theta) * scale
		s := math.Sin(theta) * scale
		for h := range heads {
			off := h * hd
			x1, x2 := float64(vec[off+d]), float64(vec[off+half+d])
			ref[off+d] = float32(x1*c - x2*s)
			ref[off+half+d] = float32(x2*c + x1*s)
		}
	}
	got, err := ctx.RoPE(append([]float32(nil), vec...), heads, hd, pos, invFreq, scale)
	if err != nil {
		t.Fatalf("RoPE: %v", err)
	}
	cos, maxAbs := cosine(got, ref)
	t.Logf("RoPE parity: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
	if cos < 0.99999 || maxAbs > 1e-3 {
		t.Errorf("RoPE diverges: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
	}
}

// TestRoPE_parityAtLongContextCeiling measures the GPU RoPE at pos=65535, the context ceiling
// decoder/fitplan.go's fit-by-default can reach at int8 KV quant (TestRoPE_parity above only
// reaches pos=37). The concern is WGSL sin/cos range-reducing a ~6.5e4 rad angle differently from
// the CPU f64 reference. Background: docs/code-notes/gpu.md#TestRoPE_parityAtLongContextCeiling.
func TestRoPE_parityAtLongContextCeiling(t *testing.T) {
	ctx, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer ctx.Close()

	const heads, hd, pos = 12, 128, 65535
	half := hd / 2
	vec := randMat(heads*hd, 5)
	invFreq := make([]float32, half)
	for d := range invFreq {
		invFreq[d] = float32(1.0 / math.Pow(1e6, float64(2*d)/float64(hd))) // typical rope base
	}
	const scale = 1.0

	ref := append([]float32(nil), vec...)
	for d := range half {
		theta := float64(pos) * float64(invFreq[d])
		c := math.Cos(theta) * scale
		s := math.Sin(theta) * scale
		for h := range heads {
			off := h * hd
			x1, x2 := float64(vec[off+d]), float64(vec[off+half+d])
			ref[off+d] = float32(x1*c - x2*s)
			ref[off+half+d] = float32(x2*c + x1*s)
		}
	}
	got, err := ctx.RoPE(append([]float32(nil), vec...), heads, hd, pos, invFreq, scale)
	if err != nil {
		t.Fatalf("RoPE: %v", err)
	}
	cos, maxAbs := cosine(got, ref)
	t.Logf("RoPE parity at pos=%d (theta up to ~%.0f rad at d=0): cosine=%.8f maxAbs=%.3e "+
		"(TestRoPE_parity's pos=37 case measures cosine=1.00000000 maxAbs=3.1e-06 for comparison)",
		pos, float64(pos)*float64(invFreq[0]), cos, maxAbs)
	// A measurement, not a pass/fail gate: reusing TestRoPE_parity's pos=37 tolerance here would
	// assert a quality bar nobody chose, and whether the error at this ceiling is acceptable for
	// model quality is a product decision. The number is logged, not asserted. Record:
	// docs/code-notes/gpu.md#TestRoPE_parityAtLongContextCeiling.measurement.
}

// TestAttention_parity checks the GPU single-query attention against the CPU
// attendQuery math (f64 two-pass softmax ref), with GQA.
func TestAttention_parity(t *testing.T) {
	ctx, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer ctx.Close()

	const nH, nKV, hd, nKeys, start = 12, 2, 128, 40, 0
	kvDim := nKV * hd
	group := nH / nKV
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	q := randMat(nH*hd, 1)
	keys := randMat(nKeys*kvDim, 2)
	vals := randMat(nKeys*kvDim, 3)

	// CPU reference (mirrors attendQuery, f64).
	ref := make([]float32, nH*hd)
	for qh := range nH {
		kvh := qh / group
		maxS := math.Inf(-1)
		sc := make([]float64, nKeys)
		for s := range nKeys {
			var dot float64
			for d := range hd {
				dot += float64(q[qh*hd+d]) * float64(keys[s*kvDim+kvh*hd+d])
			}
			sc[s] = dot * float64(scale)
			if sc[s] > maxS {
				maxS = sc[s]
			}
		}
		var sum float64
		for s := range nKeys {
			sc[s] = math.Exp(sc[s] - maxS)
			sum += sc[s]
		}
		for s := range nKeys {
			w := sc[s] / sum
			for d := range hd {
				ref[qh*hd+d] += float32(w * float64(vals[s*kvDim+kvh*hd+d]))
			}
		}
	}
	got, err := ctx.Attention(q, keys, vals, nH, nKV, hd, nKeys, start, scale)
	if err != nil {
		t.Fatalf("Attention: %v", err)
	}
	cos, maxAbs := cosine(got, ref)
	t.Logf("Attention parity: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
	if cos < 0.9999 || maxAbs > 1e-3 {
		t.Errorf("Attention diverges: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
	}
}
