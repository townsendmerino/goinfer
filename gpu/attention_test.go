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
	for d := 0; d < half; d++ {
		theta := float64(pos) * float64(invFreq[d])
		c := math.Cos(theta) * scale
		s := math.Sin(theta) * scale
		for h := 0; h < heads; h++ {
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
	for qh := 0; qh < nH; qh++ {
		kvh := qh / group
		maxS := math.Inf(-1)
		sc := make([]float64, nKeys)
		for s := start; s < nKeys; s++ {
			var dot float64
			for d := 0; d < hd; d++ {
				dot += float64(q[qh*hd+d]) * float64(keys[s*kvDim+kvh*hd+d])
			}
			sc[s] = dot * float64(scale)
			if sc[s] > maxS {
				maxS = sc[s]
			}
		}
		var sum float64
		for s := start; s < nKeys; s++ {
			sc[s] = math.Exp(sc[s] - maxS)
			sum += sc[s]
		}
		for s := start; s < nKeys; s++ {
			w := sc[s] / sum
			for d := 0; d < hd; d++ {
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
