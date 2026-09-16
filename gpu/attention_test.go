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

// TestRoPE_parityAtLongContextCeiling is N-85 (docs/audit-2026-09-10.md): TestRoPE_parity above
// only exercises pos=37, so the concern this measures — WGSL sin/cos range-reducing a large
// angle differently from the CPU's f64 reference — was flagged but never actually checked at a
// position a real served request can reach. decoder/fitplan.go's fit-by-default heuristic can
// grow context capacity to 65536 positions at int8 KV quant (the "i8 ceiling" the finding cites),
// so pos=65535 here is that real ceiling, not an arbitrary large number. theta at d=0 (invFreq≈1)
// lands at ~65535 rad, matching the finding's own "~6.5e4 rad" figure.
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
	// MEASUREMENT, not a pass/fail gate: N-85's own point is that this was UNMEASURED, and
	// reusing TestRoPE_parity's pos=37 tolerance here would silently assert a quality bar this
	// batch has no basis to pick — whether ~2.9e-3 maxAbs at the realistic long-context ceiling
	// (decoder/fitplan.go's int8-KV fit-by-default can reach 65536 positions) is ACCEPTABLE for
	// real model quality is a product decision, not something to decide by copy-pasting a
	// threshold calibrated for a 1700x shorter position. Recorded here so the real number is on
	// record instead of "unmeasured" — see docs/audit-2026-09-10.md's N-85 closure note.
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
