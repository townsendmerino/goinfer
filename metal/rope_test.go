//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestLayerB_ropeParity — Layer B: RoPE (NeoX half-split, the dense Qwen2/Llama form).
// θ = pos·invFreq[d]; rotate the pair (x[d], x[d+half]) for d<half, per head. One thread
// per (head, d) pair. Validated vs CPU (cosine ≈ 1; sin/cos ULP only).
func TestLayerB_ropeParity(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	const src = `
#include <metal_stdlib>
using namespace metal;
kernel void rope(device float*        x    [[buffer(0)]],  // [nH*hd] in place
                 device const float*  invf [[buffer(1)]],  // [hd/2]
                 constant uint&       hd   [[buffer(2)]],
                 constant uint&       pos  [[buffer(3)]],
                 constant uint&       total[[buffer(4)]],   // nH*(hd/2)
                 uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    uint hlf = hd/2;
    uint head = gid / hlf;
    uint dd   = gid % hlf;
    uint base = head*hd;
    float theta = float(pos) * invf[dd];
    float c = cos(theta), s = sin(theta);
    float x0 = x[base+dd], x1 = x[base+hlf+dd];
    x[base+dd]     = x0*c - x1*s;
    x[base+hlf+dd] = x0*s + x1*c;
}`
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "rope")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const nH, hd, pos = 12, 128, 37
	half := hd / 2
	rng := rand.New(rand.NewSource(11))
	x := make([]float32, nH*hd)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	invf := make([]float32, half)
	for i := range invf {
		invf[i] = float32(1.0 / math.Pow(10000, float64(2*i)/float64(hd)))
	}

	ref := make([]float32, len(x))
	copy(ref, x)
	for head := range nH {
		for dd := range half {
			b := head * hd
			th := float64(pos) * float64(invf[dd])
			c, s := float32(math.Cos(th)), float32(math.Sin(th))
			x0, x1 := ref[b+dd], ref[b+half+dd]
			ref[b+dd] = x0*c - x1*s
			ref[b+half+dd] = x0*s + x1*c
		}
	}

	q := d.NewCommandQueue()
	xb := NewBufferFloats(d, x)
	q.Run1D(pipe, nH*half, 64,
		xb, NewBufferFloats(d, invf),
		NewBufferU32(d, uint32(hd)), NewBufferU32(d, uint32(pos)), NewBufferU32(d, uint32(nH*half)),
	)
	got := xb.Floats()

	var dot, na, nb float64
	for i := range got {
		dot += float64(got[i]) * float64(ref[i])
		na += float64(got[i]) * float64(got[i])
		nb += float64(ref[i]) * float64(ref[i])
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	mustFinite(t, "RoPE cosine", cos)
	if cos < 0.9999999 {
		t.Fatalf("RoPE parity FAIL: cosine=%.9f", cos)
	}
	t.Logf("RoPE nH=%d hd=%d pos=%d on Metal GPU (cgo-free) vs CPU: cosine=%.9f — PARITY", nH, hd, pos, cos)
}

// TestRope_mscale gates the shipped rope kernel's scale parameter (kernels.go, added for
// FeatRopeMscale — YaRN's attention_factor) against decoder/rope.go's applyRoPE, which applies
// it to cos/sin BEFORE the rotation (c := cos(theta)*scale; s := sin(theta)*scale), not to the
// rotated output afterward — those are numerically different for anything but a pure scalar
// multiple of the whole vector, so the test checks the exact per-component values, not just a
// cosine similarity that a wrong-but-close placement could still pass. Uses the ACTUAL kernels.go
// source (not an isolated copy), so a change to the shipped kernel is what this gates.
func TestRope_mscale(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "rope")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const nH, hd, pos = 4, 16, 11
	half := hd / 2
	rng := rand.New(rand.NewSource(23))
	invf := make([]float32, half)
	for i := range invf {
		invf[i] = float32(1.0 / math.Pow(10000, float64(2*i)/float64(hd)))
	}

	ref := func(x0 []float32, scale float32) []float32 {
		x := append([]float32(nil), x0...)
		for head := range nH {
			for dd := range half {
				b := head * hd
				th := float64(pos) * float64(invf[dd])
				c := float32(math.Cos(th)) * scale
				s := float32(math.Sin(th)) * scale
				a, bb := x[b+dd], x[b+half+dd]
				x[b+dd] = a*c - bb*s
				x[b+half+dd] = a*s + bb*c
			}
		}
		return x
	}

	q := d.NewCommandQueue()
	run := func(x0 []float32, scale float32) []float32 {
		xb := NewBufferFloats(d, x0)
		q.Run1D(pipe, nH*half, 64,
			xb, NewBufferFloats(d, invf),
			NewBufferU32(d, uint32(hd)), NewBufferU32(d, uint32(pos)), NewBufferU32(d, uint32(nH*half)),
			NewBufferU32(d, uint32(half)), NewBufferFloats(d, []float32{scale}))
		return xb.Floats()
	}

	x := make([]float32, nH*hd)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	cmp := func(name string, got, want []float32) {
		var worst float64
		for i := range want {
			if diff := math.Abs(float64(got[i] - want[i])); diff > worst {
				worst = diff
			}
		}
		t.Logf("%s: max|diff| = %.3e", name, worst)
		if worst > 1e-5 {
			t.Errorf("%s: max|diff| %.3e exceeds tolerance (got %v want %v)", name, worst, got, want)
		}
	}

	// scale=1.0 must be EXACTLY the pre-existing (no-scale) behaviour — every family but YaRN
	// dispatches this every layer, every token, so a no-op regression here is not hypothetical.
	cmp("scale=1 (no-op)", run(x, 1.0), ref(x, 1.0))

	// A real YaRN mscale (values HF's yarn_get_mscale typically produces, e.g. Mellum-class
	// long-context configs) must differ from the unscaled rotation, not just from itself.
	const realScale = float32(0.85)
	got := run(x, realScale)
	cmp("scale=0.85", got, ref(x, realScale))
	unscaled := ref(x, 1.0)
	var sameAsUnscaled = true
	for i := range got {
		if math.Abs(float64(got[i]-unscaled[i])) > 1e-4 {
			sameAsUnscaled = false
			break
		}
	}
	if sameAsUnscaled {
		t.Error("scale=0.85 output is indistinguishable from scale=1 — the scale parameter is not being applied")
	}
}
