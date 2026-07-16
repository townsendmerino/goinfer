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
	xb := d.NewBufferFloats(x)
	q.Run1D(pipe, nH*half, 64,
		xb, d.NewBufferFloats(invf),
		d.NewBufferU32(uint32(hd)), d.NewBufferU32(uint32(pos)), d.NewBufferU32(uint32(nH*half)),
	)
	got := xb.Floats()

	var dot, na, nb float64
	for i := range got {
		dot += float64(got[i]) * float64(ref[i])
		na += float64(got[i]) * float64(got[i])
		nb += float64(ref[i]) * float64(ref[i])
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	if cos < 0.9999999 {
		t.Fatalf("RoPE parity FAIL: cosine=%.9f", cos)
	}
	t.Logf("RoPE nH=%d hd=%d pos=%d on Metal GPU (cgo-free) vs CPU: cosine=%.9f — PARITY", nH, hd, pos, cos)
}
