//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestLayerB_rmsnormQuantParity — Layer B step 3: the fused RMSNorm→int8-quant kernel
// (the WGSL `rmsnormQuantWGSL` analog) that produces the int8 activation + scale the
// GEMVs consume. Single-threadgroup, two threadgroup reductions (sum-of-squares, then
// maxabs) — the reduction pattern the attention/norm kernels all need. Validated vs a
// CPU reference: dequantized cosine ≈ 1 and near-exact int8 (a few ±1 from GPU rsqrt ULP
// at rounding boundaries are allowed).
func TestLayerB_rmsnormQuantParity(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	const src = `
#include <metal_stdlib>
using namespace metal;
kernel void rmsnorm_quant(device const float* x  [[buffer(0)]],  // [H]
                          device const float* w  [[buffer(1)]],  // [H] norm weight
                          device char*        aq [[buffer(2)]],  // [H] int8 out
                          device float*       asc[[buffer(3)]],  // [1] scale out
                          constant uint&      H  [[buffer(4)]],
                          constant float&     eps[[buffer(5)]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tgs [[threads_per_threadgroup]]) {
    threadgroup float red[256];
    float ss = 0.0f;
    for (uint i = tid; i < H; i += tgs) ss += x[i]*x[i];
    red[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs/2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = rsqrt(red[0]/float(H) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mx = 0.0f;
    for (uint i = tid; i < H; i += tgs) mx = max(mx, fabs(x[i]*rms*w[i]));
    red[tid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs/2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = max(red[tid], red[tid+s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sc = red[0]/127.0f;
    if (sc == 0.0f) sc = 1.0f;
    if (tid == 0) asc[0] = sc;
    float inv = 1.0f/sc;
    for (uint i = tid; i < H; i += tgs) {
        int q = int(round(x[i]*rms*w[i]*inv));
        aq[i] = char(clamp(q, -127, 127));
    }
}`
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "rmsnorm_quant")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const H = 1536
	const eps = 1e-6
	rng := rand.New(rand.NewSource(3))
	x := make([]float32, H)
	w := make([]float32, H)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
		w[i] = 0.5 + rng.Float32()
	}

	// CPU reference.
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	rms := float32(1 / math.Sqrt(ss/float64(H)+eps))
	y := make([]float32, H)
	var mx float32
	for i := range x {
		y[i] = x[i] * rms * w[i]
		if a := float32(math.Abs(float64(y[i]))); a > mx {
			mx = a
		}
	}
	refSc := mx / 127
	if refSc == 0 {
		refSc = 1
	}
	refQ := make([]int8, H)
	for i := range y {
		q := max(min(int(math.Round(float64(y[i]/refSc))), 127), -127)
		refQ[i] = int8(q)
	}

	// GPU (single threadgroup of 256).
	q := d.NewCommandQueue()
	aqBuf := d.NewBufferBytes(H) // H int8 bytes
	ascBuf := d.NewBufferLen(1)
	q.Run1D(pipe, 256, 256,
		d.NewBufferFloats(x),
		d.NewBufferFloats(w),
		aqBuf,
		ascBuf,
		d.NewBufferU32(uint32(H)),
		d.NewBufferFloats([]float32{float32(eps)}),
	)
	gotQ := aqBuf.Int8s()
	gotSc := ascBuf.Floats()[0]

	// Compare: scale close, dequantized cosine ≈ 1, near-exact int8.
	if rel := math.Abs(float64(gotSc-refSc)) / (float64(refSc) + 1e-9); rel > 1e-4 {
		t.Fatalf("scale mismatch: gpu=%v cpu=%v (rel %.2e)", gotSc, refSc, rel)
	}
	mism := 0
	var dot, na, nb float64
	for i := range H {
		if gotQ[i] != refQ[i] {
			mism++
		}
		g := float64(gotQ[i]) * float64(gotSc)
		r := float64(refQ[i]) * float64(refSc)
		dot += g * r
		na += g * g
		nb += r * r
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	mustFinite(t, "rmsnorm+quant cosine", cos)
	if cos < 0.99999 || mism > H/100 {
		t.Fatalf("rmsnorm+quant parity FAIL: cosine=%.7f int8-mismatch=%d/%d scale gpu=%v cpu=%v", cos, mism, H, gotSc, refSc)
	}
	t.Logf("RMSNorm+quant H=%d on Metal GPU (cgo-free) vs CPU: cosine=%.7f, int8 exact=%d/%d, scale rel=%.2e — PARITY",
		H, cos, H-mism, H, math.Abs(float64(gotSc-refSc))/(float64(refSc)+1e-9))
}
