//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestLayerB_swigluQuantParity — Layer B: fused SwiGLU→int8-quant (WGSL `swigluQuantWGSL`
// analog): d[i] = silu(gate[i])·up[i] with silu(g)=g·σ(g), then per-row int8 quant. Single
// threadgroup, one maxabs reduction. Validated vs CPU.
func TestLayerB_swigluQuantParity(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	const src = `
#include <metal_stdlib>
using namespace metal;
kernel void swiglu_quant(device const float* g  [[buffer(0)]],  // [I] gate
                         device const float* u  [[buffer(1)]],  // [I] up
                         device char*        dq [[buffer(2)]],  // [I] int8
                         device float*       ds [[buffer(3)]],  // [1]
                         constant uint&      I  [[buffer(4)]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint tgs [[threads_per_threadgroup]]) {
    threadgroup float red[256];
    float mx = 0.0f;
    for (uint i = tid; i < I; i += tgs) {
        float s = (g[i] / (1.0f + exp(-g[i]))) * u[i];
        mx = max(mx, fabs(s));
    }
    red[tid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs/2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = max(red[tid], red[tid+s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sc = red[0]/127.0f;
    if (sc == 0.0f) sc = 1.0f;
    if (tid == 0) ds[0] = sc;
    float inv = 1.0f/sc;
    for (uint i = tid; i < I; i += tgs) {
        float s = (g[i] / (1.0f + exp(-g[i]))) * u[i];
        dq[i] = char(clamp(int(round(s*inv)), -127, 127));
    }
}`
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "swiglu_quant")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const I = 8960 // qwen2.5-1.5b ffn width
	rng := rand.New(rand.NewSource(5))
	g := make([]float32, I)
	u := make([]float32, I)
	for i := range g {
		g[i] = rng.Float32()*4 - 2
		u[i] = rng.Float32()*2 - 1
	}

	sv := make([]float32, I)
	var mx float32
	for i := range g {
		s := (g[i] / (1 + float32(math.Exp(-float64(g[i]))))) * u[i]
		sv[i] = s
		if a := float32(math.Abs(float64(s))); a > mx {
			mx = a
		}
	}
	refSc := mx / 127
	if refSc == 0 {
		refSc = 1
	}
	refQ := make([]int8, I)
	for i := range sv {
		q := min(int(math.Round(float64(sv[i]/refSc))), 127)
		if q < -127 {
			q = -127
		}
		refQ[i] = int8(q)
	}

	q := d.NewCommandQueue()
	dq := Buffer{id: d.id.Send(selNewBufferLen, uintptr(I), uintptr(0)), n: I}
	ds := d.NewBufferLen(1)
	q.Run1D(pipe, 256, 256, d.NewBufferFloats(g), d.NewBufferFloats(u), dq, ds, d.NewBufferU32(uint32(I)))
	gotQ := dq.Int8s()
	gotSc := ds.Floats()[0]

	mism := 0
	var dot, na, nb float64
	for i := range I {
		if gotQ[i] != refQ[i] {
			mism++
		}
		gg := float64(gotQ[i]) * float64(gotSc)
		rr := float64(refQ[i]) * float64(refSc)
		dot += gg * rr
		na += gg * gg
		nb += rr * rr
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	if cos < 0.99999 || mism > I/100 {
		t.Fatalf("swiglu+quant parity FAIL: cosine=%.7f int8-mismatch=%d/%d", cos, mism, I)
	}
	t.Logf("SwiGLU+quant I=%d on Metal GPU (cgo-free) vs CPU: cosine=%.7f, int8 exact=%d/%d — PARITY", I, cos, I-mism, I)
}
