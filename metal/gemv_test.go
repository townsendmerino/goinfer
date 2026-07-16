//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestLayerB_gemvW8A8Parity — Layer B step 1: the hot kernel (quantized GEMV) ported
// to MSL, validated bit-for-bit against a CPU reference on real per-row-int8-quantized
// data. This is goinfer's W8A8 math (int8×int8 → i32 accumulate → × activation-scale ×
// per-row weight-scale) — the same packing the CUDA arc used, now in MSL. Correctness
// first (the CUDA lesson: tuning — coalescing/ILP — comes after parity).
func TestLayerB_gemvW8A8Parity(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}

	// dst[N] = (a[K]·W[N,K]ᵀ) in int8, then × aScale × bScale[n]. Each thread = one row n.
	const src = `
#include <metal_stdlib>
using namespace metal;
kernel void gemv_w8a8(device const char*  aq  [[buffer(0)]],   // [K] int8 activation
                      device const float* asc [[buffer(1)]],   // [1] activation scale
                      device const char*  bq  [[buffer(2)]],   // [N*K] int8 weights (row-major)
                      device const float* bsc [[buffer(3)]],   // [N] per-row weight scale
                      device float*       out [[buffer(4)]],   // [N]
                      constant uint&      K   [[buffer(5)]],
                      uint n [[thread_position_in_grid]]) {
    int acc = 0;
    device const char* brow = bq + (uint)n * K;
    for (uint k = 0; k < K; k++) { acc += int(aq[k]) * int(brow[k]); }
    out[n] = float(acc) * asc[0] * bsc[n];
}`
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile gemv: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "gemv_w8a8")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const N, K = 256, 1536 // qwen2.5-1.5b-ish attention proj shape
	rng := rand.New(rand.NewSource(1))

	// per-row symmetric int8 quant (goinfer's W8A8): scale = maxabs/127.
	quantRow := func(row []float32) ([]int8, float32) {
		mx := float32(0)
		for _, v := range row {
			if a := float32(math.Abs(float64(v))); a > mx {
				mx = a
			}
		}
		sc := mx / 127
		if sc == 0 {
			sc = 1
		}
		q := make([]int8, len(row))
		for i, v := range row {
			r := math.Round(float64(v / sc))
			if r > 127 {
				r = 127
			}
			if r < -127 {
				r = -127
			}
			q[i] = int8(r)
		}
		return q, sc
	}

	a := make([]float32, K)
	for i := range a {
		a[i] = rng.Float32()*2 - 1
	}
	aq, aSc := quantRow(a)

	bq := make([]int8, N*K)
	bSc := make([]float32, N)
	for n := range N {
		row := make([]float32, K)
		for k := range row {
			row[k] = rng.Float32()*2 - 1
		}
		q, sc := quantRow(row)
		copy(bq[n*K:(n+1)*K], q)
		bSc[n] = sc
	}

	// CPU reference — identical integer math + f32 finalize.
	ref := make([]float32, N)
	for n := range N {
		var acc int32
		for k := range K {
			acc += int32(aq[k]) * int32(bq[n*K+k])
		}
		ref[n] = float32(acc) * aSc * bSc[n]
	}

	// GPU.
	q := d.NewCommandQueue()
	out := d.NewBufferLen(N)
	q.Run1D(pipe, N, 64,
		d.NewBufferInt8(aq),
		d.NewBufferFloats([]float32{aSc}),
		d.NewBufferInt8(bq),
		d.NewBufferFloats(bSc),
		out,
		d.NewBufferU32(uint32(K)),
	)
	got := out.Floats()

	// Compare — cosine + max relative diff (the finalize mul may differ by an f32 ULP).
	var dot, na, nb, maxrel float64
	for n := range N {
		dot += float64(got[n]) * float64(ref[n])
		na += float64(got[n]) * float64(got[n])
		nb += float64(ref[n]) * float64(ref[n])
		if d := math.Abs(float64(got[n] - ref[n])); d > 0 {
			rel := d / (math.Abs(float64(ref[n])) + 1e-6)
			if rel > maxrel {
				maxrel = rel
			}
		}
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	if cos < 0.99999 || maxrel > 1e-4 {
		t.Fatalf("W8A8 GEMV parity FAIL: cosine=%.7f maxrel=%.2e (got[0]=%v ref[0]=%v)", cos, maxrel, got[0], ref[0])
	}
	t.Logf("W8A8 GEMV %dx%d on Metal GPU (cgo-free) vs CPU: cosine=%.7f maxrel=%.2e — PARITY", N, K, cos, maxrel)
}
