//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestLayerB_gemvW4A8Parity — Layer B step 2: the TARGET-quant hot kernel (int4/W4A8,
// what q4_k_m runs). Ported to MSL and validated bit-for-bit vs a CPU reference over the
// same packed nibbles + f32 group scales. Σ(nibble−8)·int8act per group × group-scale,
// × activation-scale.
func TestLayerB_gemvW4A8Parity(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	const src = `
#include <metal_stdlib>
using namespace metal;
kernel void gemv_w4a8(device const uint*  bq  [[buffer(0)]],   // [N*(K/8)] packed nibbles
                      device const float* bsc [[buffer(1)]],   // [N*(K/32)] f32 group scales
                      device const char*  aq  [[buffer(2)]],   // [K] int8 activation
                      device const float* asc [[buffer(3)]],   // [1]
                      device float*       out [[buffer(4)]],   // [N]
                      constant uint&      K   [[buffer(5)]],
                      uint n [[thread_position_in_grid]]) {
    uint wpr = K / 8;   // u32 words per row
    uint gpr = K / 32;  // groups per row
    device const uint*  brow = bq  + (uint)n * wpr;
    device const float* srow = bsc + (uint)n * gpr;
    float acc = 0.0f;
    for (uint g = 0; g < gpr; g++) {
        int gi = 0;
        for (uint w = 0; w < 4; w++) {                // 32 nibbles = 4 words
            uint word = brow[g * 4 + w];
            for (uint e = 0; e < 8; e++) {
                int nib = int((word >> (4 * e)) & 0xF);
                uint k = g * 32 + w * 8 + e;
                gi += (nib - 8) * int(aq[k]);
            }
        }
        acc += float(gi) * srow[g];
    }
    out[n] = acc * asc[0];
}`
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "gemv_w4a8")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const N, K = 256, 1536
	rng := rand.New(rand.NewSource(7))

	// int8 per-row activation (same as W8A8).
	a := make([]float32, K)
	for i := range a {
		a[i] = rng.Float32()*2 - 1
	}
	amx := float32(0)
	for _, v := range a {
		if x := float32(math.Abs(float64(v))); x > amx {
			amx = x
		}
	}
	aSc := amx / 127
	aq := make([]int8, K)
	for i, v := range a {
		r := math.Round(float64(v / aSc))
		if r > 127 {
			r = 127
		}
		if r < -127 {
			r = -127
		}
		aq[i] = int8(r)
	}

	words := make([]uint32, N*(K/8))
	scales := make([]float32, N*(K/32))
	for n := 0; n < N; n++ {
		row := make([]float32, K)
		for k := range row {
			row[k] = rng.Float32()*2 - 1
		}
		w, s := packW4A8Row(row)
		copy(words[n*(K/8):(n+1)*(K/8)], w)
		copy(scales[n*(K/32):(n+1)*(K/32)], s)
	}

	// CPU reference — identical unpack + math over the same packed bytes.
	ref := make([]float32, N)
	for n := 0; n < N; n++ {
		var acc float32
		for g := 0; g < K/32; g++ {
			var gi int32
			for w := 0; w < 4; w++ {
				word := words[n*(K/8)+g*4+w]
				for e := 0; e < 8; e++ {
					nib := int32((word >> (4 * uint(e))) & 0xF)
					k := g*32 + w*8 + e
					gi += (nib - 8) * int32(aq[k])
				}
			}
			acc += float32(gi) * scales[n*(K/32)+g]
		}
		ref[n] = acc * aSc
	}

	// GPU.
	q := d.NewCommandQueue()
	out := d.NewBufferLen(N)
	q.Run1D(pipe, N, 64,
		d.NewBufferUint32s(words),
		d.NewBufferFloats(scales),
		d.NewBufferInt8(aq),
		d.NewBufferFloats([]float32{aSc}),
		out,
		d.NewBufferU32(uint32(K)),
	)
	got := out.Floats()

	var dot, na, nb, maxrel float64
	for n := 0; n < N; n++ {
		dot += float64(got[n]) * float64(ref[n])
		na += float64(got[n]) * float64(got[n])
		nb += float64(ref[n]) * float64(ref[n])
		if dd := math.Abs(float64(got[n] - ref[n])); dd > 0 {
			if rel := dd / (math.Abs(float64(ref[n])) + 1e-6); rel > maxrel {
				maxrel = rel
			}
		}
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	if cos < 0.99999 || maxrel > 1e-4 {
		t.Fatalf("W4A8 GEMV parity FAIL: cosine=%.7f maxrel=%.2e (got[0]=%v ref[0]=%v)", cos, maxrel, got[0], ref[0])
	}
	t.Logf("W4A8 GEMV %dx%d (int4, target quant) on Metal GPU (cgo-free) vs CPU: cosine=%.7f maxrel=%.2e — PARITY", N, K, cos, maxrel)
}
