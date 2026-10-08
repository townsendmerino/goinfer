//go:build darwin

package metal

import (
	"fmt"
	"math/rand"
	"os"
	"testing"
	"time"
)

// s17GemmVariants are the lever B design probe's candidate kernels (docs/tasks/task-multimodal-support-2026-10.md, "the
// f32-input kernel's design, amended"): kept so the probe that chose tower_gemm_w16's shape can be re-run.
const s17GemmVariants = `
// 2x4 fragments per simdgroup (16x32), 4 simdgroups as 2x2 -> 32x64 tile, direct device loads
kernel void gemm_f32_reg24(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], ushort sgid [[simdgroup_index_in_threadgroup]]) {
    const uint rm = tg.y * 32u + uint(sgid >> 1) * 16u, cn = tg.x * 64u + uint(sgid & 1) * 32u;
    simdgroup_float8x8 acc[2][4];
    for (uint i = 0; i < 2; i++) for (uint j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < K; k0 += 8u) {
        simdgroup_float8x8 a[2], b[4];
        for (uint i = 0; i < 2; i++) simdgroup_load(a[i], A + (rm + i * 8u) * K + k0, K);
        for (uint j = 0; j < 4; j++) simdgroup_load(b[j], B + (cn + j * 8u) * K + k0, K, ulong2(0, 0), true);
        for (uint i = 0; i < 2; i++) for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    for (uint i = 0; i < 2; i++) for (uint j = 0; j < 4; j++) simdgroup_store(acc[i][j], C + (rm + i * 8u) * N + cn + j * 8u, N);
}
// aikit's 16x16-per-simdgroup with 8 simdgroups as 4x2 -> 64x32 tile, direct loads
kernel void gemm_f32_sg8(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], ushort sgid [[simdgroup_index_in_threadgroup]]) {
    const uint rm = tg.y * 64u + uint(sgid >> 1) * 16u, cn = tg.x * 32u + uint(sgid & 1) * 16u;
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < K; k0 += 8u) {
        simdgroup_float8x8 a[2], b[2];
        for (uint i = 0; i < 2; i++) simdgroup_load(a[i], A + (rm + i * 8u) * K + k0, K);
        for (uint j = 0; j < 2; j++) simdgroup_load(b[j], B + (cn + j * 8u) * K + k0, K, ulong2(0, 0), true);
        for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) simdgroup_store(acc[i][j], C + (rm + i * 8u) * N + cn + j * 8u, N);
}
// staged: 64x64 tile, BK 32, float4 loads, 16 simdgroups (512 threads) of 16x16 (2x2 fragments)
kernel void gemm_f32_st16(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], ushort sgid [[simdgroup_index_in_threadgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float As[64 * 36];
    threadgroup float Bs[64 * 36];
    const uint m0 = tg.y * 64u, n0 = tg.x * 64u;
    const uint sr = uint(sgid >> 2) * 16u, sc = uint(sgid & 3) * 16u;
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < K; k0 += 32u) {
        { const uint r = tid >> 3, c4 = (tid & 7u) * 4u;
          *(threadgroup float4*)(As + r * 36u + c4) = *(device const float4*)(A + (m0 + r) * K + k0 + c4);
          *(threadgroup float4*)(Bs + r * 36u + c4) = *(device const float4*)(B + (n0 + r) * K + k0 + c4); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_float8x8 a[2], b[2];
            for (uint i = 0; i < 2; i++) simdgroup_load(a[i], As + (sr + i * 8u) * 36u + kk, 36);
            for (uint j = 0; j < 2; j++) simdgroup_load(b[j], Bs + (sc + j * 8u) * 36u + kk, 36, ulong2(0, 0), true);
            for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) simdgroup_store(acc[i][j], C + (m0 + sr + i * 8u) * N + n0 + sc + j * 8u, N);
}

kernel void gemm_f32_reg44(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], ushort sgid [[simdgroup_index_in_threadgroup]]) {
    const uint rm = tg.y * 64u + uint(sgid >> 1) * 32u, cn = tg.x * 64u + uint(sgid & 1) * 32u;
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < K; k0 += 8u) {
        simdgroup_float8x8 a[4], b[4];
        for (uint i = 0; i < 4; i++) simdgroup_load(a[i], A + (rm + i * 8u) * K + k0, K);
        for (uint j = 0; j < 4; j++) simdgroup_load(b[j], B + (cn + j * 8u) * K + k0, K, ulong2(0, 0), true);
        for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) simdgroup_store(acc[i][j], C + (rm + i * 8u) * N + cn + j * 8u, N);
}
`

// TestS17LeverB_kernelProbe is exploratory: a kernel-only direction read of lever B's candidate GEMM designs against
// aikit's gemm_f32_sg_big, one SigLIP-sized shape in one process (not a gate, not a record). GOINFER_S17_PROBE=1.
func TestS17LeverB_kernelProbe(t *testing.T) {
	if os.Getenv("GOINFER_S17_PROBE") != "1" {
		t.Skip("exploratory: set GOINFER_S17_PROBE=1")
	}
	d, _ := CreateSystemDefaultDevice()
	lib, err := d.CompileLibraryPrecise(eg2MSL+gvMSL+s17GemmVariants, MSL3_1)
	if err != nil {
		t.Fatal(err)
	}
	vit, err := d.NewViT()
	if err != nil {
		t.Fatal(err)
	}
	w16, _ := d.NewComputePipeline(lib, "tower_gemm_w16")
	w32, _ := d.NewComputePipeline(lib, "tower_gemm_w32")
	r44, _ := d.NewComputePipeline(lib, "gemm_f32_reg44")
	r24, _ := d.NewComputePipeline(lib, "gemm_f32_reg24")
	sg8, _ := d.NewComputePipeline(lib, "gemm_f32_sg8")
	st16, err := d.NewComputePipeline(lib, "gemm_f32_st16")
	if err != nil {
		t.Fatal(err)
	}
	q := d.NewCommandQueue()
	M, N, K := 4096, 4352, 1152
	rng := rand.New(rand.NewSource(1))
	a, w := make([]float32, M*K), make([]float32, N*K)
	h := make([]uint16, N*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	for i := range w {
		w[i] = float32(0.02 * rng.NormFloat64())
		h[i] = f32ToF16(w[i])
	}
	bA, bW, bH, bC := NewBufferFloats(d, a), NewBufferFloats(d, w), NewBufferU16s(d, h), d.NewBufferLen(M*N)
	u := func(v int) Buffer { return NewBufferU32(d, uint32(v)) }
	flops := 2 * float64(M) * float64(N) * float64(K)
	time1 := func(name string, f func(e *Encoder)) {
		for r := range 4 {
			e := q.Begin()
			t0 := time.Now()
			for range 5 {
				f(e)
			}
			e.End()
			s := time.Since(t0).Seconds() / 5
			if r > 0 {
				fmt.Printf("%-10s %.1f ms  %.2f TFLOPS\n", name, s*1e3, flops/s/1e12)
			}
		}
	}
	time1("aikit", func(e *Encoder) {
		p, gx, gy, tgx, tgy := vit.GEMMF32Plan(M, N, K)
		e.Dispatch2D(p, gx, gy, tgx, tgy, bA, bW, bC, u(M), u(N), u(K))
	})
	time1("reg24", func(e *Encoder) { e.Dispatch2D(r24, N/64, M/32, 128, 1, bA, bW, bC, u(M), u(N), u(K)) })
	time1("sg8", func(e *Encoder) { e.Dispatch2D(sg8, N/32, M/64, 256, 1, bA, bW, bC, u(M), u(N), u(K)) })
	time1("st16", func(e *Encoder) { e.Dispatch2D(st16, N/64, M/64, 512, 1, bA, bW, bC, u(M), u(N), u(K)) })
	_ = r44
	time1("w32", func(e *Encoder) {
		e.Dispatch2D(w32, (N+63)/64, (M+63)/64, 512, 1, bA, bW, bC, bW, u(M), u(N), u(K), u(0), u(0))
	})
	time1("w16", func(e *Encoder) {
		e.Dispatch2D(w16, (N+63)/64, (M+63)/64, 512, 1, bA, bH, bC, bW, u(M), u(N), u(K), u(0), u(0))
	})
	// SigLIP's MLP down shape, off every multiple: M 4096, N 1152, K 4304
	M2, N2, K2 := 4096, 1152, 4304
	fl2 := 2 * float64(M2) * float64(N2) * float64(K2)
	for _, c := range []struct {
		n string
		f func(e *Encoder)
	}{{"aikit K4304", func(e *Encoder) {
		p, gx, gy, tgx, tgy := vit.GEMMF32Plan(M2, N2, K2)
		e.Dispatch2D(p, gx, gy, tgx, tgy, bA, bW, bC, u(M2), u(N2), u(K2))
	}}, {"w16 K4304", func(e *Encoder) {
		e.Dispatch2D(w16, (N2+63)/64, (M2+63)/64, 512, 1, bA, bH, bC, bW, u(M2), u(N2), u(K2), u(0), u(0))
	}}} {
		e := q.Begin()
		c.f(e)
		e.End()
		e = q.Begin()
		t0 := time.Now()
		for range 5 {
			c.f(e)
		}
		e.End()
		sec := time.Since(t0).Seconds() / 5
		fmt.Printf("%-12s %.1f ms  %.2f TFLOPS\n", c.n, sec*1e3, fl2/sec/1e12)
	}
}
